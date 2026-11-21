"""Phase 4 fight check: order-invariant pair features, fixed-rate pair sequences, model round trip, assessor."""

import numpy as np
import torch

from app.behaviour.fight import FightAssessor, FightGRU, FightModel, PairSequencer, pair_features
from app.config.settings import BehaviourConfig, RuleThresholds
from app.ppe.types import Outcome, TrackedPerson
from app.streaming.frame import Frame

IMG = np.zeros((720, 1280, 3), np.uint8)
A, B, FAR = (400.0, 100.0, 520.0, 500.0), (500.0, 100.0, 620.0, 500.0), (1000.0, 100.0, 1120.0, 500.0)


def kps(dx=0.0, conf=0.9):
    k = np.zeros((17, 3), np.float32)
    k[:, 0], k[:, 1], k[:, 2] = np.linspace(410, 510, 17) + dx, np.linspace(120, 480, 17), conf
    return k


class Stub:
    def __init__(self, prob, seq_len=4, hz=5.0):
        self.prob, self.seq_len, self.hz, self.version = prob, seq_len, hz, "stub"

    def score(self, x):
        assert x.shape[1:] == (self.seq_len, 102)
        return np.full(len(x), self.prob, np.float32)


def run(assessor, people, seconds=1.2, dt=0.2):
    out = []
    for i in range(int(seconds / dt) + 1):
        t = i * dt
        persons = [TrackedPerson(tid, box, 0.9, kps(box[0] - 400)) for tid, box in people]
        out = assessor.assess_deferred(persons, [], Frame("cam01", i, IMG, None, t, t))
    return out


def test_pair_features_are_order_invariant_and_drop_low_confidence_points():
    f1 = pair_features(kps(), kps(100), A, B)
    assert f1.shape == (102,) and np.allclose(f1, pair_features(kps(100), kps(), B, A))
    low = pair_features(kps(conf=0.1), kps(100), A, B)
    assert np.all(low[:51] == 0)  # left person's points below min_conf are zeroed


def test_sequencer_needs_a_full_window_and_a_nearby_partner():
    sq = PairSequencer(seq_len=4, hz=5.0, pair_dist=1.2)
    for i in range(3):
        near, seqs = sq.update([TrackedPerson(1, A, 0.9, kps()), TrackedPerson(2, B, 0.9, kps(100))], i * 0.2)
        assert near == {1, 2} and seqs == []  # 3 samples < seq_len
    near, seqs = sq.update([TrackedPerson(1, A, 0.9, kps()), TrackedPerson(2, B, 0.9, kps(100))], 0.6)
    assert [p for p, _ in seqs] == [(1, 2)] and seqs[0][1].shape == (4, 102)
    near, seqs = sq.update([TrackedPerson(1, A, 0.9, kps()), TrackedPerson(3, FAR, 0.9, kps(600))], 0.8)
    assert near == set() and seqs == []


def test_assessor_outcomes():
    cfg, t = BehaviourConfig(), RuleThresholds(absent_conf=0.7)
    hot = run(FightAssessor(cfg, t, Stub(0.9)), [(1, A), (2, B)])
    assert [o.outcome for o in hot] == [Outcome.ABSENT] * 2 and hot[0].reason == "fight_score"
    calm = run(FightAssessor(cfg, t, Stub(0.1)), [(1, A), (2, B)])
    assert {o.reason for o in calm} == {"pair_normal"}
    alone = run(FightAssessor(cfg, t, Stub(0.9)), [(1, A), (3, FAR)])
    assert {o.outcome for o in alone} == {Outcome.PRESENT} and {o.reason for o in alone} == {"no_one_nearby"}
    early = run(FightAssessor(cfg, t, Stub(0.9)), [(1, A), (2, B)], seconds=0.2)
    assert {o.reason for o in early} == {"warming_up"}
    none = run(FightAssessor(cfg, t, None), [(1, A), (2, B)])
    assert {o.reason for o in none} == {"fight_model_missing"} and none[0].outcome is Outcome.UNCERTAIN


def test_model_checkpoint_round_trip(tmp_path):
    p = tmp_path / "fight.pt"
    torch.save({"state_dict": FightGRU(8).state_dict(), "hidden": 8, "seq_len": 4, "hz": 5.0}, p)
    m = FightModel(p)
    probs = m.score(np.zeros((2, 4, 102), np.float32))
    assert m.seq_len == 4 and m.hz == 5.0 and probs.shape == (2,) and np.all((probs > 0) & (probs < 1))
