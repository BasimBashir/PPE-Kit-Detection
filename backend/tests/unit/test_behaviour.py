"""Phase 4 behaviour assessors: fall (temporal sequence), stalling (stillness), phone, fight."""

import numpy as np

from app.behaviour.assessors import FallAssessor, IdleAssessor
from app.config.settings import BEHAVIOURS, BehaviourConfig, RuleThresholds
from app.ppe.state_machine import PPEStateMachine
from app.ppe.types import Outcome, PPEState, TrackedPerson
from app.streaming.frame import Frame

CFG = BehaviourConfig()
IMG = np.zeros((720, 1280, 3), np.uint8)
UPRIGHT_BOX, LYING_BOX = (450.0, 150.0, 550.0, 550.0), (350.0, 480.0, 650.0, 560.0)


def frame(t):
    return Frame("cam01", int(t * 10), IMG, None, t, t)


def torso_kps(sh, hip, conf=0.9):
    k = np.zeros((17, 3), np.float32)
    k[5], k[6] = (sh[0] - 20, sh[1], conf), (sh[0] + 20, sh[1], conf)
    k[11], k[12] = (hip[0] - 15, hip[1], conf), (hip[0] + 15, hip[1], conf)
    return k


UPRIGHT = (UPRIGHT_BOX, torso_kps((500, 200), (500, 350)))
MID = (UPRIGHT_BOX, torso_kps((440, 380), (520, 450)))  # ~49 deg: between upright and lying
LYING = (LYING_BOX, torso_kps((380, 520), (530, 520)))  # 90 deg, hips 170 px lower (> 0.25 * 400)
BENT = (UPRIGHT_BOX, torso_kps((380, 345), (500, 350)))  # torso horizontal, hips did not drop


def run(assessor, poses, dt=0.2, fsm=None):
    out = []
    for i, (box, kps) in enumerate(poses):
        t = i * dt
        (o,) = assessor.assess_deferred([TrackedPerson(1, box, 0.9, kps)], [], frame(t))
        trs = fsm.update([o], t) if fsm else []
        out.append((t, o, trs))
    return out


def test_fast_fall_then_lying_is_absent_and_confirms_after_hold():
    fsm = PPEStateMachine("cam01", BEHAVIOURS["fall"].policy, 3.0)
    res = run(FallAssessor(CFG, RuleThresholds()), [UPRIGHT] * 3 + [MID] + [LYING] * 20, fsm=fsm)
    assert [o.reason for _, o, _ in res[:4]] == ["upright"] * 3 + ["transition"]
    assert all(o.outcome is Outcome.ABSENT and o.reason == "down_after_fall" for _, o, _ in res[4:])
    confirmed = [t for t, _, trs in res for tr in trs if tr.new is PPEState.CONFIRMED_NON_COMPLIANT]
    assert confirmed and 3.8 - 1e-6 <= confirmed[0] <= 4.0 + 1e-6  # first ABSENT at 0.8 s + 3 s hold


def test_getting_up_ends_the_fall():
    res = run(FallAssessor(CFG, RuleThresholds()), [UPRIGHT] * 3 + [LYING] * 3 + [UPRIGHT])
    assert res[-1][1].outcome is Outcome.PRESENT and res[-1][1].reason == "upright"


def test_lying_down_slowly_never_starts_an_episode():
    fsm = PPEStateMachine("cam01", BEHAVIOURS["fall"].policy, 3.0)
    res = run(FallAssessor(CFG, RuleThresholds()), [UPRIGHT] + [MID] * 15 + [LYING] * 30, fsm=fsm)
    assert all(o.outcome is not Outcome.ABSENT for _, o, _ in res)
    assert res[-1][1].reason == "horizontal_without_fall"
    assert not any(tr.new is PPEState.CONFIRMED_NON_COMPLIANT for _, _, trs in res for tr in trs)


def test_bending_over_is_not_a_fall():
    res = run(FallAssessor(CFG, RuleThresholds()), [UPRIGHT] * 3 + [BENT] * 5)
    assert all(o.outcome is Outcome.PRESENT for _, o, _ in res)


def test_no_pose_is_uncertain_never_a_fall():
    (o,) = FallAssessor(CFG, RuleThresholds()).assess_deferred([TrackedPerson(1, UPRIGHT_BOX, 0.9, None)], [],
                                                                frame(0.0))
    assert o.outcome is Outcome.UNCERTAIN and o.reason == "not_assessable:torso_not_visible"


def test_small_or_low_confidence_person_is_not_assessed():
    a = FallAssessor(CFG, RuleThresholds())
    (o,) = a.assess_deferred([TrackedPerson(1, (0, 0, 20, 30), 0.9, UPRIGHT[1])], [], frame(0.0))
    assert o.reason == "not_assessable:person_too_small"
    (o,) = a.assess_deferred([TrackedPerson(2, UPRIGHT_BOX, 0.1, UPRIGHT[1])], [], frame(0.0))
    assert o.reason == "not_assessable:low_person_conf"


def body_kps(jitter=0.0, wrist_dx=0.0, rng=None):
    k = np.zeros((17, 3), np.float32)
    pts = {0: (460, 130), 5: (420, 160), 6: (500, 160), 7: (410, 240), 8: (510, 240), 9: (405 + wrist_dx, 300),
           10: (515, 300), 11: (430, 300), 12: (490, 300), 15: (440, 480), 16: (480, 480)}
    for i, (x, y) in pts.items():
        j = rng.uniform(-jitter, jitter, 2) if rng is not None else (0, 0)
        k[i] = (x + j[0], y + j[1], 0.9)
    return k


def idle_run(kps_at, seconds, dt=0.2, fsm=None):
    a, out = IdleAssessor(CFG, RuleThresholds()), []
    for i in range(int(seconds / dt) + 1):
        t = i * dt
        (o,) = a.assess_deferred([TrackedPerson(1, (400.0, 100.0, 520.0, 500.0), 0.9, kps_at(t))], [], frame(t))
        out.append((t, o, fsm.update([o], t) if fsm else []))
    return out


def test_still_person_is_absent_after_warmup():
    rng = np.random.default_rng(0)
    res = idle_run(lambda t: body_kps(jitter=1.0, rng=rng), 15)
    assert res[0][1].reason == "warming_up" and res[-1][1].outcome is Outcome.ABSENT
    assert res[-1][1].reason == "still"


def test_working_hands_count_as_moving():
    res = idle_run(lambda t: body_kps(wrist_dx=40 * np.sin(t * 3)), 15)
    assert res[-1][1].outcome is Outcome.PRESENT and res[-1][1].reason == "moving"


def test_stalling_confirms_after_the_rules_idle_time():
    policy = BEHAVIOURS["stalling"].policy.model_copy(update={"min_duration_s": 20.0})
    fsm = PPEStateMachine("cam01", policy, 3.0)
    res = idle_run(lambda t: body_kps(), 40, fsm=fsm)
    confirmed = [t for t, _, trs in res for tr in trs if tr.new is PPEState.CONFIRMED_NON_COMPLIANT]
    assert confirmed and 28.0 <= confirmed[0] <= 30.0  # ~8 s warm-up + 20 s idle
