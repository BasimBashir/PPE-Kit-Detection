"""Unit tests: geometry, head ROI, association hard negatives, zones, PPE state machine (spec 19.1)."""

import numpy as np
import pytest
from pydantic import ValidationError

from app.config.settings import AssociationConfig, RuleThresholds, TemporalPolicy
from app.inference.types import Detection
from app.ppe.association import Associator
from app.ppe.geometry import area, intersection, ioa, iou, point_in_polygon, valid_box
from app.ppe.head_roi import head_roi
from app.ppe.state_machine import PPEStateMachine
from app.ppe.types import Observation, Outcome, PPEState, TrackedPerson
from app.ppe.zones import ZoneRuntime, apply_privacy_masks, filter_ignored, zone_of
from app.streaming.frame import Frame

S = PPEState
W, H = 1280, 720


def frame(img=None):
    return Frame("cam01", 1, img if img is not None else np.zeros((H, W, 3), np.uint8), None, 0.0, 0.0)


def det(cls, box, conf=0.9):
    return Detection(cls, conf, box, "test")


def person(tid, box, conf=0.9, kps=None):
    return TrackedPerson(tid, box, conf, kps)


GEO = AssociationConfig(head_roi="geometry", use_classifier=False)
TH = RuleThresholds()
P1 = (400.0, 100.0, 520.0, 500.0)  # 120 x 400 person; geometry ROI x 418..502, y 76..200


# ---------------------------------------------------------------- geometry
def test_geometry_basics():
    a, b = (0, 0, 10, 10), (5, 5, 15, 15)
    assert area(a) == 100 and intersection(a, b) == 25
    assert iou(a, b) == pytest.approx(25 / 175)
    assert ioa(b, a) == pytest.approx(0.25)
    assert iou(a, (20, 20, 30, 30)) == 0
    assert not valid_box((5, 5, 5, 10)) and not valid_box((0, 0, float("nan"), 1)) and valid_box(a)
    sq = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert point_in_polygon((0.5, 0.5), sq) and not point_in_polygon((1.5, 0.5), sq)


# ---------------------------------------------------------------- head ROI
def test_geometry_roi():
    roi, src = head_roi(person(1, P1), GEO, W, H)
    assert src == "geometry"
    assert roi == pytest.approx((418.0, 76.0, 502.0, 200.0))


def kps_with_head(cx=460.0, cy=130.0, conf=0.9):
    k = np.zeros((17, 3), np.float32)
    k[0] = (cx, cy, conf)          # nose
    k[1] = (cx - 8, cy - 6, conf)  # eyes
    k[2] = (cx + 8, cy - 6, conf)
    k[5] = (cx - 40, cy + 60, conf)  # shoulders
    k[6] = (cx + 40, cy + 60, conf)
    return k


def test_keypoint_roi_and_fallback():
    cfg = AssociationConfig(head_roi="keypoints", use_classifier=False)
    roi, src = head_roi(person(1, P1, kps=kps_with_head()), cfg, W, H)
    assert src == "keypoints"
    assert roi[0] < 460 < roi[2] and roi[1] < 130 - 6 < roi[3]
    _, src_low = head_roi(person(1, P1, kps=kps_with_head(conf=0.2)), cfg, W, H)
    assert src_low == "geometry"
    roi_clip, _ = head_roi(person(1, (0, 0, 60, 200), kps=kps_with_head(5, 5)), cfg, W, H)
    assert roi_clip[0] >= 0 and roi_clip[1] >= 0


# ---------------------------------------------------------------- association
def assess(persons, dets, cfg=GEO, th=TH, classifier=None):
    return Associator(cfg, th, classifier).assess(persons, dets, frame())


def test_helmet_on_head():
    (o,) = assess([person(1, P1)], [det("helmet", (430, 85, 490, 125))])
    assert o.outcome is Outcome.HELMET and o.reason == "helmet_on_head_roi"


def test_bare_head():
    (o,) = assess([person(1, P1)], [det("head", (435, 95, 485, 150))])
    assert o.outcome is Outcome.NO_HELMET


def test_helmet_on_ground_not_credited():
    (o,) = assess([person(1, P1)], [det("helmet", (530, 470, 580, 500)), det("head", (435, 95, 485, 150))])
    assert o.outcome is Outcome.NO_HELMET and o.helmet_box is None


def test_helmet_in_hand_at_chest_not_credited():
    (o,) = assess([person(1, P1)], [det("helmet", (440, 260, 495, 300))])
    assert o.outcome is Outcome.UNCERTAIN and o.helmet_box is None


def test_one_helmet_never_credited_to_two_people():
    p_a, p_b = person(1, (400, 100, 520, 500)), person(2, (470, 110, 590, 510))
    # helmet sits on A's head ROI centre; it also overlaps B's ROI partially
    obs = assess([p_a, p_b], [det("helmet", (432, 85, 488, 125)), det("head", (505, 110, 555, 160))])
    by = {o.track_id: o for o in obs}
    assert by[1].outcome is Outcome.HELMET
    assert by[2].outcome is Outcome.NO_HELMET and by[2].helmet_box is None


def test_occluded_or_tiny_or_truncated_is_uncertain():
    th = RuleThresholds(min_person_height_px=60)
    (tiny,) = assess([person(1, (10, 300, 30, 340))], [], th=th)
    (top,) = assess([person(2, (400, 0, 520, 400))], [det("head", (435, 0, 485, 40))])
    (nothing,) = assess([person(3, P1)], [])
    assert tiny.outcome is Outcome.UNCERTAIN and "person_too_small" in tiny.reason
    assert top.outcome is Outcome.UNCERTAIN and "head_truncated" in top.reason
    assert nothing.outcome is Outcome.UNCERTAIN and nothing.reason == "no_head_evidence"


def test_threshold_boundaries():
    th = RuleThresholds(helmet_conf=0.5, head_conf=0.5)
    (at,) = assess([person(1, P1)], [det("helmet", (430, 85, 490, 125), conf=0.5)], th=th)
    (below,) = assess([person(1, P1)], [det("helmet", (430, 85, 490, 125), conf=0.4999)], th=th)
    assert at.outcome is Outcome.HELMET and below.outcome is Outcome.UNCERTAIN
    (lowp,) = assess([person(1, P1, conf=0.39)], [det("head", (435, 95, 485, 150))])
    assert lowp.outcome is Outcome.UNCERTAIN and "low_person_conf" in lowp.reason


class FakeClassifier:
    version = "fake"

    def __init__(self, label):
        self.label, self.calls = label, 0

    def classify(self, crops):
        self.calls += len(crops)
        assert all(c.ndim == 3 and c.size > 0 for c in crops)
        return [(self.label, 0.93)] * len(crops)


def test_conflict_goes_to_classifier_only_when_ambiguous():
    cfg = AssociationConfig(head_roi="geometry", use_classifier=True)
    clf = FakeClassifier("HELMET_ABSENT")
    both = [det("helmet", (430, 85, 490, 125), 0.6), det("head", (435, 95, 485, 150), 0.62)]
    (o,) = assess([person(1, P1)], both, cfg=cfg, classifier=clf)
    assert o.outcome is Outcome.NO_HELMET and o.classifier_used and clf.calls == 1
    (o2,) = assess([person(1, P1)], [det("helmet", (430, 85, 490, 125))], cfg=cfg, classifier=clf)
    assert not o2.classifier_used and clf.calls == 1  # clear case: classifier not invoked
    (o3,) = assess([person(1, P1)], both, cfg=GEO)
    assert o3.outcome is Outcome.UNCERTAIN and o3.reason == "conflicting_evidence"
    (o4,) = assess([person(1, P1)], both, cfg=cfg, classifier=FakeClassifier("UNCERTAIN"))
    assert o4.outcome is Outcome.UNCERTAIN


# ---------------------------------------------------------------- zones
def test_zones():
    zones = [
        ZoneRuntime("z1", "cam01", "deck", "monitor", [(0, 0), (0.5, 0), (0.5, 1), (0, 1)]),
        ZoneRuntime("ign", "cam01", "office", "ignore", [(0.5, 0), (1, 0), (1, 1), (0.5, 1)]),
        ZoneRuntime("prv", "cam01", "window", "privacy", [(0, 0), (0.1, 0), (0.1, 0.1), (0, 0.1)]),
    ]
    left, right = person(1, (100, 100, 200, 500)), person(2, (900, 100, 1000, 500))
    assert zone_of(left.box, zones, W, H) == "z1" and zone_of(right.box, zones, W, H) is None
    assert [p.track_id for p in filter_ignored([left, right], zones, W, H)] == [1]
    img = np.random.default_rng(0).integers(0, 255, (H, W, 3), dtype=np.uint8)
    out = apply_privacy_masks(img, zones)
    assert out is not img and np.array_equal(out[600:, 600:], img[600:, 600:])
    assert not np.array_equal(out[:60, :100], img[:60, :100])


# ---------------------------------------------------------------- state machine
def ob(outcome, tid=1, conf=0.8):
    return Observation(tid, outcome, conf, "t", (0, 0, 10, 10), None)


def feed(fsm, seq, t0=0.0, dt=0.2, tid=1):
    trans, t = [], t0
    for o in seq:
        trans += fsm.update([ob(o, tid)], t)
        t += dt
    return trans, t


N, Y, U = Outcome.NO_HELMET, Outcome.HELMET, Outcome.UNCERTAIN
POL = TemporalPolicy()  # 12 of 15, suspect 2, compliant 3/5, clear 8/10


def test_confirm_exactly_at_n_of_m():
    fsm = PPEStateMachine("cam01", POL, 3.0)
    trans, _ = feed(fsm, [N] * 11)
    assert [t.new for t in trans] == [S.SUSPECTED_NON_COMPLIANT] and fsm.state_of(1) is S.SUSPECTED_NON_COMPLIANT
    trans, _ = feed(fsm, [N], t0=2.2)
    assert trans[0].new is S.CONFIRMED_NON_COMPLIANT and trans[0].episode_seq == 1


def test_single_miss_does_not_flap_and_uncertain_never_votes():
    fsm = PPEStateMachine("cam01", POL, 3.0)
    feed(fsm, [N] * 6 + [Y] + [U] * 5 + [N] * 6)
    assert fsm.state_of(1) is S.CONFIRMED_NON_COMPLIANT  # 12 NO_HELMET of last 13 decisive; U ignored
    trans, _ = feed(fsm, [Y, U, U, N, Y], t0=10)
    assert trans == [] and fsm.state_of(1) is S.CONFIRMED_NON_COMPLIANT


def test_suspected_to_unknown_on_visibility_loss_keeps_episode():
    fsm = PPEStateMachine("cam01", POL, 3.0)
    feed(fsm, [N, N] + [U] * 8)
    assert fsm.state_of(1) is S.UNKNOWN
    trans, _ = feed(fsm, [N, N], t0=5)
    assert trans[-1].new is S.SUSPECTED_NON_COMPLIANT and trans[-1].episode_seq == 1


def test_clear_with_hysteresis_and_new_episode():
    fsm = PPEStateMachine("cam01", POL, 3.0)
    feed(fsm, [N] * 12)
    trans, _ = feed(fsm, [Y] * 7, t0=5)
    assert trans == []
    trans, _ = feed(fsm, [Y], t0=7)
    assert trans[0].new is S.CLEARED
    trans, _ = feed(fsm, [Y], t0=8)
    assert trans[0].new is S.COMPLIANT
    trans, _ = feed(fsm, [N, N], t0=9)
    assert trans[-1].new is S.SUSPECTED_NON_COMPLIANT and trans[-1].episode_seq == 2


def test_compliant_path():
    fsm = PPEStateMachine("cam01", POL, 3.0)
    trans, _ = feed(fsm, [Y, Y, Y])
    assert [t.new for t in trans] == [S.COMPLIANT]


def test_track_expiry():
    fsm = PPEStateMachine("cam01", POL, track_expiry_s=3.0)
    feed(fsm, [N] * 12, tid=1)
    feed(fsm, [Y] * 3, tid=2)
    assert fsm.expire(4.0) == []  # last seen 2.2 / 0.4 -> 1 still alive at 4.0? 4.0-2.2 = 1.8 < 3
    trans = fsm.expire(6.0)
    assert [(t.track_id, t.new, t.reason) for t in trans] == [(1, S.CLEARED, "track_lost")]
    assert fsm.tracks == {}


def test_duration_mode():
    pol = TemporalPolicy(mode="duration", min_duration_s=2.0)
    fsm = PPEStateMachine("cam01", pol, 3.0)
    feed(fsm, [N] * 10, dt=0.2)  # t = 0 .. 1.8 s
    assert fsm.state_of(1) is S.SUSPECTED_NON_COMPLIANT
    trans, _ = feed(fsm, [N], t0=2.0)
    assert trans[0].new is S.CONFIRMED_NON_COMPLIANT


def test_both_mode_needs_count_and_duration():
    pol = TemporalPolicy(mode="both", min_duration_s=5.0)
    fsm = PPEStateMachine("cam01", pol, 10.0)
    feed(fsm, [N] * 14, dt=0.2)  # count satisfied, only 2.6 s elapsed
    assert fsm.state_of(1) is S.SUSPECTED_NON_COMPLIANT
    feed(fsm, [N] * 2, t0=5.0)
    assert fsm.state_of(1) is S.CONFIRMED_NON_COMPLIANT


def test_policy_validation():
    with pytest.raises(ValidationError):
        TemporalPolicy(confirm_n=16, confirm_m=15)
    with pytest.raises(ValidationError):
        TemporalPolicy(window=10)


def test_deferred_classification_matches_direct_path():
    from app.ppe.association import apply_classification

    cfg = AssociationConfig(head_roi="geometry", use_classifier=True)
    both = [det("helmet", (430, 85, 490, 125), 0.6), det("head", (435, 95, 485, 150), 0.62)]
    a = Associator(cfg, TH, FakeClassifier("HELMET_ABSENT"))
    (o,) = a.assess_deferred([person(1, P1)], both, frame())
    assert o.outcome is Outcome.UNCERTAIN and not o.classifier_used
    pending = a.take_pending()
    assert len(pending) == 1 and a.take_pending() == []
    apply_classification(pending, [("HELMET_ABSENT", 0.9)])
    (direct,) = Associator(cfg, TH, FakeClassifier("HELMET_ABSENT")).assess([person(1, P1)], both, frame())
    assert (o.outcome, o.classifier_used) == (direct.outcome, direct.classifier_used) == (Outcome.NO_HELMET, True)
