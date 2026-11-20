"""Phase 3 unit tests: body-region association for vest / gloves / goggles / shoes, rule config, teacher questions."""

import numpy as np
import pytest
from pydantic import ValidationError

from app.config.settings import AssociationConfig, RuleSeed, RuleThresholds
from app.inference.teacher import instructions
from app.inference.types import Detection
from app.ppe.association import ITEMS, Associator
from app.ppe.types import Outcome, TrackedPerson
from app.streaming.frame import Frame

W, H = 1280, 720
P1 = (400.0, 100.0, 520.0, 500.0)  # 120 x 400 person
CFG = AssociationConfig(use_classifier=False)


def body_kps(conf=0.9, wrists=True):
    k = np.zeros((17, 3), np.float32)
    k[0], k[1], k[2] = (460, 130, conf), (452, 124, conf), (468, 124, conf)  # nose, eyes
    k[5], k[6] = (420, 160, conf), (500, 160, conf)  # shoulders
    k[7], k[8] = (410, 240, conf), (510, 240, conf)  # elbows
    if wrists:
        k[9], k[10] = (405, 300, conf), (515, 300, conf)  # wrists -> hand ROIs ~(375,295,428,347) / (492,295,545,347)
    k[11], k[12] = (430, 300, conf), (490, 300, conf)  # hips -> torso ROI ~(404,144,516,324)
    k[15], k[16] = (440, 480, conf), (480, 480, conf)  # ankles
    return k


def assess(item, dets, box=P1, kps=None):
    p = TrackedPerson(1, box, 0.9, body_kps() if kps is None else kps)
    frame = Frame("cam01", 1, np.zeros((H, W, 3), np.uint8), None, 0.0, 0.0)
    detections = [Detection(c, 0.9, b, "t") for c, b in dets]
    (o,) = Associator(CFG, RuleThresholds(), None, item).assess([p], detections, frame)
    return o


def test_gloves_both_hands_gloved_is_present():
    o = assess("gloves", [("gloves", (385, 305, 420, 340)), ("gloves", (500, 305, 535, 340))])
    assert o.outcome is Outcome.PRESENT and o.item_roi is not None


def test_one_bare_hand_is_a_violation():
    o = assess("gloves", [("gloves", (385, 305, 420, 340)), ("no_gloves", (500, 305, 535, 340))])
    assert o.outcome is Outcome.ABSENT and o.absent_box == (500, 305, 535, 340)


def test_gloves_without_wrist_keypoints_is_uncertain_never_a_violation():
    o = assess("gloves", [("no_gloves", (500, 305, 535, 340))], kps=body_kps(wrists=False))
    assert o.outcome is Outcome.UNCERTAIN and o.reason == "not_assessable:region_not_visible"


def test_vest_on_torso_credited_vest_on_ground_not():
    assert assess("safety_vest", [("vest", (410, 150, 510, 320))]).outcome is Outcome.PRESENT
    o = assess("safety_vest", [("vest", (530, 470, 620, 500))])
    assert o.outcome is Outcome.UNCERTAIN and o.present_box is None


def test_bare_torso_is_absent():
    assert assess("safety_vest", [("no_vest", (410, 150, 510, 320))]).outcome is Outcome.ABSENT


def test_detections_of_other_items_are_ignored():
    o = assess("safety_vest", [("helmet", (430, 85, 490, 125)), ("no_gloves", (500, 305, 535, 340))])
    assert o.outcome is Outcome.UNCERTAIN


def test_shoes_on_feet_and_feet_cut_off_by_frame():
    assert assess("safety_shoes", [("boots", (425, 470, 455, 505))]).outcome is Outcome.PRESENT
    o = assess("safety_shoes", [("no_boots", (425, 690, 455, 720))], box=(400.0, 320.0, 520.0, 720.0))
    assert o.outcome is Outcome.UNCERTAIN and o.reason == "not_assessable:feet_truncated"


def test_goggles_on_head_region():
    assert assess("goggles", [("no_goggles", (445, 115, 475, 135))]).outcome is Outcome.ABSENT


def test_thresholds_accept_phase1_names_and_dump_generic_ones():
    t = RuleThresholds.model_validate({"helmet_conf": 0.5, "head_conf": 0.6, "min_head_px": 20})
    assert (t.present_conf, t.absent_conf, t.min_roi_px) == (0.5, 0.6, 20)
    assert set(t.model_dump()) == {"person_conf", "present_conf", "absent_conf", "min_person_height_px", "min_roi_px"}


def test_rule_items_and_mandatory_default():
    r = RuleSeed(id="gloves-welding", name="Gloves", ppe_class="gloves")
    assert r.mandatory is True and r.enabled is True
    with pytest.raises(ValidationError):
        RuleSeed(id="x", name="x", ppe_class="earmuffs")


def test_shard_parsing_and_camera_split():
    """Network sharding (ADR-014): PPE_WORKER_SHARD=i/N deterministically splits cameras across N deployments."""
    from app.config.settings import Settings

    assert Settings(worker_shard=None).shard is None
    assert Settings(worker_shard="1/4").shard == (1, 4)
    with pytest.raises(ValueError):
        _ = Settings(worker_shard="4/4").shard
    with pytest.raises(ValueError):
        _ = Settings(worker_shard="x/2").shard
    ids = [f"cam{i:02d}" for i in range(1, 11)]
    halves = [[c for i, c in enumerate(ids) if i % 2 == sh] for sh in (0, 1)]
    assert halves[0] == ids[::2] and halves[1] == ids[1::2]  # disjoint, order-stable, together all cameras


def test_new_class_is_config_only(monkeypatch):
    """Future classes: a models.yaml ppe_items entry makes the item usable in rules, association and the teacher."""
    from app.config.settings import PPE_ITEMS, ModelsConfig

    cfg = ModelsConfig.model_validate({"person": {"weights": "p.pt"}, "ppe": {"weights": "h.pt"}, "ppe_items": {
        "ear_muffs": {"present": "earmuffs", "absent": "no_earmuffs", "region": "head", "label": "Ear muffs",
                      "alert": "NO EAR MUFFS", "teacher_question": "hearing protection ear muffs"}}})
    assert {"ear_muffs", "hard_hat"} <= set(cfg.ppe_items)  # merged with the built-ins
    monkeypatch.setitem(PPE_ITEMS, "ear_muffs", cfg.ppe_items["ear_muffs"])
    assert RuleSeed(id="muffs", name="Muffs", ppe_class="ear_muffs").ppe_class == "ear_muffs"
    assert assess("ear_muffs", [("no_earmuffs", (430, 100, 490, 150))]).outcome is Outcome.ABSENT
    assert "ear muffs" in instructions("ear_muffs")


@pytest.mark.parametrize("preset", ["construction", "welding"])
def test_site_presets_validate(preset):
    import yaml

    from app.config.settings import REPO_ROOT

    rules = [RuleSeed.model_validate(r) for r in
             yaml.safe_load((REPO_ROOT / "configs" / "presets" / f"{preset}.yaml").read_text())["rules"]]
    assert {r.ppe_class for r in rules} == set(ITEMS)  # every item configured: mandatory or secondary


def test_every_item_has_a_teacher_question():
    assert all(instructions(k) for k in ITEMS)
    assert "glove" in instructions("gloves") and "hard hat" in instructions("hard_hat")
