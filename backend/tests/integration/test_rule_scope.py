"""Per-camera enable/disable of a rule, for a Phase 1/3 PPE check and a Phase 4 behaviour alike.

The Rules page turns a rule off (`enabled`) and pins it to one camera (`camera_id`); both reach the worker through
the same two functions, so the two check families must scope identically. Runs the real `_load` (DB -> skip
disabled -> RuleSeed) and `_build_rules` (camera scoping + assessor per kind) - no models, no engine.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.behaviour.assessors import FallAssessor, PhoneAssessor
from app.camera.source import CameraRuntimeConfig
from app.config.settings import PPE_ITEMS, AppConfig, RuleSeed
from app.db.base import session_scope
from app.ppe.association import Associator
from app.repositories import rules as rules_repo
from app.workers.pipeline_worker import CamRuntime, PipelineWorker

pytestmark = pytest.mark.integration

PPE_B = next(k for k in sorted(PPE_ITEMS) if k != "hard_hat")  # a second PPE check, so ids stay distinct


def worker(sf) -> PipelineWorker:
    w = object.__new__(PipelineWorker)  # no engine / GPU: only the rule path (as in tests/unit)
    w.cfg, w.classifier, w.fight, w.sf = AppConfig(), None, None, sf
    w.s, w.stage_ms = SimpleNamespace(shard=None), {}
    return w


def built(w: PipelineWorker, camera_id: str, rules: list[RuleSeed]) -> CamRuntime:
    cr = CamRuntime(
        CameraRuntimeConfig(camera_id, f"rtsp://localhost:18501/{camera_id}", None, "tcp", True, 5.0), camera_id
    )
    w._build_rules(cr, rules)
    return cr


def test_enabled_and_camera_scope_apply_to_ppe_and_behaviour_rules(seeded):
    with session_scope(seeded) as s:  # 'seeded' also leaves hardhat-required enabled on all cameras
        for seed in (
            RuleSeed(id="hat-cam1", name="Hard hat cam01", ppe_class="hard_hat", camera_id="cam01"),
            RuleSeed(
                id="hat-cam1-off", name="Hard hat cam01 off", ppe_class="hard_hat", camera_id="cam01", enabled=False
            ),
            RuleSeed(id="fall-cam1", name="Fall cam01", ppe_class="fall", camera_id="cam01"),
            RuleSeed(id="fall-cam2-off", name="Fall cam02 off", ppe_class="fall", camera_id="cam02", enabled=False),
            RuleSeed(id="phone-cam2", name="Phone cam02", ppe_class="phone", camera_id="cam02"),
            RuleSeed(id=f"{PPE_B}-all", name="Second item everywhere", ppe_class=PPE_B),
        ):
            rules_repo.upsert_rule(s, seed)

    w = worker(seeded)
    _cams, rules, _zones = w._load()

    # a disabled rule never reaches the worker, whichever check it carries; camera_id survives the seed round-trip
    assert sorted(r.id for r in rules) == sorted(
        ["hardhat-required", f"{PPE_B}-all", "hat-cam1", "fall-cam1", "phone-cam2"]
    )
    assert {r.id: r.camera_id for r in rules}["fall-cam1"] == "cam01"

    cam01, cam02 = built(w, "cam01", rules), built(w, "cam02", rules)
    assert {r.id for r in cam01.rules} == {"hardhat-required", f"{PPE_B}-all", "hat-cam1", "fall-cam1"}
    assert {r.id for r in cam02.rules} == {"hardhat-required", f"{PPE_B}-all", "phone-cam2"}

    assessors = {r.id: r.associator for r in cam01.rules}
    assert isinstance(assessors["hat-cam1"], Associator)  # PPE check
    assert isinstance(assessors["fall-cam1"], FallAssessor)  # behaviour check
    assert isinstance({r.id: r for r in cam02.rules}["phone-cam2"].associator, PhoneAssessor)

    fsm01 = {r.id: r.fsm for r in cam01.rules}
    fsm02 = {r.id: r.fsm for r in cam02.rules}
    assert fsm01[f"{PPE_B}-all"] is not fsm02[f"{PPE_B}-all"]  # one state machine per (camera, rule)
