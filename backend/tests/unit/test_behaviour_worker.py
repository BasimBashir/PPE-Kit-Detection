"""Phase 4 worker wiring: behaviour rules get assessors; schedules gate rules by site-local time."""

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from app.behaviour.assessors import FallAssessor, PhoneAssessor
from app.camera.source import CameraRuntimeConfig
from app.config.settings import AppConfig, RuleSeed, Schedule
from app.inference.types import Detection
from app.observability.stats import RollingWindow
from app.ppe.association import Associator
from app.ppe.types import Outcome, TrackedPerson
from app.streaming.frame import Frame
from app.workers.pipeline_worker import CamRuntime, PipelineWorker

DXB = ZoneInfo("Asia/Dubai")


def body_kps():
    k = np.zeros((17, 3), np.float32)
    for i, (x, y) in {0: (460, 130), 5: (420, 160), 6: (500, 160), 7: (410, 240), 8: (510, 240), 9: (405, 300),
                      10: (515, 300), 11: (430, 300), 12: (490, 300), 15: (440, 480), 16: (480, 480)}.items():
        k[i] = (x, y, 0.9)
    return k


class PassThroughTracker:
    active_count = 0

    def update(self, persons, frame):
        return [TrackedPerson(i + 1, d.xyxy, d.conf, d.keypoints) for i, d in enumerate(persons)]


def bare_worker():
    w = object.__new__(PipelineWorker)  # no DB / GPU: only the rule-building and assessment paths are exercised
    w.cfg, w.classifier, w.fight = AppConfig(), None, None
    w.stage_ms = {k: RollingWindow(60.0) for k in ("tracking", "association")}
    return w


def cam(w, seeds):
    cr = CamRuntime(CameraRuntimeConfig("cam01", "rtsp://127.0.0.1:8554/cam01", None, "tcp", True, 5.0), "Cam 1")
    w._build_rules(cr, seeds)
    cr.tracker = PassThroughTracker()
    return cr


def test_rules_get_the_right_assessor():
    w = bare_worker()
    cr = cam(w, [RuleSeed(id="hat", name="Hat"), RuleSeed(id="fall", name="Fall", ppe_class="fall"),
                 RuleSeed(id="phone", name="Phone", ppe_class="phone")])
    kinds = {r.id: type(r.associator) for r in cr.rules}
    assert kinds == {"hat": Associator, "fall": FallAssessor, "phone": PhoneAssessor}


def test_schedule_gates_a_rule_by_local_time():
    w = bare_worker()
    sched = Schedule(tz="Asia/Dubai", days=[0], windows=["07:00-08:00"])
    cr = cam(w, [RuleSeed(id="phone", name="Phone", ppe_class="phone", schedule=sched)])
    dets = [Detection("person", 0.9, (400.0, 100.0, 520.0, 500.0), "t", body_kps()),
            Detection("phone", 0.9, (390.0, 300.0, 420.0, 340.0), "t")]
    img = np.zeros((720, 1280, 3), np.uint8)

    def obs_at(h, m):
        t = datetime(2026, 11, 23, h, m, tzinfo=DXB).timestamp()  # a Monday
        return w._track_assess(cr, Frame("cam01", 1, img, None, t, t), dets)["per_rule"][0][1]

    assert obs_at(9, 0) == []  # off shift: no observations, so open episodes expire and close
    (o,) = obs_at(7, 30)
    assert o.outcome is Outcome.ABSENT


def test_hot_reload_updates_the_schedule_in_place():
    w = bare_worker()
    cr = cam(w, [RuleSeed(id="fall", name="Fall", ppe_class="fall")])
    before = cr.rules[0]
    sched = Schedule(tz="Asia/Dubai", days=[0], windows=["07:00-08:00"])
    w._build_rules(cr, [RuleSeed(id="fall", name="Fall", ppe_class="fall", schedule=sched)], cr.rules)
    assert cr.rules[0] is before and before.schedule == sched  # track state kept, schedule applied
