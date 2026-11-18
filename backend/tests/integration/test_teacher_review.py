"""Teacher review against PostgreSQL with a FakeTeacher: reviews are stored, escalations become visible events
through the normal event + outbox path, and a LIKELY_FALSE_ALERT tag never touches the original event."""

import time

import numpy as np
import pytest
from sqlalchemy import event, func, select

from app.config.settings import TeacherConfig
from app.db.base import session_scope
from app.inference.teacher import FakeTeacher
from app.models import AlertOutbox, Event, TeacherReview
from app.services.teacher_review import TEACHER_REASON, TeacherReviewer
from tests.integration.test_persistence import make_service, violation

pytestmark = pytest.mark.integration
CFG = TeacherConfig(enabled=True, provider="fake", weight=0.9, escalate_threshold=0.8, false_alert_threshold=0.2)


def candidates(sf, tmp_path):
    """Run the real EventService with teacher crops on: one confirmed violation + one sampled compliant track."""
    svc, note, store = make_service(sf, tmp_path)
    svc.teacher_enabled = True
    svc.submit_confirmed(violation(track=7))
    svc.submit_teacher_sample(
        "cam01",
        9,
        "hardhat-required",
        None,
        "high",
        (600.0, 100.0, 700.0, 400.0),
        np.full((40, 30, 3), 128, np.uint8),
        0.3,
        time.time(),
    )
    svc.drain()
    svc.stop()
    cands = {n["kind"]: n for n in note if n["type"] == "teacher_candidate"}
    assert set(cands) == {"violation", "compliant"}
    for c in cands.values():
        assert c["crop_path"].startswith("teacher/") and store.open(c["crop_path"]).stat().st_size > 0
    return cands, store


def event_snapshot(sf):
    with session_scope(sf) as s:
        return [
            (e.id, e.status, e.acknowledged_at, e.resolved_at, e.feedback, e.reason, e.updated_at)
            for e in s.scalars(select(Event).order_by(Event.id))
        ]


async def test_confirmed_event_gets_a_review_row(seeded, tmp_path):
    cands, store = candidates(seeded, tmp_path)
    r = TeacherReviewer(CFG, FakeTeacher(0.95), seeded, store, ["dashboard"])
    row = await r.review_one(cands["violation"])
    assert row.verdict == "CONFIRMED" and str(row.event_id) == cands["violation"]["event_id"]
    assert row.p_teacher == 0.95 and row.p_model == 0.8 and row.provider == "fake"
    assert row.fused == pytest.approx(0.9 * 0.95 + 0.1 * 0.8, abs=1e-4)


async def test_false_alert_tag_never_changes_the_event(seeded, tmp_path, engine):
    cands, store = candidates(seeded, tmp_path)
    before = event_snapshot(seeded)
    stmts = []

    def capture(_conn, _cursor, statement, *_a):
        stmts.append(statement.lstrip().upper())

    event.listen(engine, "before_cursor_execute", capture)
    try:
        r = TeacherReviewer(CFG, FakeTeacher(0.0), seeded, store, ["dashboard"])
        row = await r.review_one(cands["violation"])
    finally:
        event.remove(engine, "before_cursor_execute", capture)
    assert row.verdict == "LIKELY_FALSE_ALERT"  # fused = 0.9 * 0 + 0.1 * 0.8 = 0.08 < 0.2
    assert any(x.startswith("INSERT INTO TEACHER_REVIEWS") for x in stmts)  # the listener did see the review
    assert [x for x in stmts if x.startswith(("UPDATE", "DELETE"))] == []  # ...and no UPDATE / DELETE at all
    assert event_snapshot(seeded) == before  # still OPEN, not acknowledged, resolved or relabelled
    with session_scope(seeded) as s:
        assert s.scalar(select(func.count()).select_from(AlertOutbox)) == 1  # the alert's delivery is untouched


async def test_escalation_creates_a_visible_event(seeded, tmp_path):
    cands, store = candidates(seeded, tmp_path)
    woke = []
    r = TeacherReviewer(CFG, FakeTeacher(0.99), seeded, store, ["dashboard"], on_event=lambda: woke.append(1))
    row = await r.review_one(cands["compliant"])
    assert row.verdict == "ESCALATED" and row.event_id is not None and woke == [1]
    with session_scope(seeded) as s:
        e = s.get(Event, row.event_id)
        assert e.status == "OPEN" and e.reason == TEACHER_REASON and e.track_id == 9
        assert (
            e.confidence_summary["raised_by"] == "teacher_review" and e.evidence_path == cands["compliant"]["crop_path"]
        )
        assert s.scalar(select(func.count()).select_from(AlertOutbox).where(AlertOutbox.event_id == e.id)) == 1
        assert s.scalar(select(func.count()).select_from(Event)) == 2  # original violation + escalation
        assert s.scalar(select(func.count()).select_from(TeacherReview)) == 1
