"""PostgreSQL integration: migrations, seeding, repositories, event service, spool, evidence (spec 19.2)."""

import time
import uuid
from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import func, select
from sqlalchemy.exc import OperationalError

from app.alerts.event_service import ConfirmedViolation, EventService
from app.alerts.spool import EventSpool
from app.config.settings import AlertsConfig, CameraSeed, EvidenceConfig, RuleSeed
from app.db.base import session_scope
from app.evidence.store import EvidenceStore
from app.models import AlertOutbox, Event, EventAction, utcnow
from app.repositories import cameras as cams
from app.repositories import events as evs
from app.repositories import outbox, users
from app.repositories.misc import seed_if_empty

pytestmark = pytest.mark.integration


def test_seed_is_idempotent(seeded):
    with session_scope(seeded) as s:
        again = seed_if_empty(s, [CameraSeed(id="cam01", name="x", stream_uri="rtsp://h/c")],
                              [RuleSeed(id="hardhat-required", name="x")], [], "another-password")
        assert again == {"cameras": 0, "rules": 0, "zones": 0, "users": 0}
        assert [c.id for c in cams.list_cameras(s)] == ["cam01", "cam02"]


def test_camera_crud_rejects_embedded_credentials(seeded):
    with session_scope(seeded) as s:
        with pytest.raises(ValueError):
            cams.create_camera(s, {"id": "cam03", "name": "c", "stream_uri": "rtsp://u:p@h/c"})
        c = cams.update_camera(s, cams.get_camera(s, "cam02"), {"enabled": False, "name": "Renamed"})
        assert c.enabled is False and c.name == "Renamed"


def test_auth_hash_sessions_expiry_revocation(seeded):
    with session_scope(seeded) as s:
        u = users.authenticate(s, "admin", "admin-password-1")
        assert u and u.role == "admin" and u.password_hash.startswith("$argon2id$")
        assert users.authenticate(s, "admin", "wrong") is None and users.authenticate(s, "ghost", "x") is None
        token, sess = users.create_session(s, u)
        assert users.user_for_token(s, token).id == u.id
        sess.expires_at = utcnow() - timedelta(seconds=1)
        s.flush()
        assert users.user_for_token(s, token) is None
        token2, _ = users.create_session(s, u)
        users.revoke(s, token2)
        assert users.user_for_token(s, token2) is None
        with pytest.raises(ValueError):
            users.create_user(s, "short", "1234567", "viewer")


class Notify(list):
    def __call__(self, msg):
        self.append(msg)


def make_service(sf, tmp_path, cfg=None, disk_free=10**12):
    ev_store = EvidenceStore(tmp_path / "evidence", EvidenceConfig(min_free_bytes=10**9),
                             disk_usage=lambda _p: SimpleNamespace(free=disk_free))
    note = Notify()
    svc = EventService(sf, ev_store, EventSpool(tmp_path / "spool"), cfg or AlertsConfig(cooldown_s=60),
                       run_id="run1", channels=["dashboard"], notify=note,
                       replay_interval_s=float("inf"))  # no background replay (3600 still fired once if uptime > 1 h)
    svc.start()
    return svc, note, ev_store


def violation(track=7, seq=1, box=(100.0, 100.0, 200.0, 400.0), cam="cam01"):
    now = time.time()
    head = (box[0] + 20, box[1] - 10, box[2] - 20, box[1] + 60)
    return ConfirmedViolation(cam, track, None, "hardhat-required", "high", seq, box, head,
                              "temporal_policy_met:count", 12, 13, 0.8, "keypoints", False,
                              np.zeros((360, 640, 3), np.uint8), now - 0.1, now - 2.5, now, {"person": "p@1"})


def count(sf, model):
    with session_scope(sf) as s:
        return s.scalar(select(func.count()).select_from(model))


def test_one_episode_one_event_with_outbox_and_evidence(seeded, tmp_path):
    svc, note, store = make_service(seeded, tmp_path)
    svc.submit_confirmed(violation())
    svc.submit_confirmed(violation())  # same episode again -> ignored (FR-ALT-008)
    svc.drain()
    svc.stop()
    assert count(seeded, Event) == 1 and count(seeded, AlertOutbox) == 1
    with session_scope(seeded) as s:
        e = s.scalar(select(Event))
        assert e.status == "OPEN" and e.evidence_error is None
        assert e.evidence_path.endswith(f"{e.id}.jpg") and store.open(e.evidence_path).stat().st_size > 0
        assert (e.persisted_at - e.frame_captured_at).total_seconds() >= 0
    assert [n["action"] for n in note] == ["created"]


def test_cooldown_reopens_and_after_cooldown_creates_new(seeded, tmp_path):
    svc, _, _ = make_service(seeded, tmp_path, AlertsConfig(cooldown_s=0.3, spatial_dedupe_s=0))
    svc.submit_confirmed(violation(seq=1))
    svc.submit_cleared("cam01", 7, "hardhat-required", time.time())
    svc.submit_confirmed(violation(seq=2))  # within cooldown -> same event reopened
    svc.drain()
    with session_scope(seeded) as s:
        e = s.scalar(select(Event))
        assert count(seeded, Event) == 1 and e.reconfirm_count == 1 and e.episode_ended_at is None
    svc.submit_cleared("cam01", 7, "hardhat-required", time.time())
    svc.drain()
    time.sleep(0.4)
    svc.submit_confirmed(violation(seq=3))
    svc.drain()
    svc.stop()
    assert count(seeded, Event) == 2


def test_spatial_dedupe_merges_new_track_at_same_place(seeded, tmp_path):
    svc, _, _ = make_service(seeded, tmp_path, AlertsConfig(spatial_dedupe_s=30))
    svc.submit_confirmed(violation(track=7))
    svc.submit_confirmed(violation(track=8, box=(105.0, 102.0, 205.0, 405.0)))  # ID switch, same person
    svc.submit_confirmed(violation(track=9, box=(600.0, 100.0, 700.0, 400.0)))  # different person
    # occluded re-detection: person box very different (IoU < 0.3) but the head is in the same place
    svc.submit_confirmed(violation(track=10, box=(100.0, 100.0, 210.0, 190.0)))
    svc.drain()
    svc.stop()
    assert count(seeded, Event) == 2


def test_db_outage_spools_then_replays_exactly_once(seeded, tmp_path):
    calls = {"down": True, "attempts": 0}

    def flaky_sf():
        if calls["down"]:
            calls["attempts"] += 1
            raise OperationalError("connect", {}, Exception("connection refused"))
        return seeded()

    svc, note, _ = make_service(flaky_sf, tmp_path)
    svc.submit_confirmed(violation(track=1))
    svc.submit_confirmed(violation(track=2, box=(700.0, 100.0, 800.0, 400.0)))
    svc.drain()
    assert len(svc.spool.pending()) == 2 and count(seeded, Event) == 0
    assert [n["action"] for n in note] == ["spooled", "spooled"]
    # 1 failed insert; the open breaker spooled the 2nd event without another connect attempt
    assert calls["attempts"] == 1
    calls["down"] = False
    assert svc.replay_spool() == 2
    assert svc.replay_spool() == 0
    svc.stop()
    assert count(seeded, Event) == 2 and count(seeded, AlertOutbox) == 2 and svc.spool.pending() == []
    # replaying the same payload again (e.g. crash between insert and spool rewrite) must not duplicate
    from app.alerts.event_service import insert_event_payload

    with session_scope(seeded) as s:
        e = s.scalar(select(Event))
    dup = {"event": {"id": str(uuid.uuid4()), "idempotency_key": e.idempotency_key, "camera_id": "cam01",
                     "track_id": 1, "zone_id": None, "rule_id": "hardhat-required", "episode_id": e.episode_id,
                     "status": "OPEN", "severity": "high", "reason": "r", "confidence_summary": {},
                     "first_seen_at": e.first_seen_at.isoformat(), "confirmed_at": e.confirmed_at.isoformat(),
                     "frame_captured_at": e.frame_captured_at.isoformat(), "last_seen_at": e.last_seen_at.isoformat(),
                     "last_box": None, "evidence_path": None, "evidence_error": None, "model_versions": {},
                     "reconfirm_count": 0}, "channels": ["dashboard"]}
    assert insert_event_payload(seeded, dup) is False and count(seeded, Event) == 2


def test_disk_low_still_persists_event(seeded, tmp_path):
    svc, _, _ = make_service(seeded, tmp_path, disk_free=10)
    svc.submit_confirmed(violation())
    svc.drain()
    svc.stop()
    with session_scope(seeded) as s:
        e = s.scalar(select(Event))
        assert e.evidence_path is None and e.evidence_error == "disk_low"


def test_event_lifecycle_filters_outbox_and_render_ack(seeded, tmp_path):
    svc, _, _ = make_service(seeded, tmp_path)
    svc.submit_confirmed(violation(track=1))
    svc.submit_confirmed(violation(track=2, box=(700.0, 100.0, 800.0, 400.0), cam="cam02"))
    svc.drain()
    svc.stop()
    with session_scope(seeded) as s:
        rows, total = evs.list_events(s, evs.EventFilter(camera_id="cam02"))
        assert total == 1 and rows[0].camera_id == "cam02"
        e = rows[0]
        with pytest.raises(evs.StateConflict):
            evs.acknowledge(s, evs.resolve(s, e, "op"), "op")
        evs.set_feedback(s, e, "op", "FALSE_POSITIVE", "shadow")
        assert [a.action for a in evs.actions_for(s, e.id)] == ["RESOLVE", "FEEDBACK"]
        assert evs.list_events(s, evs.EventFilter(feedback="FALSE_POSITIVE"))[1] == 1
        assert evs.list_events(s, evs.EventFilter(status="OPEN"))[1] == 1
        t1 = utcnow()
        assert evs.set_dashboard_rendered(s, e.id, t1) is True
        assert evs.set_dashboard_rendered(s, e.id, t1 + timedelta(seconds=5)) is False
    cfg = AlertsConfig(outbox_max_attempts=2, outbox_backoff_base_s=0)
    with session_scope(seeded) as s:
        rows = outbox.claim_due(s)
        assert len(rows) == 2
        outbox.mark_delivered(s, rows[0])
        outbox.mark_failed(s, rows[1], "boom", cfg)
    with session_scope(seeded) as s:
        (r,) = outbox.claim_due(s)
        outbox.mark_failed(s, r, "boom again", cfg)
    with session_scope(seeded) as s:
        assert outbox.counts(s) == {"DELIVERED": 1, "FAILED": 1}
        assert s.scalar(select(func.count()).select_from(EventAction)) == 2
