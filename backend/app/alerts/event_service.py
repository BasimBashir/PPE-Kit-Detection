"""Violation events: idempotent creation, cooldown re-arm, spatial dedupe, durable outbox, spool fallback.

Runs on its own thread with a bounded queue so a slow or unreachable database never blocks GPU inference
(implementation-plan 4.9, ADR-006/007). All episode state lives on this thread - no locks needed.
"""

from __future__ import annotations

import hashlib
import logging
import queue
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
from sqlalchemy import update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.exc import DBAPIError, OperationalError

from app.alerts.spool import EventSpool
from app.config.settings import AlertsConfig
from app.db.base import SessionFactory, session_scope
from app.evidence.annotate import annotate_violation
from app.evidence.store import EvidenceStore
from app.models import AlertOutbox, Event
from app.observability import metrics as m
from app.ppe.geometry import iou
from app.ppe.types import Box

log = logging.getLogger(__name__)
DB_DOWN = (OperationalError, DBAPIError)


def ts(wall: float) -> datetime:
    return datetime.fromtimestamp(wall, UTC)


@dataclass
class ConfirmedViolation:
    camera_id: str
    track_id: int
    zone_id: str | None
    rule_id: str
    severity: str
    episode_seq: int
    person_box: Box
    head_roi: Box | None
    reason: str
    no_helmet_votes: int
    decisive_votes: int
    mean_confidence: float
    roi_source: str
    classifier_used: bool
    image: np.ndarray  # privacy-masked frame (BGR); annotated here
    frame_captured_at: float  # wall
    first_seen_at: float  # wall, first NO_HELMET of the episode
    confirmed_at: float  # wall
    model_versions: dict[str, str] = field(default_factory=dict)
    camera_name: str = ""


@dataclass
class _Active:
    event_id: uuid.UUID
    rule_id: str
    last_box: Box
    last_seen_mono: float
    last_head: Box | None = None
    last_db_update: float = 0.0
    ended_mono: float | None = None


def head_distance(a: Box | None, b: Box | None) -> float:
    """Distance between head-ROI centres in units of head width (inf when unknown)."""
    if a is None or b is None:
        return float("inf")
    w = max(1e-6, ((a[2] - a[0]) + (b[2] - b[0])) / 2)
    return ((((a[0] + a[2]) - (b[0] + b[2])) / 2) ** 2 + (((a[1] + a[3]) - (b[1] + b[3])) / 2) ** 2) ** 0.5 / w


def idempotency_key(camera_id: str, track_id: int, zone_id: str | None, rule_id: str, episode_id: str) -> str:
    return hashlib.sha256(f"{camera_id}|{track_id}|{zone_id or ''}|{rule_id}|{episode_id}".encode()).hexdigest()


def insert_event_payload(session_factory: SessionFactory, payload: dict[str, Any]) -> bool:
    """INSERT event + outbox rows in one transaction; idempotent. Returns True when the event row was new."""
    ev = dict(payload["event"])
    for k in ("first_seen_at", "confirmed_at", "frame_captured_at", "last_seen_at"):
        if isinstance(ev.get(k), str):
            ev[k] = datetime.fromisoformat(ev[k])
    ev["id"] = uuid.UUID(str(ev["id"]))
    ev["persisted_at"] = datetime.now(UTC)  # actual commit time, also for spool replays
    with session_scope(session_factory) as s:
        stmt = pg_insert(Event).values(**ev).on_conflict_do_nothing(index_elements=["idempotency_key"])
        new = s.execute(stmt.returning(Event.id)).first() is not None  # rowcount is unreliable with DO NOTHING
        if new:
            for ch in payload["channels"]:
                s.execute(pg_insert(AlertOutbox).values(event_id=ev["id"], channel=ch)
                          .on_conflict_do_nothing(constraint="uq_outbox_event_channel"))
    return new


class EventService(threading.Thread):
    def __init__(self, session_factory: SessionFactory, evidence: EvidenceStore, spool: EventSpool,
                 cfg: AlertsConfig, run_id: str, channels: list[str],
                 notify: Callable[[dict[str, Any]], None] | None = None, replay_interval_s: float = 2.0):
        super().__init__(name="event-service", daemon=True)
        self.sf, self.evidence, self.spool, self.cfg = session_factory, evidence, spool, cfg
        self.run_id, self.channels = run_id, channels
        self.notify = notify if notify is not None else (lambda _msg: None)
        self.replay_interval_s = replay_interval_s
        self.q: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1024)
        self._stop_evt = threading.Event()
        self.active: dict[tuple[str, int, str], _Active] = {}  # (camera, track, rule) -> current episode event
        self.recent: dict[str, deque[_Active]] = {}  # camera -> recently active events (spatial dedupe)
        self.db_ok = True
        self.db_retry_at = 0.0  # circuit breaker: no DB attempts (except replay) until this monotonic time
        self.processed = 0

    # ------------------------------------------------------------ producer API (inference thread)
    def submit_confirmed(self, v: ConfirmedViolation) -> None:
        self._put(("confirmed", v))

    def submit_still_confirmed(self, camera_id: str, track_id: int, rule_id: str, box: Box,
                               head: Box | None = None) -> None:
        self._put(("still", (camera_id, track_id, rule_id, box, head)), droppable=True)

    def submit_cleared(self, camera_id: str, track_id: int, rule_id: str, when: float) -> None:
        self._put(("cleared", (camera_id, track_id, rule_id, when)))

    def _put(self, item: tuple[str, Any], droppable: bool = False) -> None:
        try:
            self.q.put(item, timeout=0 if droppable else 0.5)
        except queue.Full:
            if not droppable:
                log.error("event queue full - %s dropped", item[0])

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_evt.set()
        self.join(timeout)

    def drain(self, timeout: float = 5.0) -> None:
        """Test helper: wait until the queue is empty and processed."""
        deadline = time.monotonic() + timeout
        while (self.q.unfinished_tasks and time.monotonic() < deadline):
            time.sleep(0.01)

    # ------------------------------------------------------------ worker thread
    def run(self) -> None:
        last_replay = 0.0
        while not self._stop_evt.is_set() or not self.q.empty():
            try:
                kind, data = self.q.get(timeout=0.25)
            except queue.Empty:
                kind = None
            if kind is not None:
                try:
                    {"confirmed": self._confirmed, "still": self._still, "cleared": self._cleared}[kind](data)
                except Exception:
                    log.exception("event processing failed (%s)", kind)
                finally:
                    self.processed += 1
                    self.q.task_done()
            if time.monotonic() - last_replay >= self.replay_interval_s:
                last_replay = time.monotonic()
                self.replay_spool()

    def replay_spool(self) -> int:
        try:
            n = self.spool.replay(lambda p: self._replay_one(p))
        except Exception:  # noqa: BLE001
            log.exception("spool replay error")
            return 0
        if n:
            log.info("replayed %d spooled events", n)
        return n

    DB_BACKOFF_S = 5.0

    def _db_down(self) -> None:
        self.db_ok = False
        self.db_retry_at = time.monotonic() + self.DB_BACKOFF_S

    def _db_open(self) -> bool:
        """False while the breaker is open: callers spool / skip instead of waiting on connect timeouts."""
        return time.monotonic() >= self.db_retry_at

    def _replay_one(self, payload: dict[str, Any]) -> None:
        insert_event_payload(self.sf, payload)
        self.db_ok, self.db_retry_at = True, 0.0
        self.notify({"type": "event", "action": "created", "event_id": payload["event"]["id"],
                     "camera_id": payload["event"]["camera_id"], "persisted_at": time.time()})

    # ------------------------------------------------------------ handlers
    def _confirmed(self, v: ConfirmedViolation) -> None:
        key = (v.camera_id, v.track_id, v.rule_id)
        now = time.monotonic()
        cur = self.active.get(key)
        if cur and cur.ended_mono is None:
            return  # same episode already has an event (FR-TRK-008 / FR-ALT-008)
        if cur and cur.ended_mono is not None and now - cur.ended_mono < self.cfg.cooldown_s:
            self._reopen(cur, v, "cooldown")
            return
        best = (0.0, None)
        for r in self.recent.get(v.camera_id, ()):
            if r.rule_id != v.rule_id or now - r.last_seen_mono >= self.cfg.spatial_dedupe_s:
                continue
            o = iou(r.last_box, v.person_box)
            hd = head_distance(r.last_head, v.head_roi)
            best = max(best, (o, round(now - r.last_seen_mono, 1)), key=lambda b: b[0])
            # Person boxes jump under occlusion; head position is the steadier identity cue (ID switch / loop).
            if o >= self.cfg.spatial_dedupe_iou or hd <= self.cfg.spatial_dedupe_head_dist:
                self.active[key] = r  # new track id, same person/place: adopt the existing event
                self._reopen(r, v, "spatial")
                return
        log.info("new event %s track %d (closest recent event: iou=%.2f, last seen %s s ago)",
                 v.camera_id, v.track_id, best[0], best[1])
        self._create(v, key)

    def _reopen(self, a: _Active, v: ConfirmedViolation, reason: str) -> None:
        m.ppe_events_deduplicated_total.labels(reason).inc()
        a.ended_mono, a.last_box, a.last_seen_mono = None, v.person_box, time.monotonic()
        a.last_head = v.head_roi or a.last_head
        if not self._db_open():
            return  # the reopened episode stays tracked in memory; DB row catches up on the next update
        try:
            with session_scope(self.sf) as s:
                s.execute(update(Event).where(Event.id == a.event_id).values(
                    episode_ended_at=None, reconfirm_count=Event.reconfirm_count + 1,
                    last_seen_at=ts(v.confirmed_at), last_box=list(v.person_box)))
            self.notify({"type": "event", "action": "updated", "event_id": str(a.event_id),
                         "camera_id": v.camera_id, "persisted_at": time.time()})
        except DB_DOWN:
            self._db_down()

    def _create(self, v: ConfirmedViolation, key: tuple[str, int, str]) -> None:
        event_id = uuid.uuid4()
        episode_id = f"{self.run_id}:{v.camera_id}:{v.track_id}:{v.episode_seq}"
        label = f"NO HARD HAT  {v.camera_id}  track {v.track_id}"
        img = annotate_violation(v.image, v.person_box, v.head_roi, label)
        rel, err = self.evidence.save(event_id, img, v.frame_captured_at)  # before DB so spooled events keep it
        event = {
            "id": str(event_id),
            "idempotency_key": idempotency_key(v.camera_id, v.track_id, v.zone_id, v.rule_id, episode_id),
            "camera_id": v.camera_id, "track_id": v.track_id, "zone_id": v.zone_id, "rule_id": v.rule_id,
            "episode_id": episode_id, "status": "OPEN", "severity": v.severity, "reason": v.reason,
            "confidence_summary": {"no_helmet_votes": v.no_helmet_votes, "decisive_votes": v.decisive_votes,
                                   "mean_confidence": round(v.mean_confidence, 4), "roi_source": v.roi_source,
                                   "classifier_used": v.classifier_used},
            "first_seen_at": ts(v.first_seen_at).isoformat(), "confirmed_at": ts(v.confirmed_at).isoformat(),
            "frame_captured_at": ts(v.frame_captured_at).isoformat(),
            "last_seen_at": ts(v.confirmed_at).isoformat(), "last_box": list(v.person_box),
            "evidence_path": rel, "evidence_error": err, "model_versions": v.model_versions,
            "reconfirm_count": 0,
        }
        payload = {"event": event, "channels": self.channels}
        a = _Active(event_id, v.rule_id, v.person_box, time.monotonic(), v.head_roi)
        self.active[key] = a
        self.recent.setdefault(v.camera_id, deque(maxlen=32)).append(a)
        m.ppe_event_time_to_confirm_seconds.observe(max(0.0, v.confirmed_at - v.first_seen_at))
        try:
            if not self._db_open():
                raise OperationalError("insert", {}, Exception("circuit open - database recently unreachable"))
            insert_event_payload(self.sf, payload)
            persisted = time.time()
            self.db_ok = True
            m.ppe_events_created_total.labels(v.camera_id).inc()
            m.ppe_event_confirmation_latency_seconds.observe(max(0.0, persisted - v.frame_captured_at))
            self.notify({"type": "event", "action": "created", "event_id": str(event_id),
                         "camera_id": v.camera_id, "persisted_at": persisted})
        except DB_DOWN as e:
            self._db_down()
            log.warning("database unavailable (%s) - event %s spooled", type(e).__name__, event_id)
            self.spool.append(payload)
            m.ppe_events_created_total.labels(v.camera_id).inc()
            self.notify({"type": "event", "action": "spooled", "event_id": str(event_id),
                         "camera_id": v.camera_id, "persisted_at": None})

    def _still(self, data: tuple[str, int, str, Box]) -> None:
        camera_id, track_id, rule_id, box, head = data
        a = self.active.get((camera_id, track_id, rule_id))
        if not a or a.ended_mono is not None:
            return
        now = time.monotonic()
        a.last_box, a.last_seen_mono = box, now
        a.last_head = head or a.last_head
        if now - a.last_db_update < self.cfg.last_seen_update_s or not self._db_open():
            return
        a.last_db_update = now
        try:
            with session_scope(self.sf) as s:
                s.execute(update(Event).where(Event.id == a.event_id)
                          .values(last_seen_at=datetime.now(UTC), last_box=list(box)))
        except DB_DOWN:
            self._db_down()  # non-critical; the event itself is durable

    def _cleared(self, data: tuple[str, int, str, float]) -> None:
        camera_id, track_id, rule_id, when = data
        a = self.active.get((camera_id, track_id, rule_id))
        if not a or a.ended_mono is not None:
            return
        a.ended_mono = time.monotonic()
        if not self._db_open():
            return  # episode end is best-effort metadata; cooldown/dedupe state is kept in memory
        try:
            with session_scope(self.sf) as s:
                s.execute(update(Event).where(Event.id == a.event_id).values(episode_ended_at=ts(when)))
            self.notify({"type": "event", "action": "closed", "event_id": str(a.event_id),
                         "camera_id": camera_id, "persisted_at": time.time()})
        except DB_DOWN:
            self._db_down()
