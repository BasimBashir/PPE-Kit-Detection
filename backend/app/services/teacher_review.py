"""Asynchronous teacher review worker (API process). Off the alert path: candidates arrive from the pipeline worker
AFTER the on-site decision (violation events are already persisted and dispatched), are reviewed at a bounded rate,
and stored in teacher_reviews. Asymmetric safety rule (app/inference/teacher.py decide()):
- ESCALATED compliant sample -> a NEW event through the normal event + outbox path, marked "raised by teacher review";
- LIKELY_FALSE_ALERT -> only a teacher_reviews row for human review. Existing events are never updated or deleted.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from app.alerts.event_service import idempotency_key, insert_event_payload
from app.config.settings import TeacherConfig
from app.core.redaction import redact
from app.db.base import SessionFactory, session_scope
from app.evidence.store import EvidenceStore
from app.inference.teacher import CircuitBreaker, RateLimiter, Teacher, TeacherVerdict, decide, fuse
from app.models import TeacherReview
from app.observability import metrics as m

log = logging.getLogger(__name__)
TEACHER_REASON = "raised by teacher review"


class TeacherReviewer:
    def __init__(
        self,
        cfg: TeacherConfig,
        teacher: Teacher,
        sf: SessionFactory,
        evidence: EvidenceStore,
        channels: list[str],
        on_event: Callable[[], None] = lambda: None,
        maxsize: int = 256,
    ):
        self.cfg, self.teacher, self.sf, self.evidence = cfg, teacher, sf, evidence
        self.channels, self.on_event, self.maxsize = channels, on_event, maxsize
        self.q: deque[dict[str, Any]] = deque()
        self.breaker = CircuitBreaker()
        self.limiter = RateLimiter(cfg.max_reviews_per_minute)
        self.dropped = self.failures = 0
        self.last_error: str | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None

    # ------------------------------------------------------------ lifecycle
    def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    def submit(self, cand: dict[str, Any]) -> None:
        """Thread-safe (called from the telemetry pump thread)."""
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._enqueue, cand)

    def _enqueue(self, cand: dict[str, Any]) -> None:
        if len(self.q) >= self.maxsize:  # bounded: drop the oldest candidate, never block the caller
            self.q.popleft()
            self.dropped += 1
            m.ppe_teacher_failures_total.labels("dropped").inc()
        self.q.append(cand)
        self._wake.set()

    async def _run(self) -> None:
        while True:
            if not self.q:
                self._wake.clear()
                await self._wake.wait()
                continue
            wait = max(self.breaker.retry_in(), self.limiter.wait_time())
            if wait > 0:
                await asyncio.sleep(wait)
                continue
            self.limiter.try_acquire()
            try:
                await self.review_one(self.q.popleft())
            except Exception:  # noqa: BLE001 - one bad candidate must never stop the reviewer
                log.exception("teacher review crashed")

    # ------------------------------------------------------------ one review
    async def review_one(self, cand: dict[str, Any]) -> TeacherReview | None:
        try:
            crop = await asyncio.to_thread(lambda: self.evidence.open(cand["crop_path"]).read_bytes())
        except (FileNotFoundError, OSError):
            m.ppe_teacher_failures_total.labels("crop_missing").inc()
            return None
        try:
            v = await asyncio.wait_for(self.teacher.review(crop, cand.get("ppe_class", "hard_hat")), self.cfg.timeout_s)
        except Exception as e:  # noqa: BLE001 - teacher outages are expected and harmless
            reason = "timeout" if isinstance(e, TimeoutError) else "error"
            self.breaker.failure()
            self.failures += 1
            self.last_error = redact(f"{type(e).__name__}: {e}")[:200]
            m.ppe_teacher_failures_total.labels(reason).inc()
            log.warning("teacher review failed (%s): %s", reason, self.last_error)
            return None
        self.breaker.success()
        m.ppe_teacher_latency_seconds.observe(v.latency_ms / 1000)
        fused = fuse(v.p_missing, float(cand["p_model"]), self.cfg.weight)
        verdict = decide(cand["kind"], fused, self.cfg.escalate_threshold, self.cfg.false_alert_threshold)
        row = await asyncio.to_thread(self._persist, cand, v, fused, verdict)
        m.ppe_teacher_reviews_total.labels(verdict).inc()
        if verdict == "ESCALATED":
            self.on_event()  # wake the outbox dispatcher -> dashboard
        return row

    def _persist(self, cand: dict[str, Any], v: TeacherVerdict, fused: float, verdict: str) -> TeacherReview:
        event_id = cand.get("event_id")
        if verdict == "ESCALATED":
            event_id = self._escalate(cand, v, fused)
        row = TeacherReview(
            event_id=uuid.UUID(event_id) if event_id else None,
            kind=cand["kind"],
            camera_id=cand["camera_id"],
            track_id=int(cand["track_id"]),
            crop_path=cand["crop_path"],
            p_model=float(cand["p_model"]),
            p_teacher=v.p_missing,
            fused=round(fused, 4),
            verdict=verdict,
            provider=v.provider,
            latency_ms=round(v.latency_ms, 1),
            raw=v.raw,
        )
        with session_scope(self.sf) as s:
            s.add(row)
        return row

    def _escalate(self, cand: dict[str, Any], v: TeacherVerdict, fused: float) -> str:
        """INSERT a new event + outbox rows exactly like a pipeline-confirmed event (idempotent per crop)."""
        when = datetime.fromtimestamp(float(cand["captured_at"]), UTC)
        eid = str(uuid.uuid4())
        episode = f"teacher:{cand['crop_path']}"
        event = {
            "id": eid,
            "camera_id": cand["camera_id"],
            "track_id": int(cand["track_id"]),
            "zone_id": cand["zone_id"],
            "rule_id": cand["rule_id"],
            "episode_id": episode[:160],
            "status": "OPEN",
            "severity": cand["severity"],
            "idempotency_key": idempotency_key(
                cand["camera_id"], int(cand["track_id"]), cand["zone_id"], cand["rule_id"], episode
            ),
            "reason": TEACHER_REASON,
            "confidence_summary": {
                "raised_by": "teacher_review",
                "no_helmet_votes": 0,
                "decisive_votes": 0,
                "mean_confidence": round(fused, 4),
                "roi_source": "teacher_review",
                "classifier_used": False,
                "p_model": cand["p_model"],
                "p_teacher": round(v.p_missing, 4),
            },
            "first_seen_at": when,
            "confirmed_at": datetime.now(UTC),
            "frame_captured_at": when,
            "last_seen_at": when,
            "last_box": cand["person_box"],
            "evidence_path": cand["crop_path"],
            "evidence_error": None,
            "model_versions": {"teacher": f"{v.provider}:{self.cfg.model}"},
            "reconfirm_count": 0,
        }
        insert_event_payload(self.sf, {"event": event, "channels": self.channels})
        log.warning(
            "teacher review ESCALATED a compliant sample on %s track %s -> event %s (fused %.2f)",
            cand["camera_id"],
            cand["track_id"],
            eid,
            fused,
        )
        return eid

    def health(self) -> dict[str, Any]:
        return {
            "circuit": self.breaker.state,
            "consecutive_failures": self.breaker.consecutive,
            "queue_depth": len(self.q),
            "dropped": self.dropped,
            "failures": self.failures,
            "last_error": self.last_error,
        }
