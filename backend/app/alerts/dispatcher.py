"""Outbox dispatcher (API process): delivers durable alert_outbox rows to channels with bounded retries.

Channels: `dashboard` (WebSocket event_created to every connected client) and `local_sound` (optional host beep).
Woken immediately by worker `event` telemetry; also polls every second so nothing depends on the nudge.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import sys
import threading
from collections.abc import Awaitable, Callable
from datetime import timedelta
from typing import Any

from sqlalchemy import func, select

from app.config.settings import AlertsConfig
from app.db.base import SessionFactory, session_scope
from app.models import AlertOutbox, Camera, Event, utcnow
from app.observability import metrics as m
from app.repositories import outbox as outbox_repo
from app.schemas.serializers import event_out

log = logging.getLogger(__name__)
Channel = Callable[[dict[str, Any]], Awaitable[None]]
LEASE = timedelta(seconds=30)


class OutboxDispatcher:
    def __init__(self, sf: SessionFactory, cfg: AlertsConfig, channels: dict[str, Channel]):
        self.sf, self.cfg, self.channels = sf, cfg, channels
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self.loop: asyncio.AbstractEventLoop | None = None

    def start(self) -> None:
        self.loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._run(), name="outbox-dispatcher")

    def wake(self) -> None:
        if self.loop and not self.loop.is_closed():
            self.loop.call_soon_threadsafe(self._wake.set)

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()

    async def _run(self) -> None:
        while True:
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._wake.wait(), timeout=1.0)
            self._wake.clear()
            try:
                await self.dispatch_once()
            except Exception:  # noqa: BLE001 - DB down etc.; retry next tick
                log.debug("outbox dispatch round failed", exc_info=True)

    def _claim(self) -> list[tuple[int, str, dict[str, Any]]]:
        """Claim due rows under a lease: a crash mid-delivery means the row is retried after LEASE, never lost."""
        with session_scope(self.sf) as s:
            rows = outbox_repo.claim_due(s)
            out = []
            for r in rows:
                ev = s.get(Event, r.event_id)
                cam = s.get(Camera, ev.camera_id) if ev else None
                out.append((r.id, r.channel, event_out(ev, cam.name if cam else None) if ev else {}))
                r.next_attempt_at = utcnow() + LEASE
            pending = s.scalar(select(func.count()).select_from(AlertOutbox).where(AlertOutbox.status == "PENDING"))
            m.ppe_alert_outbox_pending.set(pending or 0)
            return out

    def _finish(self, row_id: int, error: str | None) -> None:
        with session_scope(self.sf) as s:
            r = s.get(AlertOutbox, row_id)
            if r is None:
                return
            if error is None:
                outbox_repo.mark_delivered(s, r)
            else:
                outbox_repo.mark_failed(s, r, error, self.cfg)

    async def dispatch_once(self) -> int:
        items = await asyncio.to_thread(self._claim)
        for row_id, channel, event in items:
            m.ppe_alert_delivery_attempts_total.labels(channel).inc()
            fn = self.channels.get(channel)
            err = None
            try:
                if fn is None:
                    raise RuntimeError(f"channel {channel} not configured")
                await fn(event)
            except Exception as e:  # noqa: BLE001
                err = f"{type(e).__name__}: {e}"
                m.ppe_alert_delivery_failures_total.labels(channel).inc()
                log.warning("alert delivery failed (%s): %s", channel, err)
            await asyncio.to_thread(self._finish, row_id, err)
        return len(items)


async def local_sound_channel(_event: dict[str, Any]) -> None:
    """FR-ALT-005 optional local notification: a host beep (Windows) / terminal bell elsewhere."""
    def beep() -> None:
        if sys.platform == "win32":
            import winsound

            winsound.Beep(1200, 300)
        else:
            print("\a", end="", flush=True)

    await asyncio.to_thread(beep)


def make_failure_injector(fail_first: int) -> Callable[[Channel], Channel]:
    """Test helper: wrap a channel so it fails the first N calls (outbox retry fault test)."""
    lock, state = threading.Lock(), {"n": 0}

    def wrap(fn: Channel) -> Channel:
        async def inner(ev: dict[str, Any]) -> None:
            with lock:
                state["n"] += 1
                n = state["n"]
            if n <= fail_first:
                raise ConnectionError(f"injected delivery failure {n}")
            await fn(ev)

        return inner

    return wrap
