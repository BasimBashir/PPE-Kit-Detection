"""WebSocket /ws/live (FR-API-002, ADR-009): snapshot on every (re)connect, then live updates."""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import UTC, datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.routes_system import build_metrics
from app.db.base import session_scope
from app.models import Camera
from app.observability import metrics as m
from app.repositories import events as ev_repo
from app.repositories import users as users_repo
from app.schemas.serializers import camera_out, event_out
from app.services.live_state import camera_status_out

router = APIRouter()
log = logging.getLogger(__name__)


def _snapshot(c) -> dict:
    with session_scope(c.sf) as s:
        counts = ev_repo.count_open_by_camera(s)
        cams = list(s.query(Camera).order_by(Camera.id))
        names = {x.id: x.name for x in cams}
        return {"type": "snapshot",
                "cameras": [camera_out(x, camera_status_out(c.live.camera(x.id)), counts.get(x.id, 0)) for x in cams],
                "open_events": [event_out(e, names.get(e.camera_id)) for e in ev_repo.open_events(s, 100)],
                "metrics": build_metrics(c, s)}


def _record_render(c, event_id: str, rendered_at: str) -> None:
    ts = datetime.fromisoformat(rendered_at.replace("Z", "+00:00"))
    with session_scope(c.sf) as s:
        eid = uuid.UUID(event_id)
        if ev_repo.set_dashboard_rendered(s, eid, ts):
            e = ev_repo.get_event(s, eid)
            lat = (ts - e.persisted_at).total_seconds()
            if lat >= 0:
                m.ppe_event_to_dashboard_latency_seconds.observe(lat)
                c.live.persist_to_dashboard.add(lat)


@router.websocket("/ws/live")
async def live(ws: WebSocket):
    c = ws.app.state.ctx
    token = ws.query_params.get("token", "")
    with session_scope(c.sf) as s:
        user = users_repo.user_for_token(s, token) if token else None
    if user is None:
        await ws.close(code=4401)
        return
    await ws.accept()
    await c.hub.add(ws)
    try:
        try:
            await ws.send_text(json.dumps(await asyncio.to_thread(_snapshot, c), default=str))
        except Exception:  # noqa: BLE001 - DB down: still stream live camera state
            log.warning("snapshot without database", exc_info=True)
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if msg.get("type") == "event_rendered" and msg.get("event_id") and msg.get("rendered_at"):
                try:
                    await asyncio.to_thread(_record_render, c, str(msg["event_id"]), str(msg["rendered_at"]))
                except Exception:  # noqa: BLE001 - malformed ack must not drop the socket
                    log.debug("bad render ack", exc_info=True)
    except WebSocketDisconnect:
        pass
    finally:
        c.hub.remove(ws)


async def periodic_broadcasts(c) -> None:
    """camera_status 1 Hz, metrics every 5 s, ping every 15 s, worker up/down on change."""
    tick, last_worker = 0, None
    while True:
        await asyncio.sleep(1.0)
        tick += 1
        if not c.hub.clients:
            continue
        cams = [{"camera_id": k, **(camera_status_out(v) or {})} for k, v in c.live.all_cameras().items()]
        await c.hub.broadcast_async({"type": "camera_status", "cameras": cams})
        w = (c.live.worker["up"], c.live.worker["pid"], c.live.worker["run_id"])
        if w != last_worker:
            last_worker = w
            await c.hub.broadcast_async({"type": "worker", "up": w[0], "pid": w[1], "run_id": w[2]})
        if tick % 5 == 0:
            try:
                def _m():
                    with c.sf() as s:
                        return build_metrics(c, s)
                await c.hub.broadcast_async({"type": "metrics", "metrics": await asyncio.to_thread(_m)})
            except Exception:  # noqa: BLE001
                log.debug("metrics broadcast failed", exc_info=True)
        if tick % 15 == 0:
            await c.hub.broadcast_async({"type": "ping", "ts": datetime.now(UTC).isoformat()})
