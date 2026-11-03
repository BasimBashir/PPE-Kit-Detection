"""WebSocket fan-out (FR-API-002). Broadcast is thread-safe: callers on any thread hop onto the event loop."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket

from app.observability import metrics as m

log = logging.getLogger(__name__)


class WebSocketHub:
    def __init__(self) -> None:
        self.clients: set[WebSocket] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self.loop = loop

    async def add(self, ws: WebSocket) -> None:
        self.clients.add(ws)
        m.ppe_websocket_clients.set(len(self.clients))

    def remove(self, ws: WebSocket) -> None:
        self.clients.discard(ws)
        m.ppe_websocket_clients.set(len(self.clients))

    async def _send_all(self, text: str) -> int:
        dead, sent = [], 0
        for ws in list(self.clients):
            try:
                await asyncio.wait_for(ws.send_text(text), timeout=2.0)
                sent += 1
            except Exception:  # noqa: BLE001 - slow/broken client must not block others
                dead.append(ws)
        for ws in dead:
            self.remove(ws)
        return sent

    async def broadcast_async(self, msg: dict[str, Any]) -> int:
        return await self._send_all(json.dumps(msg, default=str))

    def broadcast(self, msg: dict[str, Any]) -> None:
        """Fire-and-forget from any thread."""
        if self.loop is None or self.loop.is_closed() or not self.clients:
            return
        text = json.dumps(msg, default=str)
        asyncio.run_coroutine_threadsafe(self._send_all(text), self.loop)
