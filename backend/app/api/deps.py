"""Shared API dependencies: app context, DB sessions, authentication and role checks (NFR-SEC-002)."""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from fastapi import Depends, Request
from sqlalchemy import Engine
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.config.settings import AppConfig, Settings
from app.db.base import SessionFactory
from app.evidence.store import EvidenceStore
from app.models import User
from app.repositories import users as users_repo
from app.services.live_state import LiveState
from app.services.ws_hub import WebSocketHub

ROLE_RANK = {"viewer": 0, "operator": 1, "admin": 2}


class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: Any = None):
        self.status, self.code, self.message, self.details = status, code, message, details


@dataclass
class AppContext:
    settings: Settings
    cfg: AppConfig
    engine: Engine
    sf: SessionFactory
    live: LiveState
    hub: WebSocketHub
    evidence: EvidenceStore
    supervisor: Any = None  # WorkerSupervisor | None (None in API-only tests)
    dispatcher: Any = None
    simulator: Any = None  # SimulatorClient in demo mode
    api_resources: dict[str, Any] = field(default_factory=dict)
    login_attempts: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(deque))
    login_lock: threading.Lock = field(default_factory=threading.Lock)

    def send_worker(self, cmd: dict[str, Any]) -> None:
        if self.supervisor is not None:
            self.supervisor.send(cmd)


def ctx(request: Request) -> AppContext:
    return request.app.state.ctx


def db(c: AppContext = Depends(ctx)) -> Iterator[Session]:
    s = c.sf()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def bearer(request: Request) -> str | None:
    h = request.headers.get("authorization", "")
    return h[7:].strip() if h.lower().startswith("bearer ") else None


_TOKEN_CACHE: dict[str, tuple[Any, float]] = {}
TOKEN_CACHE_TTL_S = 300.0


def current_user(request: Request, s: Session = Depends(db), c: AppContext = Depends(ctx)) -> User:
    """Validate the bearer token. If the database is unreachable, a token validated within the last 5 minutes is
    still accepted, so operators keep live visibility during a DB outage (read-mostly degraded mode)."""
    token = bearer(request)
    if not token:
        raise ApiError(401, "unauthorized", "missing or invalid bearer token")
    try:
        if not c.live.db_up:  # health probe says DB down: avoid a connect timeout on every request
            raise OperationalError("auth", {}, Exception("database marked down by health probe"))
        u = users_repo.user_for_token(s, token)
    except OperationalError:
        cached = _TOKEN_CACHE.get(token)
        if cached and time.monotonic() - cached[1] < TOKEN_CACHE_TTL_S:
            request.state.user = cached[0]
            return cached[0]
        raise ApiError(503, "unavailable", "database unavailable") from None
    if u is None:
        _TOKEN_CACHE.pop(token, None)
        raise ApiError(401, "unauthorized", "missing or invalid bearer token")
    s.expunge(u)
    _TOKEN_CACHE[token] = (u, time.monotonic())
    request.state.user = u
    return u


def require(role: str):
    def dep(u: User = Depends(current_user)) -> User:
        if ROLE_RANK[u.role] < ROLE_RANK[role]:
            raise ApiError(403, "forbidden", f"requires role {role}")
        return u

    return dep


def check_login_rate(c: AppContext, ip: str, limit: int = 5, window_s: float = 60.0) -> None:
    now = time.monotonic()
    with c.login_lock:
        q = c.login_attempts[ip]
        while q and now - q[0] > window_s:
            q.popleft()
        if len(q) >= limit:
            raise ApiError(429, "rate_limited", "too many login attempts; try again later")
        q.append(now)
