"""FastAPI application: REST /api/v1, WebSocket /ws/live, Prometheus /metrics, static dashboard (offline).

Run: uvicorn app.main:factory --factory --host 127.0.0.1 --port 8000   (from backend/)
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException

from app import __version__
from app.alerts.dispatcher import OutboxDispatcher, local_sound_channel
from app.api import routes_auth, routes_cameras, routes_demo, routes_events, routes_system, routes_ws
from app.api.deps import ApiError, AppContext
from app.config.settings import (
    REPO_ROOT,
    get_settings,
    load_app_config,
    load_camera_seeds,
    load_rule_seeds,
)
from app.core.logging import setup_logging
from app.core.redaction import redact
from app.db.base import db_healthcheck, make_engine, make_session_factory, session_scope
from app.db.migrate import upgrade_to_head
from app.evidence.store import EvidenceStore
from app.observability import metrics as m
from app.observability.resources import ResourceSampler
from app.repositories.misc import get_setting, record_health, seed_if_empty
from app.services.live_state import LiveState
from app.services.ws_hub import WebSocketHub

log = logging.getLogger("api")
FRONTEND_DIST = REPO_ROOT / "frontend" / "dist"


def _load_dotenv_into_environ() -> None:
    """Camera credential_refs resolve from os.environ; make .env values visible (never overriding real env)."""
    p = REPO_ROOT / ".env"
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            k, sep, v = line.partition("=")
            if sep and not line.lstrip().startswith("#"):
                os.environ.setdefault(k.strip(), v.strip())


def build_context() -> AppContext:
    s = get_settings()
    cfg = load_app_config(s)
    engine = make_engine(s.database_url, pool_size=10)
    return AppContext(settings=s, cfg=cfg, engine=engine, sf=make_session_factory(engine), live=LiveState(),
                      hub=WebSocketHub(), evidence=EvidenceStore(s.resolve(s.ppe_evidence_dir), cfg.evidence))


async def _resource_loop(c: AppContext) -> None:
    sampler = ResourceSampler("api", {"evidence": c.evidence.root})
    while True:
        c.api_resources = await asyncio.to_thread(sampler.sample)
        ok = await asyncio.to_thread(db_healthcheck, c.engine)
        c.live.db_up = ok
        m.db_up.set(1 if ok else 0)
        await asyncio.sleep(5.0)


async def _retention_loop(c: AppContext) -> None:
    while True:
        await asyncio.sleep(c.cfg.evidence.retention_check_interval_s)
        try:
            def run():
                with c.sf() as s:
                    days = int(get_setting(s, "evidence_retention_days", c.cfg.evidence.retention_days))
                return c.evidence.delete_older_than(days)
            files, freed = await asyncio.to_thread(run)
            if files:
                log.info("retention deleted %d evidence files (%d bytes)", files, freed)
        except Exception:  # noqa: BLE001
            log.warning("retention run failed", exc_info=True)


def _on_worker_message(c: AppContext, msg: dict) -> None:
    if msg.get("type") == "event" and c.dispatcher is not None:
        c.dispatcher.wake()


def _mark_cameras_unknown(c: AppContext, reason: str) -> None:
    """Worker gone => camera health unknown; recorded so uptime is never overstated (spec 11.5)."""
    try:
        with session_scope(c.sf) as s:
            from app.models import Camera

            for cam in s.query(Camera).filter(Camera.enabled.is_(True)):
                record_health(s, cam.id, "OFFLINE", reason)
    except Exception:  # noqa: BLE001
        log.debug("could not record worker-down health", exc_info=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging("api")
    _load_dotenv_into_environ()
    c: AppContext = app.state.ctx
    upgrade_to_head(c.settings.database_url)
    rules, zones = load_rule_seeds(c.settings)
    with session_scope(c.sf) as s:
        pw = c.settings.ppe_admin_password.get_secret_value() if c.settings.ppe_admin_password else None
        added = seed_if_empty(s, load_camera_seeds(c.settings), rules, zones, pw)
    log.info("startup seed: %s", added)
    loop = asyncio.get_running_loop()
    c.hub.bind(loop)
    channels = {"dashboard": lambda ev: c.hub.broadcast_async({"type": "event_created", "event": ev})}
    if c.cfg.alerts.local_sound_enabled:
        channels["local_sound"] = local_sound_channel
    c.dispatcher = OutboxDispatcher(c.sf, c.cfg.alerts, channels)
    c.dispatcher.start()
    tasks = [asyncio.create_task(routes_ws.periodic_broadcasts(c)), asyncio.create_task(_resource_loop(c)),
             asyncio.create_task(_retention_loop(c))]
    if c.cfg.mode == "demo":
        from app.services.simulator import SimulatorClient

        c.simulator = SimulatorClient(c.cfg.demo)
        if c.cfg.demo.autostart_simulator and app.state.start_worker:
            up = await asyncio.to_thread(c.simulator.ensure_running)
            log.info("RTSP simulator %s", "ready" if up else "NOT available - demo cameras will stay OFFLINE")
    if app.state.start_worker:
        from app.workers.supervisor import WorkerSupervisor

        sup = WorkerSupervisor(c.live, on_message=lambda msg: _on_worker_message(c, msg))
        orig_down = c.live.worker_down

        def down_hook() -> None:
            orig_down()
            _mark_cameras_unknown(c, "pipeline worker down")

        c.live.worker_down = down_hook  # type: ignore[method-assign]
        c.supervisor = sup
        sup.start()
    log.info("PPE Intelligence API %s ready (mode=%s)", __version__, c.cfg.mode)
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()
        await c.dispatcher.stop()
        if c.supervisor is not None:
            await asyncio.to_thread(c.supervisor.stop)
        if c.simulator is not None:
            await asyncio.to_thread(c.simulator.stop_if_owned)
        c.engine.dispose()


def _err(status: int, code: str, message: str, details=None) -> JSONResponse:
    return JSONResponse({"error": {"code": code, "message": message, "details": details},
                         "request_id": uuid.uuid4().hex[:12]}, status_code=status)


def create_app(start_worker: bool = True, context: AppContext | None = None) -> FastAPI:
    app = FastAPI(title="PPE Intelligence API", version=__version__, lifespan=lifespan,
                  docs_url="/api/docs", openapi_url="/api/openapi.json", redoc_url=None)
    app.state.ctx = context or build_context()
    app.state.start_worker = start_worker

    @app.exception_handler(ApiError)
    async def _api_err(_r: Request, e: ApiError):
        return _err(e.status, e.code, e.message, e.details)

    @app.exception_handler(RequestValidationError)
    async def _val_err(_r: Request, e: RequestValidationError):
        details = [{k: v for k, v in err.items() if k in ("loc", "msg", "type")} for err in e.errors()]
        return _err(422, "validation_error", "request validation failed", details)

    @app.exception_handler(StarletteHTTPException)
    async def _http_err(_r: Request, e: StarletteHTTPException):
        code = {401: "unauthorized", 403: "forbidden", 404: "not_found", 405: "bad_request"}.get(e.status_code,
                                                                                                "bad_request")
        return _err(e.status_code, code, str(e.detail))

    @app.exception_handler(Exception)
    async def _unhandled(_r: Request, e: Exception):
        log.error("unhandled error: %s", redact(f"{type(e).__name__}: {e}"))
        return _err(503 if "OperationalError" in type(e).__name__ else 500,
                    "unavailable" if "OperationalError" in type(e).__name__ else "internal_error",
                    "database unavailable" if "OperationalError" in type(e).__name__ else "internal server error")

    @app.middleware("http")
    async def _timing(request: Request, call_next):
        t0 = time.perf_counter()
        resp = await call_next(request)
        resp.headers["X-Response-Time-ms"] = f"{1000 * (time.perf_counter() - t0):.1f}"
        return resp

    for r in (routes_auth.router, routes_cameras.router, routes_events.router, routes_system.router,
              routes_demo.router, routes_ws.router):
        app.include_router(r)

    @app.get("/metrics", include_in_schema=False)
    def prometheus() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    if FRONTEND_DIST.exists():  # offline dashboard (FR-OFF-003): no CDN, served by the API itself
        app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")

        @app.get("/{path:path}", include_in_schema=False)
        def spa(path: str):
            f = (FRONTEND_DIST / path).resolve()
            if path and f.is_file() and f.is_relative_to(FRONTEND_DIST.resolve()):
                return FileResponse(f)
            return FileResponse(FRONTEND_DIST / "index.html")

    return app


def factory() -> FastAPI:
    """uvicorn app.main:factory --factory   (PPE_START_WORKER=0 runs the API without the pipeline worker)."""
    return create_app(start_worker=os.environ.get("PPE_START_WORKER", "1") != "0")
