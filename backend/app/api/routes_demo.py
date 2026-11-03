"""Demo control (admin, demo mode only): choose how many simulated cameras stream, unplug/stall/degrade any of them,
restart the simulated NVR. Lets a reviewer exercise the real pipeline without physical cameras."""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.api.deps import ApiError, AppContext, ctx, db, require
from app.models import Camera, User
from app.repositories.misc import audit
from app.services.simulator import SimulatorUnavailable

router = APIRouter(prefix="/api/v1/demo", tags=["demo"])


class StreamsIn(BaseModel):
    count: int = Field(ge=0, le=10)


class NetworkIn(BaseModel):
    kind: Literal["latency", "bandwidth", "timeout", "reset_peer", "clear"]
    latency: int | None = Field(default=None, ge=0, le=10000)
    jitter: int | None = Field(default=None, ge=0, le=10000)
    rate: int | None = Field(default=None, ge=1)
    timeout: int | None = Field(default=None, ge=0, le=600000)


def _sim(c: AppContext):
    if c.cfg.mode != "demo" or c.simulator is None:
        raise ApiError(409, "conflict", "demo controls are only available in demo mode (configs/app.yaml mode: demo)")
    return c.simulator


async def _call(c: AppContext, method: str, path: str, body: dict | None = None):
    try:
        return await asyncio.to_thread(_sim(c).call, method, path, body)
    except SimulatorUnavailable as e:
        raise ApiError(503, "unavailable", f"{e} - start it with: python scripts/demo/rtsp_simulator.py serve") from e


@router.get("/streams")
async def streams(c: AppContext = Depends(ctx), _u: User = Depends(require("admin"))):
    return await _call(c, "GET", "/streams")


@router.post("/streams")
async def set_streams(body: StreamsIn, c: AppContext = Depends(ctx), s: Session = Depends(db),
                      u: User = Depends(require("admin"))):
    """Publish the first N demo clips and enable exactly those cameras (others disabled, not deleted)."""
    out = await _call(c, "POST", "/streams", {"count": body.count})
    live = {x["camera_id"] for x in out["cameras"] if x["wanted"]}
    for cam in s.scalars(select(Camera)):
        if cam.stream_uri.rsplit("/", 1)[-1] in {x["camera_id"] for x in out["cameras"]}:  # simulated cameras only
            cam.enabled = cam.id in live
    audit(s, u.username, "demo_set_streams", "simulator", {"count": body.count})
    s.commit()
    c.send_worker({"type": "reload_cameras"})
    return out


@router.post("/streams/{camera_id}/{action}")
async def stream_action(camera_id: str, action: Literal["stop", "start", "stall", "resume", "disable_proxy",
                                                         "enable_proxy"],
                        c: AppContext = Depends(ctx), s: Session = Depends(db), u: User = Depends(require("admin"))):
    out = await _call(c, "POST", f"/cameras/{camera_id}/{action}")
    audit(s, u.username, f"demo_{action}", f"camera:{camera_id}")
    return out


@router.post("/streams/{camera_id}/network")
async def network(camera_id: str, body: NetworkIn, c: AppContext = Depends(ctx), s: Session = Depends(db),
                  u: User = Depends(require("admin"))):
    out = await _call(c, "POST", f"/cameras/{camera_id}/network", body.model_dump(exclude_none=True))
    audit(s, u.username, "demo_network", f"camera:{camera_id}", body.model_dump(exclude_none=True))
    return out


@router.post("/nvr/restart")
async def nvr_restart(c: AppContext = Depends(ctx), s: Session = Depends(db), u: User = Depends(require("admin"))):
    out = await _call(c, "POST", "/nvr/restart")
    audit(s, u.username, "demo_nvr_restart", "simulator")
    return out
