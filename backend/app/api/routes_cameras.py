from __future__ import annotations

import asyncio
import base64
import time

import av
from fastapi import APIRouter, Depends, Response
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import ApiError, AppContext, ctx, db, require
from app.camera.source import file_path_of
from app.camera.uri import resolve_uri
from app.config.settings import ZoneSeed
from app.core.redaction import redact
from app.models import User
from app.repositories import cameras as cam_repo
from app.repositories import events as ev_repo
from app.repositories import rules as rules_repo
from app.repositories.misc import audit
from app.schemas.requests import CameraCreate, CameraUpdate, ZoneIn, ZoneUpdate
from app.schemas.serializers import camera_out, zone_out
from app.services.live_state import camera_status_out, iso

router = APIRouter(prefix="/api/v1", tags=["cameras"])


def _cam_or_404(s: Session, camera_id: str):
    cam = cam_repo.get_camera(s, camera_id)
    if cam is None:
        raise ApiError(404, "not_found", f"camera {camera_id} not found")
    return cam


def _out(c: AppContext, s: Session, cam, open_counts: dict[str, int] | None = None):
    counts = open_counts if open_counts is not None else ev_repo.count_open_by_camera(s)
    return camera_out(cam, camera_status_out(c.live.camera(cam.id)), counts.get(cam.id, 0))


@router.get("/cameras")
def list_cameras(c: AppContext = Depends(ctx), s: Session = Depends(db), _u: User = Depends(require("viewer"))):
    counts = ev_repo.count_open_by_camera(s)
    return [_out(c, s, cam, counts) for cam in cam_repo.list_cameras(s)]


@router.post("/cameras", status_code=201)
def create_camera(body: CameraCreate, c: AppContext = Depends(ctx), s: Session = Depends(db),
                  u: User = Depends(require("admin"))):
    try:
        cam = cam_repo.create_camera(s, body.model_dump(exclude_none=True))
    except IntegrityError as e:
        raise ApiError(409, "conflict", f"camera {body.id} already exists") from e
    audit(s, u.username, "camera_create", f"camera:{cam.id}", {"stream_uri": cam.stream_uri})
    s.commit()
    c.send_worker({"type": "reload_cameras"})
    return _out(c, s, cam)


@router.get("/cameras/{camera_id}")
def get_camera(camera_id: str, c: AppContext = Depends(ctx), s: Session = Depends(db),
               _u: User = Depends(require("viewer"))):
    return _out(c, s, _cam_or_404(s, camera_id))


@router.patch("/cameras/{camera_id}")
def update_camera(camera_id: str, body: CameraUpdate, c: AppContext = Depends(ctx), s: Session = Depends(db),
                  u: User = Depends(require("admin"))):
    cam = _cam_or_404(s, camera_id)
    data = body.model_dump(exclude_unset=True)
    cam_repo.update_camera(s, cam, data)
    audit(s, u.username, "camera_update", f"camera:{cam.id}", {k: str(v) for k, v in data.items()})
    s.commit()
    c.send_worker({"type": "reload_cameras"})
    return _out(c, s, cam)


@router.delete("/cameras/{camera_id}", status_code=204)
def delete_camera(camera_id: str, c: AppContext = Depends(ctx), s: Session = Depends(db),
                  u: User = Depends(require("admin"))):
    cam = _cam_or_404(s, camera_id)
    if cam_repo.event_count(s, camera_id):
        raise ApiError(409, "conflict", "camera has recorded events; disable it instead of deleting (audit trail)")
    cam_repo.delete_camera(s, cam)
    audit(s, u.username, "camera_delete", f"camera:{camera_id}")
    s.commit()
    c.send_worker({"type": "reload_cameras"})


def _probe(uri: str, transport: str) -> dict:
    t0 = time.perf_counter()
    if uri.startswith("file:"):
        cont = av.open(str(file_path_of(uri)))
    else:
        cont = av.open(uri, options={"rtsp_transport": transport}, timeout=(5.0, 5.0))
    try:
        st = cont.streams.video[0]
        frame = next(cont.decode(st))
        rate = st.average_rate or st.guessed_rate
        return {"ok": True, "latency_ms": round(1000 * (time.perf_counter() - t0), 1), "codec": st.codec_context.name,
                "width": frame.width, "height": frame.height, "fps": float(rate) if rate else None, "error": None}
    finally:
        cont.close()


@router.post("/cameras/{camera_id}/test")
async def test_camera(camera_id: str, c: AppContext = Depends(ctx), u: User = Depends(require("admin"))):
    with c.sf() as s:
        cam = _cam_or_404(s, camera_id)
        uri_raw, ref, transport = cam.stream_uri, cam.credential_ref, cam.transport
    try:
        uri = resolve_uri(uri_raw, ref)
        return await asyncio.wait_for(asyncio.to_thread(_probe, uri, transport), timeout=15)
    except Exception as e:  # noqa: BLE001 - report, redacted, never leak credentials (FR-CAM-008)
        return {"ok": False, "latency_ms": None, "codec": None, "width": None, "height": None, "fps": None,
                "error": redact(f"{type(e).__name__}: {e}")}


@router.get("/cameras/{camera_id}/snapshot")
def snapshot(camera_id: str, c: AppContext = Depends(ctx), _u: User = Depends(require("viewer"))):
    p = c.live.preview(camera_id)
    if p is None:
        raise ApiError(404, "not_found", "no live frame for this camera")
    return Response(p["jpeg"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})


@router.get("/cameras/{camera_id}/frame")
def frame(camera_id: str, c: AppContext = Depends(ctx), _u: User = Depends(require("viewer"))):
    p = c.live.preview(camera_id)
    if p is None:
        raise ApiError(404, "not_found", "no live frame for this camera")
    return {"frame_id": p["frame_id"], "captured_at": iso(p["captured_at"]), "width": p["width"],
            "height": p["height"], "image_b64": base64.b64encode(p["jpeg"]).decode(), "tracks": p["tracks"],
            "detections": p["detections"]}


@router.get("/cameras/{camera_id}/zones")
def list_zones(camera_id: str, s: Session = Depends(db), _u: User = Depends(require("viewer"))):
    _cam_or_404(s, camera_id)
    return [zone_out(z) for z in rules_repo.list_zones(s, camera_id)]


@router.post("/cameras/{camera_id}/zones", status_code=201)
def create_zone(camera_id: str, body: ZoneIn, c: AppContext = Depends(ctx), s: Session = Depends(db),
                u: User = Depends(require("admin"))):
    _cam_or_404(s, camera_id)
    if rules_repo.get_zone(s, body.id):
        raise ApiError(409, "conflict", f"zone {body.id} exists")
    z = rules_repo.upsert_zone(s, ZoneSeed(camera_id=camera_id, **body.model_dump()))
    audit(s, u.username, "zone_create", f"zone:{z.id}")
    s.commit()
    c.send_worker({"type": "reload_rules"})
    return zone_out(z)


@router.patch("/zones/{zone_id}")
def update_zone(zone_id: str, body: ZoneUpdate, c: AppContext = Depends(ctx), s: Session = Depends(db),
                u: User = Depends(require("admin"))):
    z = rules_repo.get_zone(s, zone_id)
    if z is None:
        raise ApiError(404, "not_found", "zone not found")
    merged = {**{k: getattr(z, k) for k in ("id", "camera_id", "name", "kind", "polygon", "enabled")},
              **body.model_dump(exclude_unset=True)}
    z = rules_repo.upsert_zone(s, ZoneSeed.model_validate(merged))
    audit(s, u.username, "zone_update", f"zone:{z.id}")
    s.commit()
    c.send_worker({"type": "reload_rules"})
    return zone_out(z)


@router.delete("/zones/{zone_id}", status_code=204)
def delete_zone(zone_id: str, c: AppContext = Depends(ctx), s: Session = Depends(db),
                u: User = Depends(require("admin"))):
    z = rules_repo.get_zone(s, zone_id)
    if z is None:
        raise ApiError(404, "not_found", "zone not found")
    rules_repo.delete_zone(s, z)
    audit(s, u.username, "zone_delete", f"zone:{zone_id}")
    s.commit()
    c.send_worker({"type": "reload_rules"})
