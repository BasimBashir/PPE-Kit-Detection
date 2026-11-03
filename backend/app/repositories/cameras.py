from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.camera.uri import validate_credential_ref, validate_stream_uri
from app.models import Camera, Event

EDITABLE = ("name", "stream_uri", "credential_ref", "transport", "enabled", "site", "location", "target_fps")


def list_cameras(s: Session) -> list[Camera]:
    return list(s.scalars(select(Camera).order_by(Camera.id)))


def get_camera(s: Session, camera_id: str) -> Camera | None:
    return s.get(Camera, camera_id)


def _validate(data: dict[str, Any]) -> None:
    if "stream_uri" in data:
        data["stream_uri"] = validate_stream_uri(data["stream_uri"])
    if "credential_ref" in data:
        data["credential_ref"] = validate_credential_ref(data["credential_ref"])


def create_camera(s: Session, data: dict[str, Any]) -> Camera:
    _validate(data)
    cam = Camera(id=data["id"], **{k: data[k] for k in EDITABLE if k in data})
    s.add(cam)
    s.flush()
    return cam


def update_camera(s: Session, cam: Camera, data: dict[str, Any]) -> Camera:
    _validate(data)
    for k in EDITABLE:
        if k in data:
            setattr(cam, k, data[k])
    s.flush()
    return cam


def event_count(s: Session, camera_id: str) -> int:
    return s.scalar(select(func.count()).select_from(Event).where(Event.camera_id == camera_id)) or 0


def delete_camera(s: Session, cam: Camera) -> None:
    s.delete(cam)
    s.flush()
