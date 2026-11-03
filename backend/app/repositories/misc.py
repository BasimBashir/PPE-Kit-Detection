"""Audit log, key/value settings, model provenance, camera-health history and first-start seeding."""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.config.settings import CameraSeed, RuleSeed, ZoneSeed
from app.core.redaction import redact
from app.models import AuditLog, Camera, CameraHealthRecord, ModelVersion, PPERule, Setting, User
from app.repositories import rules as rules_repo
from app.repositories.users import create_user


def audit(s: Session, actor: str, action: str, resource: str, meta: dict[str, Any] | None = None) -> None:
    clean = {k: redact(v) if isinstance(v, str) else v for k, v in (meta or {}).items()}
    s.add(AuditLog(actor_id=actor, action=action, resource=resource, meta=clean))


def list_audit(s: Session, limit: int, offset: int) -> tuple[list[AuditLog], int]:
    total = s.scalar(select(func.count()).select_from(AuditLog)) or 0
    rows = s.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(limit).offset(offset))
    return list(rows), total


def get_setting(s: Session, key: str, default: Any = None) -> Any:
    row = s.get(Setting, key)
    return default if row is None else row.value


def set_setting(s: Session, key: str, value: Any) -> None:
    row = s.get(Setting, key)
    if row is None:
        s.add(Setting(key=key, value=value))
    else:
        row.value = value


def record_model_versions(s: Session, versions: dict[str, str], license_: str = "AGPL-3.0") -> None:
    for v in versions.values():
        if s.get(ModelVersion, v) is None:
            name, _, h = v.partition("@")
            s.add(ModelVersion(id=v, model_name=name, version=h, artifact_hash=h, license=license_, config={}))


def record_health(s: Session, camera_id: str, state: str, reason: str) -> None:
    s.add(CameraHealthRecord(camera_id=camera_id, state=state, reason=redact(reason)))


def seed_if_empty(s: Session, cameras: list[CameraSeed], rules: list[RuleSeed], zones: list[ZoneSeed],
                  admin_password: str | None) -> dict[str, int]:
    """Idempotent: inserts only rows whose ids are missing; creates `admin` only when no users exist."""
    added = {"cameras": 0, "rules": 0, "zones": 0, "users": 0}
    for c in cameras:
        if s.get(Camera, c.id) is None:
            s.add(Camera(**c.model_dump()))
            added["cameras"] += 1
    s.flush()
    for z in zones:
        if rules_repo.get_zone(s, z.id) is None:
            rules_repo.upsert_zone(s, z)
            added["zones"] += 1
    for r in rules:
        if s.get(PPERule, r.id) is None:
            rules_repo.upsert_rule(s, r)
            added["rules"] += 1
    if admin_password and not s.scalar(select(func.count()).select_from(User)):
        create_user(s, "admin", admin_password, "admin")
        added["users"] += 1
    s.flush()
    return added
