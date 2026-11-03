"""ORM -> API JSON (docs/system-design/10-api-and-websocket-contracts.md). Credentials are never serialised."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from app.models import AuditLog, Camera, Event, EventAction, PPERule, User, Zone


def iso(dt: datetime | None) -> str | None:
    return None if dt is None else dt.isoformat().replace("+00:00", "Z")


def _secs(a: datetime | None, b: datetime | None) -> float | None:
    return None if a is None or b is None else round((b - a).total_seconds(), 3)


def user_out(u: User) -> dict[str, Any]:
    return {"id": str(u.id), "username": u.username, "role": u.role, "enabled": u.enabled,
            "created_at": iso(u.created_at)}


def has_credentials(ref: str | None) -> bool:
    if not ref or not ref.startswith("env:"):
        return False
    prefix = ref[4:]
    return bool(os.environ.get(f"{prefix}_USER")) and os.environ.get(f"{prefix}_PASS") is not None


def camera_out(c: Camera, status: dict[str, Any] | None, open_violations: int) -> dict[str, Any]:
    return {"id": c.id, "name": c.name, "stream_uri": c.stream_uri, "credential_ref": c.credential_ref,
            "has_credentials": has_credentials(c.credential_ref), "transport": c.transport, "enabled": c.enabled,
            "site": c.site, "location": c.location, "target_fps": c.target_fps, "created_at": iso(c.created_at),
            "updated_at": iso(c.updated_at), "status": status, "open_violations": open_violations}


def zone_out(z: Zone) -> dict[str, Any]:
    return {"id": z.id, "camera_id": z.camera_id, "name": z.name, "kind": z.kind, "polygon": z.polygon,
            "enabled": z.enabled}


def rule_out(r: PPERule, target_fps: float) -> dict[str, Any]:
    tp = r.temporal_policy
    confirm_s = (tp["confirm_n"] - 1) / target_fps  # N observations span N-1 sampling intervals
    if tp["mode"] == "duration":
        confirm_s = tp["min_duration_s"]
    elif tp["mode"] == "both":
        confirm_s = max(confirm_s, tp["min_duration_s"])
    return {"id": r.id, "name": r.name, "ppe_class": r.ppe_class, "camera_id": r.camera_id, "zone_id": r.zone_id,
            "severity": r.severity, "enabled": r.enabled, "thresholds": r.thresholds, "temporal_policy": tp,
            "effective_confirm_seconds": round(confirm_s, 2), "updated_at": iso(r.updated_at)}


def event_out(e: Event, camera_name: str | None = None) -> dict[str, Any]:
    return {
        "id": str(e.id), "camera_id": e.camera_id, "camera_name": camera_name or e.camera_id, "track_id": e.track_id,
        "zone_id": e.zone_id, "rule_id": e.rule_id, "episode_id": e.episode_id, "status": e.status,
        "severity": e.severity, "reason": e.reason, "confidence_summary": e.confidence_summary,
        "first_seen_at": iso(e.first_seen_at), "confirmed_at": iso(e.confirmed_at),
        "frame_captured_at": iso(e.frame_captured_at), "persisted_at": iso(e.persisted_at),
        "last_seen_at": iso(e.last_seen_at), "episode_ended_at": iso(e.episode_ended_at),
        "reconfirm_count": e.reconfirm_count, "has_evidence": bool(e.evidence_path),
        "evidence_error": e.evidence_error, "model_versions": e.model_versions, "feedback": e.feedback,
        "feedback_note": e.feedback_note, "acknowledged_by": e.acknowledged_by,
        "acknowledged_at": iso(e.acknowledged_at), "resolved_by": e.resolved_by, "resolved_at": iso(e.resolved_at),
        "latency": {"time_to_confirm_s": _secs(e.first_seen_at, e.confirmed_at),
                    "confirm_to_persist_s": _secs(e.confirmed_at, e.persisted_at),
                    "frame_to_persist_s": _secs(e.frame_captured_at, e.persisted_at),
                    "persist_to_dashboard_s": _secs(e.persisted_at, e.dashboard_rendered_at)},
    }


def action_out(a: EventAction) -> dict[str, Any]:
    return {"id": a.id, "action": a.action, "actor": a.actor_id, "note": a.note, "created_at": iso(a.created_at)}


def audit_out(a: AuditLog) -> dict[str, Any]:
    return {"id": a.id, "actor": a.actor_id, "action": a.action, "resource": a.resource,
            "created_at": iso(a.created_at), "metadata": a.meta}
