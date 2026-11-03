from __future__ import annotations

import csv
import io
import time
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app import __version__
from app.api.deps import ApiError, AppContext, ctx, db, require
from app.db.base import db_healthcheck
from app.models import Camera, CameraHealthRecord, Event, User
from app.observability import metrics as m
from app.repositories import events as ev_repo
from app.repositories import outbox as outbox_repo
from app.repositories.misc import audit, get_setting, list_audit, set_setting
from app.schemas.requests import FaultIn, SettingsPatch
from app.schemas.serializers import audit_out
from app.services.live_state import iso

router = APIRouter(tags=["system"])
LIVE_STATES = ("ONLINE", "DEGRADED")


@router.get("/api/v1/health")
def health(c: AppContext = Depends(ctx)):
    return {"status": "ok", "version": __version__, "uptime_s": round(time.time() - c.live.started_at, 1)}


@router.get("/api/v1/ready")
def ready(c: AppContext = Depends(ctx)):
    db_ok = db_healthcheck(c.engine)
    c.live.db_up = db_ok
    m.db_up.set(1 if db_ok else 0)
    w = c.live.worker
    free = c.evidence.free_bytes()
    checks = {
        "database": {"ok": db_ok, "detail": "reachable" if db_ok else "unreachable"},
        "worker": {"ok": bool(w["up"]), "detail": f"pid {w['pid']}" if w["up"] else "not running"},
        "models": {"ok": bool(w.get("model_versions")), "detail": ", ".join(w.get("model_versions", {}).values())},
        "disk": {"ok": free >= c.cfg.evidence.min_free_bytes, "detail": f"{free / 1e9:.1f} GB free"},
    }
    ok = all(v["ok"] for v in checks.values())
    return JSONResponse({"ready": ok, "checks": checks}, status_code=200 if ok else 503)


def build_metrics(c: AppContext, s: Session | None) -> dict:
    stats, worker, history = c.live.snapshot_stats()
    cams = c.live.all_cameras()
    try:
        if not c.live.db_up:
            raise RuntimeError("database marked down")
        with c.sf() as s2:
            all_cams = [(x.id, x.enabled) for x in s2.scalars(select(Camera))]
    except Exception:  # noqa: BLE001 - DB down: fall back to the cameras the worker reports
        all_cams = [(k, True) for k in cams]
        c.live.db_up = False
    counts = {k: 0 for k in ("online", "degraded", "offline", "error", "disabled")}
    for cam_id, enabled in all_cams:
        st = (cams.get(cam_id) or {}).get("state")
        if not enabled:
            counts["disabled"] += 1
        elif st == "ONLINE":
            counts["online"] += 1
        elif st == "DEGRADED":
            counts["degraded"] += 1
        elif st == "ERROR":
            counts["error"] += 1
        else:
            counts["offline"] += 1  # OFFLINE / CONNECTING / RECOVERING / unknown (worker down)
    active = events_hour = None
    outbox_counts: dict[str, int] = {}
    if s is not None and c.live.db_up:
        try:
            active = sum(ev_repo.count_open_by_camera(s).values())
            events_hour = ev_repo.count_since(s, datetime.now(UTC) - timedelta(hours=1))
            outbox_counts = outbox_repo.counts(s)
        except Exception:  # noqa: BLE001 - DB down: report what is live
            c.live.db_up = False
    r = (stats or {}).get("resources") or {}
    p2d = c.live.persist_to_dashboard.quantiles()
    lat_conf = [x for x in cams.values() if x.get("frame_age_at_inference_p95_s") is not None]
    return {
        "generated_at": iso(time.time()), "mode": c.cfg.mode,
        "aggregate": {"cameras_total": len(all_cams), **counts,
                      "processed_fps": (stats or {}).get("aggregate_processed_fps", 0.0) if worker["up"] else 0.0,
                      "target_fps_total": sum(x.get("target_fps", 0) for x in cams.values()),
                      "active_violations": active, "events_last_hour": events_hour},
        "cameras": list(cams.values()),
        "inference": (stats or {}).get("inference", {}),
        "batch_size_mean": (stats or {}).get("batch_size_mean", 0.0),
        "latency": {"frame_age_at_inference_p95_s": max((x["frame_age_at_inference_p95_s"] for x in lat_conf),
                                                        default=None),
                    "persist_to_dashboard_p95_s": p2d["p95"] if p2d else None},
        "resources": {"api_rss_bytes": c.api_resources.get("rss_bytes"),
                      "worker_rss_bytes": r.get("rss_bytes"), "worker_cpu_percent": r.get("cpu_percent"),
                      "system_cpu_percent": r.get("system_cpu_percent"),
                      "system_memory_used_bytes": r.get("system_memory_used_bytes"),
                      "gpu_util_percent": r.get("gpu_util_percent"),
                      "gpu_memory_used_bytes": r.get("gpu_memory_used_bytes"),
                      "gpu_memory_total_bytes": r.get("gpu_memory_total_bytes"),
                      "gpu_temperature_c": r.get("gpu_temperature_c"), "gpu_power_w": r.get("gpu_power_w"),
                      "torch_cuda_allocated_bytes": r.get("torch_cuda_allocated_bytes"),
                      "disk_free_bytes": c.evidence.free_bytes()},
        "worker": {"up": worker["up"], "pid": worker["pid"], "run_id": worker["run_id"],
                   "uptime_s": round(time.time() - worker["started_at"], 1) if worker["up"] and worker["started_at"]
                   else None, "restarts": worker["restarts"], "device": worker.get("device")},
        "database": {"up": c.live.db_up},
        "outbox": {"pending": outbox_counts.get("PENDING", 0), "failed": outbox_counts.get("FAILED", 0)},
        "spool": {"pending": (stats or {}).get("spool_pending", 0)},
        "history": history,
    }


@router.get("/api/v1/metrics")
def metrics_summary(c: AppContext = Depends(ctx), s: Session = Depends(db), _u: User = Depends(require("viewer"))):
    return build_metrics(c, s)


@router.get("/api/v1/system")
def system(c: AppContext = Depends(ctx), _u: User = Depends(require("viewer"))):
    return {"version": __version__, "mode": c.cfg.mode, "models": c.live.worker.get("model_versions", {}),
            "device": c.live.worker.get("device"), "started_at": iso(c.live.started_at),
            "config": {"target_fps": c.cfg.pipeline.target_fps, "max_batch": c.cfg.pipeline.max_batch,
                       "head_roi": c.cfg.association.head_roi, "classifier_enabled": c.cfg.association.use_classifier}}


@router.post("/api/v1/system/worker/restart", status_code=202)
def restart_worker(c: AppContext = Depends(ctx), s: Session = Depends(db), u: User = Depends(require("admin"))):
    if c.supervisor is None:
        raise ApiError(503, "unavailable", "worker supervisor not running")
    audit(s, u.username, "worker_restart", "worker")
    c.supervisor.request_restart()
    return {"accepted": True}


@router.get("/api/v1/settings")
def get_settings_(c: AppContext = Depends(ctx), s: Session = Depends(db), _u: User = Depends(require("viewer"))):
    return {"evidence_retention_days": get_setting(s, "evidence_retention_days", c.cfg.evidence.retention_days),
            "evidence_min_free_bytes": c.cfg.evidence.min_free_bytes}


@router.patch("/api/v1/settings")
def patch_settings(body: SettingsPatch, c: AppContext = Depends(ctx), s: Session = Depends(db),
                   u: User = Depends(require("admin"))):
    if body.evidence_retention_days is not None:
        set_setting(s, "evidence_retention_days", body.evidence_retention_days)
        audit(s, u.username, "settings_update", "settings", body.model_dump(exclude_none=True))
    s.flush()
    return get_settings_(c, s, u)


@router.post("/api/v1/system/retention/run")
def run_retention(c: AppContext = Depends(ctx), s: Session = Depends(db), u: User = Depends(require("admin"))):
    days = int(get_setting(s, "evidence_retention_days", c.cfg.evidence.retention_days))
    files, freed = c.evidence.delete_older_than(days)
    audit(s, u.username, "retention_run", "evidence", {"days": days, "files": files})
    return {"deleted_files": files, "freed_bytes": freed}


@router.get("/api/v1/audit-logs")
def audit_logs(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0), s: Session = Depends(db),
               _u: User = Depends(require("admin"))):
    rows, total = list_audit(s, limit, offset)
    return {"items": [audit_out(a) for a in rows], "total": total, "limit": limit, "offset": offset}


def online_seconds(s: Session, camera_id: str, since: datetime, until: datetime) -> float:
    """Integrate ONLINE/DEGRADED time from the camera_health transition history within [since, until]."""
    prev = s.scalar(select(CameraHealthRecord).where(CameraHealthRecord.camera_id == camera_id,
                                                     CameraHealthRecord.recorded_at <= since)
                    .order_by(CameraHealthRecord.recorded_at.desc()).limit(1))
    rows = list(s.scalars(select(CameraHealthRecord).where(CameraHealthRecord.camera_id == camera_id,
                                                           CameraHealthRecord.recorded_at > since,
                                                           CameraHealthRecord.recorded_at < until)
                          .order_by(CameraHealthRecord.recorded_at)))
    total, state, t = 0.0, prev.state if prev else None, since
    for r in rows:
        if state in LIVE_STATES:
            total += (r.recorded_at - t).total_seconds()
        state, t = r.state, r.recorded_at
    if state in LIVE_STATES:
        total += (until - t).total_seconds()
    return total


@router.get("/api/v1/evaluation/report")
def evaluation_report(since: datetime | None = None, until: datetime | None = None,
                      format: str = Query("json", pattern="^(json|csv)$"), s: Session = Depends(db),
                      _u: User = Depends(require("viewer"))):
    """Operator-labelled alert quality (spec 12.4/12.5): FP per camera-hour and per system-hour."""
    until = until or datetime.now(UTC)
    if since is None:
        first = s.scalar(select(CameraHealthRecord.recorded_at).order_by(CameraHealthRecord.recorded_at).limit(1))
        since = first or until - timedelta(hours=1)
    rows = []
    for cam in s.scalars(select(Camera).order_by(Camera.id)):
        evs = list(s.scalars(select(Event).where(Event.camera_id == cam.id, Event.confirmed_at >= since,
                                                 Event.confirmed_at < until)))
        hours = online_seconds(s, cam.id, since, until) / 3600
        tp = sum(e.feedback == "TRUE_POSITIVE" for e in evs)
        fp = sum(e.feedback == "FALSE_POSITIVE" for e in evs)
        rows.append({"camera_id": cam.id, "events": len(evs), "true_positive": tp, "false_positive": fp,
                     "unlabelled": len(evs) - tp - fp, "online_hours": round(hours, 3),
                     "fp_per_camera_hour": round(fp / hours, 3) if hours > 0 else None,
                     "events_per_camera_hour": round(len(evs) / hours, 3) if hours > 0 else None})
    wall_h = (until - since).total_seconds() / 3600
    fp_total = sum(r["false_positive"] for r in rows)
    summary = {"since": since.isoformat(), "until": until.isoformat(), "system_hours": round(wall_h, 3),
               "camera_hours": round(sum(r["online_hours"] for r in rows), 3), "events": sum(r["events"] for r in rows),
               "labelled": sum(r["true_positive"] + r["false_positive"] for r in rows), "false_positive": fp_total,
               "fp_per_system_hour": round(fp_total / wall_h, 3) if wall_h > 0 else None,
               "note": "Labels are operator feedback (evaluation metadata), not ground truth; unlabelled events "
                       "are excluded from FP rates. Event recall needs labelled violation episodes and is not "
                       "computed here."}
    if format == "csv":
        buf = io.StringIO()
        w = csv.DictWriter(buf, fieldnames=list(rows[0]) if rows else ["camera_id"])
        w.writeheader()
        w.writerows(rows)
        return PlainTextResponse(buf.getvalue(), media_type="text/csv")
    return {"summary": summary, "cameras": rows}


@router.post("/api/v1/system/faults", status_code=202)
def inject_fault(body: FaultIn, c: AppContext = Depends(ctx), s: Session = Depends(db),
                 u: User = Depends(require("admin"))):
    if not (c.settings.ppe_enable_fault_injection or c.cfg.mode == "demo"):
        raise ApiError(403, "forbidden",
                       "fault injection is only enabled in demo mode or with PPE_ENABLE_FAULT_INJECTION=1")
    audit(s, u.username, "fault_injection", f"fault:{body.kind}", body.model_dump(exclude_none=True))
    c.send_worker({"type": "fault", **body.model_dump(exclude_none=True)})
    return {"accepted": True}
