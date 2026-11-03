from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import ApiError, AppContext, ctx, db, require
from app.config.settings import RuleSeed
from app.models import Camera, User
from app.repositories import events as ev_repo
from app.repositories import rules as rules_repo
from app.repositories.misc import audit
from app.schemas.requests import FeedbackIn, NoteIn, RulePatch
from app.schemas.serializers import action_out, event_out, rule_out

router = APIRouter(prefix="/api/v1", tags=["events", "rules"])


def _event_or_404(s: Session, event_id: uuid.UUID):
    e = ev_repo.get_event(s, event_id)
    if e is None:
        raise ApiError(404, "not_found", "event not found")
    return e


def _out(s: Session, e) -> dict:
    cam = s.get(Camera, e.camera_id)
    return event_out(e, cam.name if cam else None)


@router.get("/events")
def list_events(camera_id: str | None = None,
                status: Literal["OPEN", "ACKNOWLEDGED", "RESOLVED"] | None = None,
                rule_id: str | None = None,
                feedback: Literal["TRUE_POSITIVE", "FALSE_POSITIVE", "NONE"] | None = None,
                since: datetime | None = None, until: datetime | None = None,
                limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                s: Session = Depends(db), _u: User = Depends(require("viewer"))):
    rows, total = ev_repo.list_events(s, ev_repo.EventFilter(camera_id, status, rule_id, feedback, since, until),
                                      limit, offset)
    names = {c.id: c.name for c in s.query(Camera).all()}
    return {"items": [event_out(e, names.get(e.camera_id)) for e in rows], "total": total, "limit": limit,
            "offset": offset}


@router.get("/events/{event_id}")
def get_event(event_id: uuid.UUID, s: Session = Depends(db), _u: User = Depends(require("viewer"))):
    e = _event_or_404(s, event_id)
    return {**_out(s, e), "actions": [action_out(a) for a in ev_repo.actions_for(s, e.id)]}


@router.get("/events/{event_id}/evidence")
def evidence(event_id: uuid.UUID, c: AppContext = Depends(ctx), s: Session = Depends(db),
             _u: User = Depends(require("viewer"))):
    e = _event_or_404(s, event_id)
    if not e.evidence_path:
        raise ApiError(404, "not_found", f"no evidence ({e.evidence_error or 'not stored'})")
    try:
        path = c.evidence.open(e.evidence_path)
    except FileNotFoundError as err:
        raise ApiError(404, "not_found", "evidence file missing (retention or disk issue)") from err
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


def _broadcast_update(c: AppContext, payload: dict) -> None:
    c.hub.broadcast({"type": "event_updated", "event": payload})


@router.post("/events/{event_id}/acknowledge")
def acknowledge(event_id: uuid.UUID, body: NoteIn | None = None, c: AppContext = Depends(ctx),
                s: Session = Depends(db), u: User = Depends(require("operator"))):
    e = _event_or_404(s, event_id)
    try:
        ev_repo.acknowledge(s, e, u.username, body.note if body else None)
    except ev_repo.StateConflict as err:
        raise ApiError(409, "conflict", str(err)) from err
    s.commit()
    out = _out(s, e)
    _broadcast_update(c, out)
    return out


@router.post("/events/{event_id}/resolve")
def resolve(event_id: uuid.UUID, body: NoteIn | None = None, c: AppContext = Depends(ctx),
            s: Session = Depends(db), u: User = Depends(require("operator"))):
    e = _event_or_404(s, event_id)
    try:
        ev_repo.resolve(s, e, u.username, body.note if body else None)
    except ev_repo.StateConflict as err:
        raise ApiError(409, "conflict", str(err)) from err
    s.commit()
    out = _out(s, e)
    _broadcast_update(c, out)
    return out


@router.post("/events/{event_id}/feedback")
def feedback(event_id: uuid.UUID, body: FeedbackIn, c: AppContext = Depends(ctx), s: Session = Depends(db),
             u: User = Depends(require("operator"))):
    e = _event_or_404(s, event_id)
    ev_repo.set_feedback(s, e, u.username, body.label, body.note)
    s.commit()
    out = _out(s, e)
    _broadcast_update(c, out)
    return out


# ---------------------------------------------------------------- rules


def _target_fps(c: AppContext) -> float:
    return c.cfg.pipeline.target_fps


@router.get("/rules")
def list_rules(c: AppContext = Depends(ctx), s: Session = Depends(db), _u: User = Depends(require("viewer"))):
    return [rule_out(r, _target_fps(c)) for r in rules_repo.list_rules(s)]


@router.post("/rules")
def upsert_rule(body: RuleSeed, c: AppContext = Depends(ctx), s: Session = Depends(db),
                u: User = Depends(require("admin"))):
    r = rules_repo.upsert_rule(s, body)
    audit(s, u.username, "rule_upsert", f"rule:{r.id}", {"temporal_policy": body.temporal_policy.model_dump()})
    s.commit()
    c.send_worker({"type": "reload_rules"})
    return rule_out(r, _target_fps(c))


@router.patch("/rules/{rule_id}")
def patch_rule(rule_id: str, body: RulePatch, c: AppContext = Depends(ctx), s: Session = Depends(db),
               u: User = Depends(require("admin"))):
    r = rules_repo.get_rule(s, rule_id)
    if r is None:
        raise ApiError(404, "not_found", "rule not found")
    merged = rules_repo.rule_to_seed(r).model_dump()
    merged.update({k: v for k, v in body.model_dump(exclude_unset=True).items()})
    seed = RuleSeed.model_validate(merged)  # re-validates N<=M, ranges etc. (FR-DET-008)
    r = rules_repo.upsert_rule(s, seed)
    audit(s, u.username, "rule_update", f"rule:{r.id}", body.model_dump(exclude_unset=True, mode="json"))
    s.commit()
    c.send_worker({"type": "reload_rules"})
    return rule_out(r, _target_fps(c))
