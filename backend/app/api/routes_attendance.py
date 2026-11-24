"""Attendance (Phase 5): enrolled persons, door crossings, outside-time against a daily budget (ADR-016).

Identity here is deliberately not an `event` field: names live on persons/crossings only. Unmatched crossings
stay in the Unknown bucket with their descriptor, so an operator can assign or enrol them afterwards.
"""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import APIRouter, Depends, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.api.deps import ApiError, AppContext, ctx, db, require
from app.attendance.report import outside_by_day
from app.models import Camera, Person, User
from app.repositories import attendance as att_repo
from app.repositories.misc import audit
from app.schemas.requests import CrossingAssign, EnrolIn, PersonIn, PersonUpdate
from app.schemas.serializers import crossing_out, iso, person_out

router = APIRouter(prefix="/api/v1/attendance", tags=["attendance"])

MAX_REPORT_DAYS = 92


def _person_or_404(s: Session, person_id: uuid.UUID) -> Person:
    p = att_repo.get_person(s, person_id)
    if p is None:
        raise ApiError(404, "not_found", "person not found")
    return p


def _crossing_or_404(s: Session, crossing_id: uuid.UUID):
    x = att_repo.get_crossing(s, crossing_id)
    if x is None:
        raise ApiError(404, "not_found", "crossing not found")
    return x


def _tz(c: AppContext) -> ZoneInfo:
    try:
        return ZoneInfo(c.cfg.attendance.site_tz)
    except (ZoneInfoNotFoundError, ValueError) as e:
        raise ApiError(422, "invalid_tz", f"unknown time zone {c.cfg.attendance.site_tz!r}") from e


@router.get("/persons")
def list_persons(s: Session = Depends(db), _u: User = Depends(require("viewer"))):
    inside, counts = att_repo.inside_ids(s), att_repo.crossing_counts(s)
    return [person_out(p, p.id in inside, counts.get(p.id, 0)) for p in att_repo.list_persons(s)]


@router.post("/persons", status_code=201)
def create_person(body: PersonIn, s: Session = Depends(db), u: User = Depends(require("operator"))):
    if any(p.name == body.name for p in att_repo.list_persons(s)):
        raise ApiError(409, "conflict", f"person {body.name!r} exists")
    p = Person(name=body.name, allowed_minutes_per_day=body.allowed_minutes_per_day)
    s.add(p)
    s.flush()
    audit(s, u.username, "person_create", f"person:{p.id}")
    s.commit()
    return person_out(p)


@router.patch("/persons/{person_id}")
def update_person(person_id: uuid.UUID, body: PersonUpdate, s: Session = Depends(db),
                  u: User = Depends(require("operator"))):
    p = _person_or_404(s, person_id)
    for k, v in body.model_dump(exclude_unset=True).items():
        setattr(p, k, v)
    audit(s, u.username, "person_update", f"person:{p.id}")
    s.commit()
    return person_out(p, p.id in att_repo.inside_ids(s), att_repo.crossing_counts(s).get(p.id, 0))


@router.delete("/persons/{person_id}", status_code=204)
def delete_person(person_id: uuid.UUID, s: Session = Depends(db), u: User = Depends(require("operator"))):
    p = _person_or_404(s, person_id)
    s.delete(p)  # intervals cascade; crossings fall back to Unknown (person_id SET NULL) with descriptor kept
    audit(s, u.username, "person_delete", f"person:{person_id}")
    s.commit()


@router.post("/persons/{person_id}/gallery")
def enrol(person_id: uuid.UUID, body: EnrolIn, c: AppContext = Depends(ctx), s: Session = Depends(db),
          u: User = Depends(require("operator"))):
    p, x = _person_or_404(s, person_id), _crossing_or_404(s, body.crossing_id)
    if not x.descriptor:
        raise ApiError(422, "no_descriptor", "that crossing has no appearance descriptor")
    if x.descriptor not in (p.gallery or []):
        p.gallery = [*(p.gallery or []), x.descriptor][-c.cfg.attendance.gallery_max:]
    if x.person_id != p.id:
        old = x.person_id
        x.person_id, x.match_score = p.id, None
        att_repo.rebuild_intervals(s, p.id)
        if old:
            att_repo.rebuild_intervals(s, old)
    audit(s, u.username, "person_enrol", f"person:{p.id}")
    s.commit()
    return person_out(p, p.id in att_repo.inside_ids(s), att_repo.crossing_counts(s).get(p.id, 0))


@router.get("/crossings")
def list_crossings(person_id: uuid.UUID | None = None, unknown: bool = False,
                   since: datetime | None = None, until: datetime | None = None,
                   limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0),
                   s: Session = Depends(db), _u: User = Depends(require("viewer"))):
    rows, total = att_repo.list_crossings(s, person_id, unknown, since, until, limit, offset)
    cams = {c.id: c.name for c in s.query(Camera).all()}
    names = {p.id: p.name for p in s.query(Person).all()}
    return {"items": [crossing_out(x, cams.get(x.camera_id), names.get(x.person_id)) for x in rows],
            "total": total, "limit": limit, "offset": offset}


@router.get("/crossings/{crossing_id}/image")
def crossing_image(crossing_id: uuid.UUID, c: AppContext = Depends(ctx), s: Session = Depends(db),
                   _u: User = Depends(require("viewer"))):
    x = _crossing_or_404(s, crossing_id)
    if not x.image_path:
        raise ApiError(404, "not_found", f"no crop ({x.image_error or 'not stored'})")
    try:
        path = c.evidence.open(x.image_path)
    except FileNotFoundError as err:
        raise ApiError(404, "not_found", "crop file missing (retention or disk issue)") from err
    return FileResponse(path, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=3600"})


@router.patch("/crossings/{crossing_id}")
def assign_crossing(crossing_id: uuid.UUID, body: CrossingAssign, s: Session = Depends(db),
                    u: User = Depends(require("operator"))):
    x = _crossing_or_404(s, crossing_id)
    if body.person_id is not None:
        _person_or_404(s, body.person_id)
    old = x.person_id
    x.person_id, x.match_score = body.person_id, None
    for pid in {p for p in (old, body.person_id) if p}:
        att_repo.rebuild_intervals(s, pid)  # intervals are derived, so a correction heals both people
    audit(s, u.username, "crossing_assign", f"crossing:{x.id}")
    s.commit()
    names = {p.id: p.name for p in s.query(Person).all()}
    cams = {c.id: c.name for c in s.query(Camera).all()}
    return crossing_out(x, cams.get(x.camera_id), names.get(x.person_id))


@router.get("/report")
def report(person_id: uuid.UUID, start: date, end: date, c: AppContext = Depends(ctx),
           s: Session = Depends(db), _u: User = Depends(require("viewer"))):
    if end < start:
        raise ApiError(422, "invalid_range", "end must not be before start")
    if (end - start).days > MAX_REPORT_DAYS:
        raise ApiError(422, "invalid_range", f"range is capped at {MAX_REPORT_DAYS} days")
    p = _person_or_404(s, person_id)
    tz = _tz(c)
    lo = datetime.combine(start, time.min, tz).astimezone(UTC)
    hi = datetime.combine(end + timedelta(days=1), time.min, tz).astimezone(UTC)
    days = outside_by_day(att_repo.intervals(s, p.id, lo, hi), start, end, str(tz), p.allowed_minutes_per_day)
    return {"person": person_out(p), "from": iso(lo), "to": iso(hi), "days": days,
            "total": {"outside_minutes": round(sum(d["outside_minutes"] for d in days), 1),
                      "over_minutes": round(sum(d["over_minutes"] for d in days), 1)}}
