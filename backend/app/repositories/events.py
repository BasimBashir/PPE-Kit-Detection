"""Event queries and lifecycle (FR-ALT-006, FR-UI-004/005). OPEN -> ACKNOWLEDGED -> RESOLVED."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.models import Event, EventAction, utcnow


class StateConflict(Exception):
    pass


@dataclass
class EventFilter:
    camera_id: str | None = None
    status: str | None = None
    rule_id: str | None = None
    feedback: str | None = None  # TRUE_POSITIVE | FALSE_POSITIVE | NONE
    since: datetime | None = None
    until: datetime | None = None


def _where(q, f: EventFilter):
    if f.camera_id:
        q = q.where(Event.camera_id == f.camera_id)
    if f.status:
        q = q.where(Event.status == f.status)
    if f.rule_id:
        q = q.where(Event.rule_id == f.rule_id)
    if f.feedback == "NONE":
        q = q.where(Event.feedback.is_(None))
    elif f.feedback:
        q = q.where(Event.feedback == f.feedback)
    if f.since:
        q = q.where(Event.confirmed_at >= f.since)
    if f.until:
        q = q.where(Event.confirmed_at < f.until)
    return q


def list_events(s: Session, f: EventFilter, limit: int = 50, offset: int = 0) -> tuple[list[Event], int]:
    total = s.scalar(_where(select(func.count()).select_from(Event), f)) or 0
    rows = s.scalars(_where(select(Event), f).order_by(Event.confirmed_at.desc()).limit(limit).offset(offset))
    return list(rows), total


def get_event(s: Session, event_id: uuid.UUID) -> Event | None:
    return s.get(Event, event_id)


def actions_for(s: Session, event_id: uuid.UUID) -> list[EventAction]:
    return list(s.scalars(select(EventAction).where(EventAction.event_id == event_id).order_by(EventAction.id)))


def _act(s: Session, ev: Event, actor: str, action: str, note: str | None) -> None:
    s.add(EventAction(event_id=ev.id, actor_id=actor, action=action, note=note))


def acknowledge(s: Session, ev: Event, actor: str, note: str | None = None) -> Event:
    if ev.status != "OPEN":
        raise StateConflict(f"event is {ev.status}, only OPEN events can be acknowledged")
    ev.status, ev.acknowledged_by, ev.acknowledged_at = "ACKNOWLEDGED", actor, utcnow()
    _act(s, ev, actor, "ACKNOWLEDGE", note)
    s.flush()
    return ev


def resolve(s: Session, ev: Event, actor: str, note: str | None = None) -> Event:
    if ev.status == "RESOLVED":
        raise StateConflict("event is already RESOLVED")
    ev.status, ev.resolved_by, ev.resolved_at = "RESOLVED", actor, utcnow()
    _act(s, ev, actor, "RESOLVE", note)
    s.flush()
    return ev


def set_feedback(s: Session, ev: Event, actor: str, label: str, note: str | None = None) -> Event:
    if label not in ("TRUE_POSITIVE", "FALSE_POSITIVE"):
        raise ValueError("label must be TRUE_POSITIVE or FALSE_POSITIVE")
    ev.feedback, ev.feedback_note = label, note
    _act(s, ev, actor, "FEEDBACK", f"{label}: {note}" if note else label)
    s.flush()
    return ev


def set_dashboard_rendered(s: Session, event_id: uuid.UUID, ts: datetime) -> bool:
    """First write wins; returns True when this call recorded the timestamp."""
    res = s.execute(update(Event).where(Event.id == event_id, Event.dashboard_rendered_at.is_(None))
                    .values(dashboard_rendered_at=ts))
    return bool(res.rowcount)


def open_events(s: Session, limit: int = 100) -> list[Event]:
    return list(s.scalars(select(Event).where(Event.status != "RESOLVED")
                          .order_by(Event.confirmed_at.desc()).limit(limit)))


def count_open_by_camera(s: Session) -> dict[str, int]:
    rows = s.execute(select(Event.camera_id, func.count()).where(Event.status != "RESOLVED")
                     .group_by(Event.camera_id))
    return {c: n for c, n in rows}


def count_since(s: Session, since: datetime) -> int:
    return s.scalar(select(func.count()).select_from(Event).where(Event.confirmed_at >= since)) or 0
