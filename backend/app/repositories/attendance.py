"""Persons, crossings and derived presence intervals (Phase 5, ADR-016)."""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime

from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from app.db.base import SessionFactory, session_scope
from app.models import Crossing, Person, PresenceInterval


def crossing_key(camera_id: str, track_id: int, zone_id: str, at: float) -> str:
    return hashlib.sha256(f"{camera_id}|{track_id}|{zone_id}|{at:.3f}".encode()).hexdigest()


def list_persons(s: Session) -> list[Person]:
    return list(s.scalars(select(Person).order_by(Person.name)))


def get_person(s: Session, person_id: uuid.UUID) -> Person | None:
    return s.get(Person, person_id)


def inside_ids(s: Session) -> set[uuid.UUID]:
    return set(s.scalars(select(PresenceInterval.person_id).where(PresenceInterval.ended_at.is_(None))))


def crossing_counts(s: Session) -> dict[uuid.UUID, int]:
    return dict(s.execute(select(Crossing.person_id, func.count(Crossing.id)).group_by(Crossing.person_id)).all())


def gallery(s: Session, inside: set[uuid.UUID], direction: str) -> list[tuple[uuid.UUID, list[float]]]:
    """(person, descriptor) pairs from people whose current state fits the crossing: an EXIT can only belong to
    someone currently inside, an ENTER only to someone currently outside."""
    out = []
    for p in s.scalars(select(Person)):
        if (p.id in inside) != (direction == "EXIT"):
            continue
        out.extend((p.id, vec) for vec in (p.gallery or []))
    return out


def insert_crossing(sf: SessionFactory, payload: dict) -> bool:
    """Idempotent insert; True when the row was new. The caller rebuilds intervals for `person_id` if set."""
    values = {k: v for k, v in payload.items() if k != "at"}
    values["occurred_at"] = datetime.fromtimestamp(payload["at"], UTC)
    stmt = (pg_insert(Crossing)
            .values(idempotency_key=crossing_key(payload["camera_id"], payload["track_id"], payload["zone_id"],
                                                 payload["at"]), **values)
            .on_conflict_do_nothing(index_elements=["idempotency_key"])
            .returning(Crossing.id))
    with session_scope(sf) as s:
        return s.execute(stmt).first() is not None


def list_crossings(s: Session, person_id: uuid.UUID | None, unknown: bool, since: datetime | None,
                   until: datetime | None, limit: int, offset: int) -> tuple[list[Crossing], int]:
    q = select(Crossing)
    if unknown:
        q = q.where(Crossing.person_id.is_(None))
    elif person_id:
        q = q.where(Crossing.person_id == person_id)
    if since:
        q = q.where(Crossing.occurred_at >= since)
    if until:
        q = q.where(Crossing.occurred_at <= until)
    total = s.scalar(select(func.count()).select_from(q.subquery())) or 0
    rows = list(s.scalars(q.order_by(Crossing.occurred_at.desc()).limit(limit).offset(offset)))
    return rows, total


def get_crossing(s: Session, crossing_id: uuid.UUID) -> Crossing | None:
    return s.get(Crossing, crossing_id)


def rebuild_intervals(s: Session, person_id: uuid.UUID) -> None:
    """Presence intervals are derived: replay the person's crossings in order (EXIT opens, ENTER closes)."""
    s.execute(delete(PresenceInterval).where(PresenceInterval.person_id == person_id))
    open_at: datetime | None = None
    rows = s.execute(select(Crossing.direction, Crossing.occurred_at)
                     .where(Crossing.person_id == person_id).order_by(Crossing.occurred_at)).all()
    for direction, at in rows:
        if direction == "EXIT" and open_at is None:
            open_at = at
        elif direction == "ENTER" and open_at is not None:
            s.add(PresenceInterval(person_id=person_id, started_at=open_at, ended_at=at))
            open_at = None
    if open_at is not None:
        s.add(PresenceInterval(person_id=person_id, started_at=open_at))
    s.flush()


def intervals(s: Session, person_id: uuid.UUID, lo: datetime, hi: datetime) -> list[tuple[datetime, datetime | None]]:
    rows = s.execute(select(PresenceInterval.started_at, PresenceInterval.ended_at)
                     .where(PresenceInterval.person_id == person_id, PresenceInterval.started_at < hi,
                            (PresenceInterval.ended_at.is_(None)) | (PresenceInterval.ended_at > lo))
                     .order_by(PresenceInterval.started_at)).all()
    return [(a, b) for a, b in rows]
