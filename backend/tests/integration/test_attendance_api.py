"""Attendance API (Phase 5, ADR-016): door zone, persons, assign/enrol, per-day outside report vs budget."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select

from app.db.base import session_scope
from app.models import Crossing
from app.repositories import attendance as att_repo

pytestmark = pytest.mark.integration


def login(c, user="admin", pw="admin-password-1"):
    r = c.post("/api/v1/auth/login", json={"username": user, "password": pw})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def at(day: int, hour: int) -> float:
    return datetime(2026, 1, day, hour, tzinfo=UTC).timestamp()


def crossing(sf, t: float, direction: str, *, descriptor: list[float] | None = None) -> tuple[str, str]:
    ok = att_repo.insert_crossing(sf, {"camera_id": "cam01", "zone_id": "door1", "track_id": 7, "direction": direction,
                                       "at": t, "descriptor": descriptor or [1.0, 0.0], "image_path": None,
                                       "image_error": "test"})
    assert ok
    with session_scope(sf) as s:
        return str(s.scalar(select(Crossing.id).where(
            Crossing.idempotency_key == att_repo.crossing_key("cam01", 7, "door1", t))))


def test_door_zone_person_assign_enrol_and_report(client, sf):
    admin = login(client)
    door = {"id": "door1", "name": "Main door", "kind": "door", "polygon": [[0.5, 0.2], [0.5, 0.8]], "enabled": True}
    assert client.post("/api/v1/cameras/cam01/zones", headers=admin, json=door).status_code == 201
    bad = {**door, "id": "door2", "polygon": [[0, 0], [1, 0], [1, 1]]}
    assert client.post("/api/v1/cameras/cam01/zones", headers=admin, json=bad).status_code == 422

    r = client.post("/api/v1/attendance/persons", headers=admin, json={"name": "Alice", "allowed_minutes_per_day": 30})
    assert r.status_code == 201 and r.json()["gallery_size"] == 0
    alice = r.json()["id"]
    assert client.post("/api/v1/attendance/persons", headers=admin,
                       json={"name": "Alice", "allowed_minutes_per_day": 0}).status_code == 409

    exit_key = crossing(sf, at(5, 8), "EXIT")
    enter_key = crossing(sf, at(5, 10), "ENTER")
    unknown_key = crossing(sf, at(5, 12), "EXIT", descriptor=[0.0, 1.0])
    page = client.get("/api/v1/attendance/crossings", headers=admin).json()
    assert page["total"] == 3 and all(x["person_id"] is None for x in page["items"])
    assert client.get("/api/v1/attendance/crossings?unknown=true", headers=admin).json()["total"] == 3

    # assign the two crossings that make an interval; the trailing EXIT stays open and counts against the budget
    for key in (exit_key, enter_key):
        r = client.patch(f"/api/v1/attendance/crossings/{key}", headers=admin, json={"person_id": alice})
        assert r.status_code == 200, r.text
        assert r.json()["person_name"] == "Alice"
    assert client.get(f"/api/v1/attendance/crossings?person_id={alice}", headers=admin).json()["total"] == 2

    # enrolling stores the crop descriptor and assigns that crossing too
    r = client.post(f"/api/v1/attendance/persons/{alice}/gallery", headers=admin, json={"crossing_id": unknown_key})
    assert r.json()["gallery_size"] == 1 and r.json()["inside"] is True
    assert client.get("/api/v1/attendance/crossings?unknown=true", headers=admin).json()["total"] == 0

    rep = client.get(f"/api/v1/attendance/report?person_id={alice}&start=2026-01-05&end=2026-01-05",
                     headers=admin).json()
    assert rep["days"][0]["allowed_minutes"] == 30
    # 08:00 EXIT -> 10:00 ENTER = 120 min, plus 12:00 EXIT open to the end of the day
    assert rep["days"][0]["outside_minutes"] == 840.0 and rep["days"][0]["over_minutes"] == 810.0
    assert rep["total"]["over_minutes"] == 810.0
    assert client.get(f"/api/v1/attendance/report?person_id={alice}&start=2026-01-06&end=2026-01-05",
                      headers=admin).status_code == 422

    # intervals cascade; crossings fall back to Unknown with their descriptor kept
    assert client.delete(f"/api/v1/attendance/persons/{alice}", headers=admin).status_code == 204
    assert client.get("/api/v1/attendance/crossings?unknown=true", headers=admin).json()["total"] == 3


def test_attendance_roles_and_errors(client):
    admin = login(client)
    assert client.post("/api/v1/users", headers=admin,
                       json={"username": "viewer2", "password": "viewer-pass-2", "role": "viewer"}).status_code == 201
    viewer = login(client, "viewer2", "viewer-pass-2")
    assert client.get("/api/v1/attendance/persons", headers=viewer).status_code == 200
    assert client.post("/api/v1/attendance/persons", headers=viewer, json={"name": "Bob"}).status_code == 403
    assert client.get("/api/v1/attendance/persons?limit=1", headers=viewer).status_code == 200
    assert client.patch("/api/v1/attendance/crossings/00000000-0000-0000-0000-000000000000", headers=admin,
                        json={"person_id": None}).status_code == 404
    assert client.get("/api/v1/attendance/report?person_id=00000000-0000-0000-0000-000000000000"
                      "&start=2026-01-01&end=2026-01-02", headers=viewer).status_code == 404
