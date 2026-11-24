"""Phase 5 unit logic: door-line geometry, appearance descriptor, per-day outside report, zone validation."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from app.attendance.crossing import MAX_GAP_S, CrossingState, side
from app.attendance.identity import describe, match
from app.attendance.report import outside_by_day
from app.config.settings import AttendanceConfig
from app.ppe.zones import ZoneRuntime
from app.schemas.requests import ZoneIn
from app.streaming.frame import Frame

W = H = 64
DOOR = ZoneRuntime(id="door1", camera_id="cam1", name="door", kind="door", polygon=[(0.5, 0.2), (0.5, 0.8)])


def person(track_id: int, x: float) -> SimpleNamespace:
    """Person centred on normalised x, standing on the line y=0.5 (box in pixels)."""
    cx = x * W
    return SimpleNamespace(track_id=track_id, box=(cx - 5, 20, cx + 5, 40))


def frame(mono: float) -> Frame:
    return Frame("cam1", 0, np.zeros((H, W, 3), np.uint8), None, 1_700_000_000.0, mono)


def step(state: CrossingState, mono: float, *people: SimpleNamespace) -> list:
    return state.update([DOOR], frame(mono), people, AttendanceConfig())


def test_side_sign_is_positive_on_the_walking_side():
    # door drawn top -> bottom: the inside half-plane is to the walker's left of that direction
    assert side((0.5, 0.2), (0.5, 0.8), (0.3, 0.5)) > 0
    assert side((0.5, 0.2), (0.5, 0.8), (0.7, 0.5)) < 0


def test_enter_and_exit_need_a_firm_previous_side():
    s = CrossingState()
    assert step(s, 0.0, person(1, 0.7)) == []  # learns OUTSIDE
    out = step(s, 0.1, person(1, 0.3))
    assert [(x.track_id, x.direction) for x in out] == [(1, "ENTER")]
    out = step(s, 0.2, person(1, 0.7))
    assert [(x.track_id, x.direction) for x in out] == [(1, "EXIT")]


def test_deadband_keeps_the_side_across_the_line():
    s = CrossingState()
    step(s, 0.0, person(1, 0.7))
    assert step(s, 0.1, person(1, 0.5)) == []  # on the line: side not updated
    assert s._sides[("door1", 1)][0] < 0
    assert step(s, 0.2, person(1, 0.3))[0].direction == "ENTER"  # still crossing from OUTSIDE


def test_gap_longer_than_max_relearns_without_a_crossing():
    s = CrossingState()
    step(s, 0.0, person(1, 0.7))
    assert step(s, MAX_GAP_S + 1, person(1, 0.3)) == []  # track id may have been reused
    assert step(s, MAX_GAP_S + 1.1, person(1, 0.7))[0].direction == "EXIT"


def test_crossing_carries_crop_and_descriptor():
    img = np.zeros((H, W, 3), np.uint8)
    img[20:40, 30:45] = (0, 0, 200)  # red torso
    s = CrossingState()
    s.update([DOOR], Frame("cam1", 0, img, None, 1.0, 0.0), [person(1, 0.7)], AttendanceConfig())
    out = s.update([DOOR], Frame("cam1", 0, img, None, 1.0, 0.1), [person(1, 0.3)], AttendanceConfig())
    assert out[0].image is img and len(out[0].descriptor) == 256
    assert abs(float(np.linalg.norm(out[0].descriptor)) - 1.0) < 1e-6


def test_describe_is_empty_for_a_degenerate_crop_and_match_handles_it():
    assert describe(np.zeros((H, W, 3), np.uint8), (10, 10, 10, 10)) == []
    assert match([], [(uuid.uuid4(), [1.0])], 0.0) is None


def test_match_picks_the_closest_above_the_threshold():
    red = np.zeros((H, W, 3), np.uint8)
    red[20:40, 25:35] = (0, 0, 220)
    blue = np.zeros((H, W, 3), np.uint8)
    blue[20:40, 25:35] = (220, 0, 0)
    box = (25, 20, 35, 40)
    a, b = uuid.uuid4(), uuid.uuid4()
    gallery = [(a, describe(red, box)), (b, describe(blue, box))]
    assert match(describe(red, box), gallery, 0.55)[0] == a
    assert match(describe(red, box), gallery, 0.999)[0] == a  # identical histogram scores 1.0
    assert match(describe(red, box), [(b, describe(blue, box))], 0.99) is None


def test_outside_minutes_are_clipped_to_local_days_and_the_budget():
    rows = outside_by_day([(datetime(2026, 1, 1, 23, 0, tzinfo=UTC), datetime(2026, 1, 2, 1, 0, tzinfo=UTC))],
                          date(2026, 1, 1), date(2026, 1, 2), "UTC", allowed_minutes=30)
    assert [r["outside_minutes"] for r in rows] == [60.0, 60.0]
    assert [r["over_minutes"] for r in rows] == [30.0, 30.0]


def test_open_interval_counts_up_to_now():
    now = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    rows = outside_by_day([(datetime(2026, 1, 1, 11, 0, tzinfo=UTC), None)], date(2026, 1, 1), date(2026, 1, 1),
                          "UTC", allowed_minutes=0, now=now)
    assert rows[0]["outside_minutes"] == 60.0


def test_zone_in_accepts_only_two_points_for_a_door():
    ok = ZoneIn(id="d1", name="door", kind="door", polygon=[(0.1, 0.5), (0.2, 0.5)])
    assert len(ok.polygon) == 2
    with pytest.raises(ValidationError):
        ZoneIn(id="d2", name="door", kind="door", polygon=[(0, 0), (1, 0), (1, 1)])
    with pytest.raises(ValidationError):
        ZoneIn(id="m1", name="monitor", kind="monitor", polygon=[(0, 0), (1, 0)])
