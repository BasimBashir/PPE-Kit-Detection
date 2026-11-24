"""Phase 5 unit logic: door-line geometry, appearance descriptor, per-day outside report, zone validation."""

from __future__ import annotations

import uuid
from datetime import UTC, date, datetime
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from app.attendance.crossing import MAX_GAP_S, CrossingState, side
from app.attendance.faces import FRONTAL, PROFILE, QUARTER, angle_of
from app.attendance.identity import FACE, TORSO, describe, entries, entry, match
from app.attendance.report import outside_by_day
from app.config.settings import AttendanceConfig
from app.ppe.zones import ZoneRuntime
from app.repositories.attendance import enrol_entries
from app.schemas.requests import EnrolIn, ZoneIn
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
    assert match([], [], "", [], AttendanceConfig()) is None


def test_legacy_bare_vector_galleries_read_as_torso():
    assert entries([[1.0, 0.0]])[0]["kind"] == TORSO
    assert entries([]) == []


def test_angle_buckets_come_from_the_landmarks():
    assert angle_of(0.0, 0.0) == FRONTAL
    assert angle_of(0.0, 0.6) == QUARTER  # head down under a roof mount is not a frontal probe
    assert angle_of(0.4, 0.0) == QUARTER
    assert angle_of(0.8, 0.0) == PROFILE


def test_face_match_loosens_its_threshold_as_the_face_turns():
    probe = [0.4, float(np.sqrt(1 - 0.16))]  # cosine 0.40 to the reference below
    pid = uuid.uuid4()
    gallery = [(pid, entry(FACE, [1.0, 0.0], PROFILE))]
    assert match([], probe, FRONTAL, gallery, AttendanceConfig()) is None  # 0.40 < 0.45
    hit = match([], probe, QUARTER, gallery, AttendanceConfig())
    assert hit and hit[0] == pid and hit[2] == FACE and hit[1] == pytest.approx(0.4)


def test_face_match_needs_a_margin_over_the_runner_up():
    a, b = uuid.uuid4(), uuid.uuid4()
    gallery = [(a, entry(FACE, [1.0, 0.0], "")), (b, entry(FACE, [1.0, 0.01], ""))]
    assert match([], [1.0, 0.0], FRONTAL, gallery, AttendanceConfig()) is None  # near-identical twins
    assert match([], [1.0, 0.0], FRONTAL, gallery, AttendanceConfig(face_margin=0))[0] == a


def test_face_is_preferred_and_torso_covers_only_a_missing_face():
    red = np.zeros((H, W, 3), np.uint8)
    red[20:40, 25:35] = (0, 0, 220)
    a, b = uuid.uuid4(), uuid.uuid4()
    gallery = [(a, entry(FACE, [1.0, 0.0], "")), (b, entry(TORSO, describe(red, (25, 20, 35, 40))))]
    hit = match(describe(red, (25, 20, 35, 40)), [1.0, 0.0], FRONTAL, gallery, AttendanceConfig())
    assert hit and hit[0] == a and hit[2] == FACE
    hit = match(describe(red, (25, 20, 35, 40)), [], "", gallery, AttendanceConfig())
    assert hit and hit[0] == b and hit[2] == TORSO
    blue = np.zeros((H, W, 3), np.uint8)
    blue[20:40, 25:35] = (220, 0, 0)
    assert match(describe(blue, (25, 20, 35, 40)), [], "", gallery, AttendanceConfig(match_threshold=0.99)) is None


def test_enrol_entries_dedupes_caps_and_keeps_the_angle_spread():
    p = SimpleNamespace(gallery=[])
    for _ in range(3):
        enrol_entries(p, [entry(FACE, [1.0], FRONTAL)], 4)
    assert len(p.gallery) == 1  # the same reference is not stored twice
    enrol_entries(p, [entry(FACE, [0.0, 1.0], PROFILE)], 4)
    for i in range(5):
        enrol_entries(p, [entry(FACE, [float(i), 2.0], FRONTAL)], 4)
    assert len(p.gallery) == 4
    assert sum(e["angle"] == PROFILE for e in p.gallery) == 1  # the over-represented frontal group was trimmed


def test_enrol_in_needs_exactly_one_source():
    EnrolIn(crossing_id=uuid.uuid4())
    EnrolIn(camera_id="cam1", box=(1, 2, 3, 4))
    bad = ({}, {"crossing_id": uuid.uuid4(), "camera_id": "cam1"},
           {"crossing_id": uuid.uuid4(), "box": (1, 2, 3, 4)})
    for body in bad:
        with pytest.raises(ValidationError):
            EnrolIn(**body)


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
