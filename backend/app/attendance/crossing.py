"""Door-line crossing detection (ADR-016).

A door zone is two normalised points. The operator draws it in the direction people walk when entering, so the
half-plane with a positive signed distance is INSIDE: s<0 -> s>0 is an ENTER, s>0 -> s<0 an EXIT. A deadband
around the line stops foot-point jitter from flapping the side.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from app.attendance.identity import describe
from app.config.settings import AttendanceConfig
from app.ppe.types import Box
from app.ppe.zones import ZoneRuntime
from app.streaming.frame import Frame

# shortcut: track ids are reused after the tracker expires them; a gap longer than this re-learns the side
#            without emitting a crossing. Upgrade to a monotonic track id if the tracker ever changes.
MAX_GAP_S = 5.0


@dataclass(slots=True)
class Crossing:
    camera_id: str
    zone_id: str
    track_id: int
    direction: str  # ENTER | EXIT
    at: float  # wall clock of the processed frame
    person_box: Box
    image: np.ndarray  # privacy-masked frame (BGR) for the evidence crop
    descriptor: list[float]


def side(p0: tuple[float, float], p1: tuple[float, float], point: tuple[float, float]) -> float:
    """Signed distance of `point` from the directed line p0->p1, scaled to be unit-free (negative = outside)."""
    dx, dy = p1[0] - p0[0], p1[1] - p0[1]
    n = float(np.hypot(dx, dy)) or 1e-9
    return (dx * (point[1] - p0[1]) - dy * (point[0] - p0[0])) / n


def _foot(box: Box, w: float, h: float) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2 / w, box[3] / h)


@dataclass
class CrossingState:
    """Per-camera memory of which side each track was last firmly on. Cheap: two floats per active track."""

    _sides: dict[tuple[str, int], tuple[float, float]] = field(default_factory=dict)  # (zone, track) -> (side, mono)

    def update(self, doors: Sequence[ZoneRuntime], frame: Frame, persons: Sequence,
               cfg: AttendanceConfig) -> list[Crossing]:
        if not doors:
            self._sides.clear()
            return []
        out: list[Crossing] = []
        seen: set[tuple[str, int]] = set()
        for person in persons:
            foot = _foot(person.box, frame.width, frame.height)
            for door in doors:
                key = (door.id, person.track_id)
                seen.add(key)
                if not (door.enabled and len(door.polygon) == 2):
                    continue
                p0, p1 = door.polygon
                s = side(p0, p1, foot)
                if abs(s) < cfg.line_deadband:
                    continue
                prev = self._sides.get(key)
                self._sides[key] = (s, frame.captured_mono)
                if prev is None or frame.captured_mono - prev[1] > MAX_GAP_S:
                    continue
                if prev[0] < 0 <= s:
                    direction = "ENTER"
                elif prev[0] > 0 >= s:
                    direction = "EXIT"
                else:
                    continue
                out.append(Crossing(frame.camera_id, door.id, person.track_id, direction, frame.captured_at,
                                    person.box, frame.image, describe(frame.image, person.box)))
        stale = [k for k in self._sides if k not in seen and frame.captured_mono - self._sides[k][1] > MAX_GAP_S * 4]
        for k in stale:
            del self._sides[k]
        return out
