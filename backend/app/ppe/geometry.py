"""Bounding-box and polygon geometry helpers (pure, unit-tested)."""

from __future__ import annotations

import math

from app.ppe.types import Box


def valid_box(b: Box) -> bool:
    x1, y1, x2, y2 = b
    return all(math.isfinite(v) for v in b) and x2 > x1 and y2 > y1


def area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def intersection(a: Box, b: Box) -> float:
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return w * h if w > 0 and h > 0 else 0.0


def iou(a: Box, b: Box) -> float:
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def ioa(a: Box, b: Box) -> float:
    """Intersection over the area of `a` (share of a lying inside b)."""
    aa = area(a)
    return intersection(a, b) / aa if aa > 0 else 0.0


def center(b: Box) -> tuple[float, float]:
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def contains_point(b: Box, p: tuple[float, float]) -> bool:
    return b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]


def diagonal(b: Box) -> float:
    return math.hypot(b[2] - b[0], b[3] - b[1])


def clip(b: Box, width: float, height: float) -> Box:
    return (min(max(b[0], 0.0), width), min(max(b[1], 0.0), height),
            min(max(b[2], 0.0), width), min(max(b[3], 0.0), height))


def scale(b: Box, sx: float, sy: float) -> Box:
    return (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)


def point_in_polygon(p: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    """Ray casting; points on the boundary may go either way."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xin = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xin:
                inside = not inside
    return inside
