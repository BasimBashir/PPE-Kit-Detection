"""Zones: monitor (rule scope), ignore (drop detections) and privacy (blur in previews/evidence)."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.ppe.geometry import point_in_polygon
from app.ppe.types import Box, TrackedPerson


@dataclass(slots=True)
class ZoneRuntime:
    id: str
    camera_id: str
    name: str
    kind: str  # monitor | ignore | privacy
    polygon: list[tuple[float, float]]  # normalised 0..1
    enabled: bool = True


def _foot(box: Box, w: float, h: float) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2 / w, box[3] / h)


def zone_of(person_box: Box, zones: list[ZoneRuntime], w: float, h: float) -> str | None:
    """First enabled monitor zone containing the person's foot point; None when no monitor zone matches."""
    p = _foot(person_box, w, h)
    for z in zones:
        if z.enabled and z.kind == "monitor" and point_in_polygon(p, z.polygon):
            return z.id
    return None


def has_monitor_zones(zones: list[ZoneRuntime]) -> bool:
    return any(z.enabled and z.kind == "monitor" for z in zones)


def filter_ignored(persons: list[TrackedPerson], zones: list[ZoneRuntime], w: float, h: float) -> list[TrackedPerson]:
    ignore = [z for z in zones if z.enabled and z.kind == "ignore"]
    if not ignore:
        return persons
    return [p for p in persons if not any(point_in_polygon(_foot(p.box, w, h), z.polygon) for z in ignore)]


def apply_privacy_masks(image: np.ndarray, zones: list[ZoneRuntime]) -> np.ndarray:
    """Copy of `image` with privacy polygons pixelated beyond recognition."""
    priv = [z for z in zones if z.enabled and z.kind == "privacy"]
    out = image.copy()
    if not priv:
        return out
    h, w = image.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for z in priv:
        pts = np.array([[x * w, y * h] for x, y in z.polygon], np.int32)
        cv2.fillPoly(mask, [pts], 255)
    small = cv2.resize(image, (max(1, w // 32), max(1, h // 32)), interpolation=cv2.INTER_LINEAR)
    pix = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    out[mask > 0] = pix[mask > 0]
    return out
