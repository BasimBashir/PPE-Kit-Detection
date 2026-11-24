"""Appearance descriptor for crossings: HSV histogram of the upper torso (ADR-016).

Deliberately non-biometric and dependency-free: no faces, no ReID network. It separates people by clothing colour
and is weak on identical uniforms - which is why unmatched crossings stay Unknown and the operator can re-assign.
"""

from __future__ import annotations

import uuid

import cv2
import numpy as np

from app.ppe.types import Box

# torso band as fractions of the person box: y0, y1, x0, x1 (top 15-60 %, sides trimmed)
TORSO = (0.15, 0.60, 0.20, 0.80)
BINS = (8, 8, 4)  # H, S, V -> 256 floats


def describe(image: np.ndarray, box: Box) -> list[float]:
    """L2-normalised HSV histogram of the torso crop; [] when the crop is empty."""
    h, w = image.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x0, x1 = int(box[0] + bw * TORSO[2]), int(box[0] + bw * TORSO[3])
    y0, y1 = int(box[1] + bh * TORSO[0]), int(box[1] + bh * TORSO[1])
    crop = image[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if crop.size == 0:
        return []
    hist = cv2.calcHist([cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, BINS,
                        [0, 180, 0, 256, 0, 256]).ravel()
    norm = float(np.linalg.norm(hist))
    return (hist / norm).tolist() if norm else []


def match(descriptor: list[float], gallery: list[tuple[uuid.UUID, list[float]]],
          threshold: float) -> tuple[uuid.UUID, float] | None:
    """Best gallery entry above `threshold` (cosine on unit-length histograms), else None."""
    if not descriptor:
        return None
    d = np.asarray(descriptor, dtype=np.float32)
    best: tuple[uuid.UUID, float] | None = None
    for person_id, vec in gallery:
        if not vec:
            continue
        score = float(d @ np.asarray(vec, dtype=np.float32))
        if score >= threshold and (best is None or score > best[1]):
            best = (person_id, score)
    return best
