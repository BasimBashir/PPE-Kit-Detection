"""Identity descriptor + matching for crossings (ADR-016, revised).

Primary signal is an SFace face embedding, matched against the person's enrolled reference faces with a threshold
that loosens as the probe face turns away from the camera (frontal > quarter > profile) and a margin over the runner-up
person. A face is not always visible (back turned, overhead mount, too far), so a crossing also carries the old HSV
torso histogram, matched only when no face was found. Neither carries it alone - the event service additionally
carries a person across the tracks of one pass (track continuity) and unmatched crossings stay in the Unknown bucket.

Gallery entries are dicts `{kind: "face"|"torso", vec: [...], angle: str, crossing_id: str|None}`; galleries written
before the face upgrade are bare float lists and are read as torso entries.
"""

from __future__ import annotations

import uuid

import cv2
import numpy as np

from app.attendance.faces import FRONTAL, PROFILE, QUARTER
from app.ppe.types import Box

FACE, TORSO = "face", "torso"

# torso band as fractions of the person box: y0, y1, x0, x1 (top 15-60 %, sides trimmed)
TORSO_BAND = (0.15, 0.60, 0.20, 0.80)
BINS = (8, 8, 4)  # H, S, V -> 256 floats


def describe(image: np.ndarray, box: Box) -> list[float]:
    """L2-normalised HSV histogram of the torso crop; [] when the crop is empty."""
    h, w = image.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    x0, x1 = int(box[0] + bw * TORSO_BAND[2]), int(box[0] + bw * TORSO_BAND[3])
    y0, y1 = int(box[1] + bh * TORSO_BAND[0]), int(box[1] + bh * TORSO_BAND[1])
    crop = image[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if crop.size == 0:
        return []
    hist = cv2.calcHist([cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)], [0, 1, 2], None, BINS,
                        [0, 180, 0, 256, 0, 256]).ravel()
    norm = float(np.linalg.norm(hist))
    return (hist / norm).tolist() if norm else []


def entry(kind: str, vec: list[float], angle: str = "", crossing_id: str | None = None) -> dict:
    return {"kind": kind, "vec": vec, "angle": angle, "crossing_id": crossing_id}


def entries(gallery: list) -> list[dict]:
    """Gallery normalisation: bare vectors (pre-face galleries) become torso entries."""
    return [g if isinstance(g, dict) else entry(TORSO, g) for g in (gallery or [])]


def _score(vec: list[float], gallery: list[dict], kind: str) -> list[tuple[uuid.UUID, float]]:
    d = np.asarray(vec, dtype=np.float32)
    return [(pid, float(d @ np.asarray(g["vec"], dtype=np.float32)))
            for pid, g in gallery if g["kind"] == kind and g.get("vec")]


def _face_threshold(probe_angle: str, cfg) -> float:
    return {FRONTAL: cfg.face_frontal, QUARTER: cfg.face_quarter, PROFILE: cfg.face_profile}.get(
        probe_angle, cfg.face_quarter)


def match_face(face: list[float], angle: str, gallery: list[tuple[uuid.UUID, dict]], cfg
               ) -> tuple[uuid.UUID, float] | None:
    """Best person above the angle's threshold, and clear of the runner-up person by `cfg.face_margin`."""
    if not face:
        return None
    best: dict[uuid.UUID, float] = {}
    for pid, score in _score(face, gallery, FACE):
        best[pid] = max(best.get(pid, -1.0), score)
    if not best:
        return None
    ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)
    pid, score = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else -1.0
    if score >= _face_threshold(angle, cfg) and score - runner_up >= cfg.face_margin:
        return pid, score
    return None


def match_torso(descriptor: list[float], gallery: list[tuple[uuid.UUID, dict]],
                threshold: float) -> tuple[uuid.UUID, float] | None:
    if not descriptor:
        return None
    best = max(((pid, score) for pid, score in _score(descriptor, gallery, TORSO)), key=lambda kv: kv[1],
               default=None)
    return best if best and best[1] >= threshold else None


def match(descriptor: list[float], face: list[float], angle: str,
          gallery: list[tuple[uuid.UUID, dict]], cfg) -> tuple[uuid.UUID, float, str] | None:
    """Face first (blind on its own when turned away -> torso), each only within the direction-valid gallery."""
    hit = match_face(face, angle, gallery, cfg)
    if hit:
        return (*hit, FACE)
    hit = match_torso(descriptor, gallery, cfg.match_threshold)
    return (*hit, TORSO) if hit else None
