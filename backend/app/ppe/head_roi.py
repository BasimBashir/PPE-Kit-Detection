"""Head ROI estimation (FR-DET-003 / plan 4.7 step 2)."""

from __future__ import annotations

import numpy as np

from app.config.settings import AssociationConfig
from app.inference.types import (
    KP_LEAR,
    KP_LEYE,
    KP_LSHOULDER,
    KP_NOSE,
    KP_REAR,
    KP_REYE,
    KP_RSHOULDER,
)
from app.ppe.geometry import clip
from app.ppe.types import Box, TrackedPerson

HEAD_KP = (KP_NOSE, KP_LEYE, KP_REYE, KP_LEAR, KP_REAR)


def _keypoint(person: TrackedPerson, i: int) -> tuple[float, float, float]:
    """(x, y, conf) of keypoint i; (-1, -1, 0) when the pose model gave none."""
    kp = person.keypoints
    if kp is None:
        return -1.0, -1.0, 0.0
    arr = np.asarray(kp)
    if arr.ndim != 2 or i >= arr.shape[0]:
        return -1.0, -1.0, 0.0
    return float(arr[i][0]), float(arr[i][1]), float(arr[i][2])


def _shoulder_width(person: TrackedPerson, min_conf: float) -> float:
    lx, _, lc = _keypoint(person, KP_LSHOULDER)
    rx, _, rc = _keypoint(person, KP_RSHOULDER)
    if lc >= min_conf and rc >= min_conf and lx >= 0 and rx >= 0:
        return abs(rx - lx)
    return 0.0


def head_roi(person: TrackedPerson, cfg: AssociationConfig, frame_w: float, frame_h: float) -> tuple[Box, str]:
    """Return (roi, source). Keypoints mode with geometry fallback; always clipped to the frame."""
    box = person.box
    if cfg.head_roi == "keypoints":
        pts = [(x, y) for x, y, c in (_keypoint(person, i) for i in HEAD_KP) if c >= cfg.keypoint_min_conf and x >= 0]
        if len(pts) >= max(2, cfg.min_head_keypoints):
            xs, ys = zip(*pts, strict=False)
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            s = max(
                max(xs) - min(xs),
                0.45 * _shoulder_width(person, cfg.keypoint_min_conf),
                0.18 * (box[3] - box[1]),
            )
            if s > 0:
                w, h, up = 1.4 * s, 1.6 * s, 0.35 * s
                return clip((cx - w / 2, cy - h / 2 - up, cx + w / 2, cy + h / 2 - up), frame_w, frame_h), "keypoints"

    h = box[3] - box[1]
    w = box[2] - box[0]
    up = cfg.roi_up_frac * h
    roi = (
        box[0] + cfg.roi_side_margin_frac * w,
        box[1] - up,
        box[2] - cfg.roi_side_margin_frac * w,
        box[1] + cfg.roi_top_frac * h,
    )
    return clip(roi, frame_w, frame_h), "geometry"
