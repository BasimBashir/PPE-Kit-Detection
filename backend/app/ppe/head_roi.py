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


# ---------------------------------------------------------------- Phase 3: other body regions

KP_LELBOW, KP_RELBOW, KP_LWRIST, KP_RWRIST, KP_LHIP, KP_RHIP, KP_LANKLE, KP_RANKLE = 7, 8, 9, 10, 11, 12, 15, 16


def _pt(person: TrackedPerson, i: int, min_conf: float) -> tuple[float, float] | None:
    x, y, c = _keypoint(person, i)
    return (x, y) if c >= min_conf and x >= 0 else None


def _square(cx: float, cy: float, side: float, w: float, h: float) -> Box:
    return clip((cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2), w, h)


def region_rois(region: str, person: TrackedPerson, cfg: AssociationConfig, frame_w: float,
                frame_h: float) -> list[tuple[Box, str]]:
    """ROIs where an item must be: head/eyes (helmet, goggles), torso (vest), one per hand (gloves), one per foot
    (shoes). Keypoints first, person-box geometry as fallback; hands have no geometric fallback (they move too much),
    so without wrist keypoints a glove rule is UNCERTAIN, never a violation."""
    if region in ("head", "eyes"):
        return [head_roi(person, cfg, frame_w, frame_h)]
    box, mc = person.box, cfg.keypoint_min_conf
    bw, bh = box[2] - box[0], box[3] - box[1]
    if region == "torso":
        pts = [p for p in (_pt(person, i, mc) for i in (KP_LSHOULDER, KP_RSHOULDER, KP_LHIP, KP_RHIP)) if p]
        if len(pts) >= 3:
            xs, ys = zip(*pts, strict=False)
            px = 0.2 * max(max(xs) - min(xs), 0.15 * bh)
            return [(clip((min(xs) - px, min(ys) - 0.04 * bh, max(xs) + px, max(ys) + 0.06 * bh), frame_w, frame_h),
                     "keypoints")]
        return [(clip((box[0] + 0.1 * bw, box[1] + 0.18 * bh, box[2] - 0.1 * bw, box[1] + 0.6 * bh), frame_w, frame_h),
                 "geometry")]
    if region == "hands":
        out = []
        for wi, ei in ((KP_LWRIST, KP_LELBOW), (KP_RWRIST, KP_RELBOW)):
            w = _pt(person, wi, mc)
            if w is None:
                continue
            e = _pt(person, ei, mc)
            cx, cy = (w[0] + 0.35 * (w[0] - e[0]), w[1] + 0.35 * (w[1] - e[1])) if e else w  # hand is past the wrist
            out.append((_square(cx, cy, 0.13 * bh, frame_w, frame_h), "keypoints"))
        return out
    if region == "feet":
        out = [(_square(a[0], a[1] + 0.03 * bh, 0.15 * bh, frame_w, frame_h), "keypoints")
               for a in (_pt(person, i, mc) for i in (KP_LANKLE, KP_RANKLE)) if a]
        return out or [(clip((box[0], box[3] - 0.15 * bh, box[2], box[3] + 0.02 * bh), frame_w, frame_h), "geometry")]
    raise ValueError(f"unknown region {region!r}")
