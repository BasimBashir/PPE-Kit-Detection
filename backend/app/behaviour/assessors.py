"""Phase 4 behaviour assessors (ADR-015) with the Associator interface, so the PPE state machine, events, dedupe,
schedules and alerts apply unchanged. Outcome mapping: ABSENT = the behaviour is happening (violation evidence),
PRESENT = normal, UNCERTAIN = not assessable (never a violation)."""

from __future__ import annotations

import math
from collections import deque

import numpy as np

from app.config.settings import AssociationConfig, BehaviourConfig, RuleThresholds
from app.inference.types import Detection
from app.ppe.geometry import clip, ioa
from app.ppe.head_roi import head_roi, region_rois
from app.ppe.types import Box, Observation, Outcome, TrackedPerson
from app.streaming.frame import Frame

KP_LSH, KP_RSH, KP_LHIP, KP_RHIP = 5, 6, 11, 12
FORGET_S = 30.0  # per-track history of tracks unseen this long is dropped (bounded memory)


def gate(p: TrackedPerson, t: RuleThresholds) -> str | None:
    """Same person gates as the PPE rules: unreliable or tiny detections are never assessed."""
    if p.conf < t.person_conf:
        return "low_person_conf"
    if p.box[3] - p.box[1] < t.min_person_height_px:
        return "person_too_small"
    return None


def _obs(p: TrackedPerson, outcome: Outcome, conf: float, reason: str) -> Observation:
    return Observation(track_id=p.track_id, outcome=outcome, confidence=conf, reason=reason, person_box=p.box,
                       head_roi=None)


class BehaviourAssessor:
    """Shared plumbing: thresholds, person gate, per-track pruning. Subclasses implement _one() (and _drop())."""

    def __init__(self, cfg: BehaviourConfig, thresholds: RuleThresholds):
        self.cfg, self.t = cfg, thresholds
        self.seen: dict[int, float] = {}

    def set_thresholds(self, thresholds: RuleThresholds) -> None:
        self.t = thresholds

    def take_pending(self) -> list:
        return []  # no classifier stage

    def _drop(self, track_id: int) -> None:
        pass

    def _forget(self, now: float) -> None:
        for tid in [k for k, last in self.seen.items() if now - last > FORGET_S]:
            del self.seen[tid]
            self._drop(tid)

    def assess_deferred(self, persons: list[TrackedPerson], dets: list[Detection], frame: Frame) -> list[Observation]:
        self._forget(frame.captured_at)
        out = []
        for p in persons:
            self.seen[p.track_id] = frame.captured_at
            reason = gate(p, self.t)
            out.append(_obs(p, Outcome.UNCERTAIN, 0.0, f"not_assessable:{reason}") if reason
                       else self._one(p, dets, frame))
        return out

    def _one(self, p: TrackedPerson, dets: list[Detection], frame: Frame) -> Observation:
        raise NotImplementedError


def torso(p: TrackedPerson, min_conf: float) -> tuple[float, float, float] | None:
    """(torso angle from vertical in degrees, hip-centre y, mean keypoint conf); None without shoulders + hips."""
    k = p.keypoints
    if k is None or min(k[i, 2] for i in (KP_LSH, KP_RSH, KP_LHIP, KP_RHIP)) < min_conf:
        return None
    sx, sy = (k[KP_LSH, 0] + k[KP_RSH, 0]) / 2, (k[KP_LSH, 1] + k[KP_RSH, 1]) / 2
    hx, hy = (k[KP_LHIP, 0] + k[KP_RHIP, 0]) / 2, (k[KP_LHIP, 1] + k[KP_RHIP, 1]) / 2
    if math.hypot(sx - hx, sy - hy) < 1e-6:
        return None
    conf = float(np.mean([k[i, 2] for i in (KP_LSH, KP_RSH, KP_LHIP, KP_RHIP)]))
    return math.degrees(math.atan2(abs(sx - hx), abs(sy - hy))), float(hy), conf


class FallAssessor(BehaviourAssessor):
    """Fall = a FAST transition: torso upright -> lying within fall_window_s while the hips drop >= drop_frac of the
    standing body height. Lying down slowly, bending (hips stay up) and squatting (torso stays upright) never start an
    episode. After a fall, ABSENT while the person stays lying; the rule's duration policy confirms 'still down'."""

    def __init__(self, cfg: BehaviourConfig, thresholds: RuleThresholds):
        super().__init__(cfg, thresholds)
        self.hist: dict[int, deque[tuple[float, float, float]]] = {}  # (ts, angle, hip_y) within fall_window_s
        self.stand_h: dict[int, float] = {}  # person box height at the last upright sample
        self.fallen: set[int] = set()

    def _drop(self, track_id: int) -> None:
        self.hist.pop(track_id, None)
        self.stand_h.pop(track_id, None)
        self.fallen.discard(track_id)

    def _one(self, p: TrackedPerson, dets: list[Detection], frame: Frame) -> Observation:
        cfg, now, tid = self.cfg, frame.captured_at, p.track_id
        tr = torso(p, cfg.keypoint_min_conf)
        if tr is None:
            return _obs(p, Outcome.UNCERTAIN, 0.0, "not_assessable:torso_not_visible")
        angle, hip_y, conf = tr
        h = self.hist.setdefault(tid, deque())
        h.append((now, angle, hip_y))
        while now - h[0][0] > cfg.fall_window_s:
            h.popleft()
        if angle <= cfg.upright_max_deg:
            self.stand_h[tid] = p.box[3] - p.box[1]
            self.fallen.discard(tid)
            return _obs(p, Outcome.PRESENT, conf, "upright")
        if tid in self.fallen:  # still down at any non-upright angle; getting up hits the branch above
            return _obs(p, Outcome.ABSENT, conf, "down_after_fall")
        if angle < cfg.lying_min_deg:
            return _obs(p, Outcome.UNCERTAIN, 0.0, "transition")  # wait for the next sample
        stand = self.stand_h.get(tid)
        if stand and any(a <= cfg.upright_max_deg and hip_y - y >= cfg.drop_frac * stand for _, a, y in h):
            self.fallen.add(tid)
            return _obs(p, Outcome.ABSENT, conf, "down_after_fall")
        return _obs(p, Outcome.PRESENT, conf, "horizontal_without_fall")


class IdleAssessor(BehaviourAssessor):
    """Stalling = the body barely moves: over idle_window_s every keypoint (and the box centre) stays within
    idle_max_range body heights (5th-95th percentile range, robust to single-frame keypoint glitches). Hands-only
    work counts as moving. Measures stillness, not intent: scope the rule with a work-area zone and a schedule."""

    def __init__(self, cfg: BehaviourConfig, thresholds: RuleThresholds):
        super().__init__(cfg, thresholds)
        self.hist: dict[int, deque[tuple[float, np.ndarray, float]]] = {}  # (ts, (18,2) pixels/NaN, box height)
        self.cached: dict[int, tuple[float, float]] = {}  # track -> (calc ts, move); nanpercentile is ~1.5 ms/call

    def _drop(self, track_id: int) -> None:
        self.hist.pop(track_id, None)
        self.cached.pop(track_id, None)

    def _one(self, p: TrackedPerson, dets: list[Detection], frame: Frame) -> Observation:
        cfg, now = self.cfg, frame.captured_at
        pts = np.full((18, 2), np.nan, np.float32)
        pts[0] = ((p.box[0] + p.box[2]) / 2, (p.box[1] + p.box[3]) / 2)
        k = p.keypoints
        if k is not None:
            ok = k[:, 2] >= cfg.keypoint_min_conf
            pts[1:][ok] = k[ok, :2]
        h = self.hist.setdefault(p.track_id, deque())
        h.append((now, pts, max(p.box[3] - p.box[1], 1e-6)))
        while now - h[0][0] > cfg.idle_window_s:
            h.popleft()
        if now - h[0][0] < 0.8 * cfg.idle_window_s:
            return _obs(p, Outcome.UNCERTAIN, 0.0, "warming_up")
        prev = self.cached.get(p.track_id)
        if prev is not None and now - prev[0] < 0.5:  # shortcut: 2 Hz recompute; a 10 s window changes slowly
            move = prev[1]
        else:
            arr = np.stack([x for _, x, _ in h])  # (T, 18, 2)
            valid = (~np.isnan(arr[..., 0])).sum(axis=0) >= 2
            lo = np.nanpercentile(arr[:, valid], 5, axis=0)
            hi = np.nanpercentile(arr[:, valid], 95, axis=0)
            span = (hi - lo).T  # (2, V)
            move = float(np.max(np.hypot(span[0], span[1]))) / float(np.median([b for _, _, b in h]))
            self.cached[p.track_id] = (now, move)
        if move <= cfg.idle_max_range:
            return _obs(p, Outcome.ABSENT, 1.0 - move / cfg.idle_max_range, "still")
        return _obs(p, Outcome.PRESENT, min(1.0, move / cfg.idle_max_range - 1.0), "moving")


def _grow(b: Box, s: float, w: float, h: float) -> Box:
    cx, cy, hw, hh = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, (b[2] - b[0]) * s / 2, (b[3] - b[1]) * s / 2
    return clip((cx - hw, cy - hh, cx + hw, cy + hh), w, h)


class PhoneAssessor(BehaviourAssessor):
    """Phone use = a detected phone over a hand or at the head (call). Dropouts shorter than phone_gap_s keep the
    evidence, so a small, intermittently detected phone still builds an episode. One phone is evidence for one person
    only (highest overlap wins). Evidence conf = the phone detection's conf."""

    def __init__(self, cfg: BehaviourConfig, thresholds: RuleThresholds, assoc: AssociationConfig):
        super().__init__(cfg, thresholds)
        self.assoc = assoc
        self.last: dict[int, tuple[float, Box, float]] = {}  # track -> (ts, phone box, conf) of the last match

    def _drop(self, track_id: int) -> None:
        self.last.pop(track_id, None)

    def assess_deferred(self, persons: list[TrackedPerson], dets: list[Detection], frame: Frame) -> list[Observation]:
        self._forget(frame.captured_at)
        cfg, w, h = self.cfg, frame.width, frame.height
        phones = [d for d in dets if d.cls == "phone" and d.conf >= self.t.absent_conf]
        gated = [p for p in persons if gate(p, self.t) is None]
        rois = {p.track_id: [_grow(r, cfg.phone_roi_scale, w, h) for r, _ in region_rois("hands", p, self.assoc, w, h)]
                           + [_grow(head_roi(p, self.assoc, w, h)[0], cfg.phone_roi_scale, w, h)] for p in gated}
        cands = sorted(((ioa(d.xyxy, r), p.track_id, i) for p in gated for i, d in enumerate(phones)
                        for r in rois[p.track_id]), reverse=True)
        match, taken = {}, set()
        for score, tid, i in cands:  # greedy by overlap: one phone -> one person
            if score < cfg.phone_min_ioa:
                break
            if tid not in match and i not in taken:
                match[tid] = phones[i]
                taken.add(i)
        out = []
        for p in persons:
            self.seen[p.track_id] = frame.captured_at
            reason = gate(p, self.t)
            out.append(_obs(p, Outcome.UNCERTAIN, 0.0, f"not_assessable:{reason}") if reason
                       else self._decide(p, match.get(p.track_id), frame))
        return out

    def _decide(self, p: TrackedPerson, phone: Detection | None, frame: Frame) -> Observation:
        now = frame.captured_at
        if phone is not None:
            self.last[p.track_id] = (now, phone.xyxy, phone.conf)
        seen = self.last.get(p.track_id)
        if seen and now - seen[0] <= self.cfg.phone_gap_s:
            o = _obs(p, Outcome.ABSENT, seen[2], "phone_at_hand_or_head")
            o.item_roi = o.absent_box = seen[1]
            return o
        return _obs(p, Outcome.PRESENT, 1.0, "no_phone")
