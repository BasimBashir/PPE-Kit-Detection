"""Helmet/head -> person association on a head ROI (FR-DET-003/004/007, implementation-plan 4.7, ADR-005)."""

from __future__ import annotations

import math

import numpy as np

from app.config.settings import AssociationConfig, RuleThresholds
from app.inference.types import Detection, HeadClassifier
from app.ppe.geometry import center, clip, diagonal, ioa
from app.ppe.head_roi import head_roi
from app.ppe.types import Box, Observation, Outcome, TrackedPerson
from app.streaming.frame import Frame


class Associator:
    def __init__(self, cfg: AssociationConfig, thresholds: RuleThresholds, classifier: HeadClassifier | None = None):
        self.cfg = cfg
        self.t = thresholds
        self.classifier = classifier if cfg.use_classifier else None
        self._pending: list[tuple[Observation, np.ndarray]] = []

    def set_thresholds(self, thresholds: RuleThresholds) -> None:
        self.t = thresholds

    def _not_assessable(self, p: TrackedPerson, roi: Box) -> str | None:
        if p.conf < self.t.person_conf:
            return "low_person_conf"
        if p.box[3] - p.box[1] < self.t.min_person_height_px:
            return "person_too_small"
        if p.box[1] <= 2.0:
            return "head_truncated"  # box touches the top edge: head may be cut off
        if roi[2] - roi[0] < self.t.min_head_px or roi[3] - roi[1] < self.t.min_head_px:
            return "head_too_small"
        return None

    def _score(self, d: Detection, p: TrackedPerson, roi: Box) -> float:
        """0 when d cannot be on p's head, else ioa * conf * (1 - normalised centre distance)."""
        overlap = ioa(d.xyxy, roi)
        if overlap < self.cfg.min_ioa:
            return 0.0
        (dx, dy), (rx, ry) = center(d.xyxy), center(roi)
        dist = math.hypot(dx - rx, dy - ry) / max(diagonal(roi), 1e-6)
        if dist > self.cfg.max_center_dist:
            return 0.0
        pw = max(p.box[2] - p.box[0], 1e-6)
        lo, hi = self.cfg.helmet_width_ratio
        if not lo <= (d.xyxy[2] - d.xyxy[0]) / pw <= hi:
            return 0.0
        return overlap * d.conf * (1.0 - dist)

    def assess_deferred(self, persons: list[TrackedPerson], dets: list[Detection], frame: Frame) -> list[Observation]:
        """Like assess(), but ambiguous cases are left UNCERTAIN and queued (take_pending) for one batched
        classifier call across all frames of an inference batch."""
        helmets = [d for d in dets if d.cls == "helmet" and d.conf >= self.t.helmet_conf]
        heads = [d for d in dets if d.cls == "head" and d.conf >= self.t.head_conf]
        rois: list[tuple[Box, str]] = [head_roi(p, self.cfg, frame.width, frame.height) for p in persons]
        blocked = [self._not_assessable(p, roi) for p, (roi, _) in zip(persons, rois, strict=True)]

        # Global greedy matching: highest score first, each detection used once, one helmet + one head per person.
        cands: list[tuple[float, str, int, int]] = []
        for pi, p in enumerate(persons):
            if blocked[pi]:
                continue
            roi = rois[pi][0]
            for kind, pool in (("helmet", helmets), ("head", heads)):
                for di, d in enumerate(pool):
                    s = self._score(d, p, roi)
                    if s > 0:
                        cands.append((s, kind, pi, di))
        cands.sort(key=lambda c: c[0], reverse=True)
        got: dict[tuple[str, int], Detection] = {}
        used: set[tuple[str, int]] = set()
        for _, kind, pi, di in cands:
            if (kind, pi) in got or (kind, di) in used:
                continue
            got[(kind, pi)] = (helmets if kind == "helmet" else heads)[di]
            used.add((kind, di))

        obs: list[Observation] = []
        ambiguous: list[int] = []
        for pi, p in enumerate(persons):
            roi, src = rois[pi]
            helmet, head = got.get(("helmet", pi)), got.get(("head", pi))
            o = Observation(
                track_id=p.track_id, outcome=Outcome.UNCERTAIN, confidence=0.0, reason="",
                person_box=p.box, head_roi=roi, roi_source=src,
                helmet_box=helmet.xyxy if helmet else None, head_box=head.xyxy if head else None,
            )
            if blocked[pi]:
                o.reason = f"not_assessable:{blocked[pi]}"
            elif helmet and not head:
                o.outcome, o.confidence, o.reason = Outcome.HELMET, helmet.conf, "helmet_on_head_roi"
            elif head and not helmet:
                o.outcome, o.confidence, o.reason = Outcome.NO_HELMET, head.conf, "bare_head_on_head_roi"
            elif helmet and head and abs(helmet.conf - head.conf) >= self.cfg.conflict_margin:
                win = helmet if helmet.conf > head.conf else head
                o.outcome = Outcome.HELMET if win is helmet else Outcome.NO_HELMET
                o.confidence, o.reason = win.conf, "conflict_resolved_by_confidence"
            else:
                o.reason = "conflicting_evidence" if helmet and head else "no_head_evidence"
                ambiguous.append(pi)
            obs.append(o)

        self._pending = [(obs[pi], self._crop(frame.image, obs[pi].head_roi))  # type: ignore[arg-type]
                         for pi in ambiguous] if self.classifier else []
        return obs

    def assess(self, persons: list[TrackedPerson], dets: list[Detection], frame: Frame) -> list[Observation]:
        obs = self.assess_deferred(persons, dets, frame)
        pending = self.take_pending()
        if pending and self.classifier:
            apply_classification(pending, self.classifier.classify([c for _, c in pending]))
        return obs

    def take_pending(self) -> list[tuple[Observation, np.ndarray]]:
        """Ambiguous observations (with head crops) left by the last assess_deferred() call."""
        p, self._pending = self._pending, []
        return p

    @staticmethod
    def _crop(image: np.ndarray, roi: Box) -> np.ndarray:
        h, w = image.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in clip(roi, w, h))
        return image[y1:max(y2, y1 + 1), x1:max(x2, x1 + 1)].copy()


def apply_classification(pending: list[tuple[Observation, np.ndarray]], results: list[tuple[str, float]]) -> None:
    """Resolve ambiguous observations with classifier results (one batched call per inference batch - measured
    bottleneck when called per frame, see docs/benchmarks)."""
    for (o, _crop), (label, prob) in zip(pending, results, strict=True):
        o.classifier_used = True
        o.confidence = prob
        if label == "HELMET_PRESENT":
            o.outcome, o.reason = Outcome.HELMET, f"classifier:{o.reason}"
        elif label == "HELMET_ABSENT":
            o.outcome, o.reason = Outcome.NO_HELMET, f"classifier:{o.reason}"
        else:
            o.reason = f"classifier_uncertain:{o.reason}"
