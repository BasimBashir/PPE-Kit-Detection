"""PPE item -> person association on body-region ROIs (FR-DET-003/004/007, implementation-plan 4.7, ADR-005, ADR-014).

Phase 1 matched helmet / bare-head detections to a head ROI. Phase 3 runs the same matcher for every item: the item
detection (e.g. "gloves") is evidence PRESENT, the bare-region detection (e.g. "no_gloves") is evidence ABSENT, and
each is matched to the region where the item must be (head, torso, each hand, each foot).
"""

from __future__ import annotations

import math

import numpy as np

from app.config.settings import PPE_ITEMS as ITEMS  # item catalogue (configurable: models.yaml ppe_items)
from app.config.settings import AssociationConfig, RuleThresholds
from app.inference.types import Detection, HeadClassifier
from app.ppe.geometry import center, clip, diagonal, ioa
from app.ppe.head_roi import head_roi, region_rois
from app.ppe.types import Box, Observation, Outcome, TrackedPerson
from app.streaming.frame import Frame


class Associator:
    def __init__(self, cfg: AssociationConfig, thresholds: RuleThresholds, classifier: HeadClassifier | None = None,
                 ppe_class: str = "hard_hat"):
        self.cfg = cfg
        self.t = thresholds
        self.item = ITEMS[ppe_class]
        self.width_ratio = self.item.width_ratio or cfg.helmet_width_ratio
        # the head-crop classifier only knows hard hats (FR-DET-006); other items stay UNCERTAIN when ambiguous
        self.classifier = classifier if cfg.use_classifier and ppe_class == "hard_hat" else None
        self._pending: list[tuple[Observation, np.ndarray]] = []

    def set_thresholds(self, thresholds: RuleThresholds) -> None:
        self.t = thresholds

    def _assessable_rois(self, p: TrackedPerson, rois: list[tuple[Box, str]],
                         frame_h: float) -> tuple[str | None, list[tuple[Box, str]]]:
        if p.conf < self.t.person_conf:
            return "low_person_conf", []
        if p.box[3] - p.box[1] < self.t.min_person_height_px:
            return "person_too_small", []
        if self.item.region in ("head", "eyes") and p.box[1] <= 2.0:
            return "head_truncated", []  # box touches the top edge: head may be cut off
        if self.item.region == "feet" and p.box[3] >= frame_h - 2.0:
            return "feet_truncated", []  # box touches the bottom edge: feet may be cut off
        if not rois:
            return "region_not_visible", []  # e.g. no wrist keypoints for a glove rule
        m = self.t.min_roi_px
        kept = [(r, s) for r, s in rois if r[2] - r[0] >= m and r[3] - r[1] >= m]
        if not kept:
            return ("head_too_small" if self.item.region in ("head", "eyes") else "roi_too_small"), []
        return None, kept

    def _score(self, d: Detection, p: TrackedPerson, roi: Box) -> float:
        """0 when d cannot be on this region of p, else ioa * conf * (1 - normalised centre distance)."""
        overlap = ioa(d.xyxy, roi)
        if overlap < self.cfg.min_ioa:
            return 0.0
        (dx, dy), (rx, ry) = center(d.xyxy), center(roi)
        dist = math.hypot(dx - rx, dy - ry) / max(diagonal(roi), 1e-6)
        if dist > self.cfg.max_center_dist:
            return 0.0
        pw = max(p.box[2] - p.box[0], 1e-6)
        lo, hi = self.width_ratio
        if not lo <= (d.xyxy[2] - d.xyxy[0]) / pw <= hi:
            return 0.0
        return overlap * d.conf * (1.0 - dist)

    def assess_deferred(self, persons: list[TrackedPerson], dets: list[Detection], frame: Frame) -> list[Observation]:
        """Like assess(), but ambiguous hard-hat cases are left UNCERTAIN and queued (take_pending) for one batched
        classifier call across all frames of an inference batch."""
        it = self.item
        pres = [d for d in dets if d.cls == it.present and d.conf >= self.t.present_conf]
        absn = [d for d in dets if d.cls == it.absent and d.conf >= self.t.absent_conf]
        heads = [head_roi(p, self.cfg, frame.width, frame.height) for p in persons]
        blocked: list[str | None] = []
        slots: list[tuple[int, Box, str]] = []  # (person index, roi, roi source); hands / feet give two per person
        for pi, p in enumerate(persons):
            rois = [heads[pi]] if it.region in ("head", "eyes") else region_rois(
                it.region, p, self.cfg, frame.width, frame.height)
            reason, kept = self._assessable_rois(p, rois, frame.height)
            blocked.append(reason)
            slots += [(pi, r, s) for r, s in kept]

        # Global greedy matching: highest score first, each detection used once, one of each kind per slot.
        cands: list[tuple[float, str, int, int]] = []
        for si, (pi, roi, _) in enumerate(slots):
            for kind, pool in (("present", pres), ("absent", absn)):
                for di, d in enumerate(pool):
                    s = self._score(d, persons[pi], roi)
                    if s > 0:
                        cands.append((s, kind, si, di))
        cands.sort(key=lambda c: c[0], reverse=True)
        got: dict[tuple[str, int], Detection] = {}
        used: set[tuple[str, int]] = set()
        for _, kind, si, di in cands:
            if (kind, si) in got or (kind, di) in used:
                continue
            got[(kind, si)] = (pres if kind == "present" else absn)[di]
            used.add((kind, di))

        # Per-slot verdict, then per person: any bare region wins (one bare hand = no gloves), then any item seen.
        verdicts: dict[int, list[tuple[Outcome, float, str, int]]] = {}
        for si, (pi, _, _) in enumerate(slots):
            a, b = got.get(("present", si)), got.get(("absent", si))
            if a and not b:
                v = (Outcome.PRESENT, a.conf, "present_on_roi")
            elif b and not a:
                v = (Outcome.ABSENT, b.conf, "absent_on_roi")
            elif a and b and abs(a.conf - b.conf) >= self.cfg.conflict_margin:
                v = (Outcome.PRESENT, a.conf, "conflict_resolved_by_confidence") if a.conf > b.conf else (
                    Outcome.ABSENT, b.conf, "conflict_resolved_by_confidence")
            else:
                v = (Outcome.UNCERTAIN, 0.0, "conflicting_evidence" if a and b else "no_region_evidence")
            verdicts.setdefault(pi, []).append((*v, si))

        obs: list[Observation] = []
        ambiguous: list[tuple[int, int]] = []  # (obs index, slot index)
        for pi, p in enumerate(persons):
            o = Observation(track_id=p.track_id, outcome=Outcome.UNCERTAIN, confidence=0.0, reason="",
                            person_box=p.box, head_roi=heads[pi][0], roi_source=heads[pi][1])
            vs = verdicts.get(pi, [])
            if blocked[pi]:
                o.reason = f"not_assessable:{blocked[pi]}"
            elif vs:
                pick = (max((v for v in vs if v[0] is Outcome.ABSENT), key=lambda v: v[1], default=None)
                        or max((v for v in vs if v[0] is Outcome.PRESENT), key=lambda v: v[1], default=None)
                        or vs[0])
                o.outcome, o.confidence, o.reason, si = pick
                _, o.item_roi, o.roi_source = slots[si]
                pb, ab = got.get(("present", si)), got.get(("absent", si))
                o.present_box, o.absent_box = (pb.xyxy if pb else None), (ab.xyxy if ab else None)
                if o.outcome is Outcome.UNCERTAIN and len(vs) == 1:
                    ambiguous.append((len(obs), si))
            obs.append(o)

        self._pending = [(obs[oi], self._crop(frame.image, slots[si][1])) for oi, si in ambiguous] \
            if self.classifier else []
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
    """Resolve ambiguous hard-hat observations with classifier results (one batched call per inference batch -
    measured bottleneck when called per frame, see docs/benchmarks)."""
    for (o, _crop), (label, prob) in zip(pending, results, strict=True):
        o.classifier_used = True
        o.confidence = prob
        if label == "HELMET_PRESENT":
            o.outcome, o.reason = Outcome.PRESENT, f"classifier:{o.reason}"
        elif label == "HELMET_ABSENT":
            o.outcome, o.reason = Outcome.ABSENT, f"classifier:{o.reason}"
        else:
            o.reason = f"classifier_uncertain:{o.reason}"
