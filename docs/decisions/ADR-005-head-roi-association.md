# ADR-005: Helmet-to-person association on a head ROI (keypoints first, geometry fallback), global greedy matching

**Status:** ACCEPTED  
**Date:** 2026-11-02

## Context / problem
FR-DET-003/004/007: a helmet near a person, on the ground or in a hand must not count as compliance; occluded heads must abstain.
The interview baseline associated helmets with the upper 10-15 % of the person box by IoU, which credits a helmet carried at
chest height or worn by a neighbour standing close by.

## Decision
Head ROI from pose keypoints (nose/eyes/ears) when at least 2 are confident, otherwise the top 25 % of the person box. A helmet
is a candidate only if most of it lies inside the ROI (IoA >= 0.5), its centre is close to the ROI centre and its width is
plausible relative to the person. Helmets and heads are matched globally greedy by score so one helmet is never credited to two
people. A `head` detection is positive evidence of non-compliance; with neither helmet nor head evidence the outcome is
`UNCERTAIN` (never a violation). The optional classifier only resolves ambiguous cases (cascade, spec 6.5).
All thresholds are in `configs/app.example.yaml` (`association:`).

## Alternatives considered
- IoU(helmet, person box) threshold - fails for helmets carried at chest height.
- IoU on the top 10-15 % of the box (interview baseline) - kept as the `head_roi: geometry` option and benchmarked.
- A classifier on every person - costly (spec 6.5); used only on ambiguity instead.

## Trade-offs
Keypoint ROIs are tighter than the geometric band, so on their own they leave more helmets failing the IoA gate (more
`UNCERTAIN`); they pay off together with the classifier cascade. Abstaining on poor visibility means distant or occluded
workers can go unassessed rather than be falsely flagged.

## Consequences
Requires the pose person model (`yolo26s-pose.pt`); with a plain COCO person model the geometry fallback is used for everyone.
Association stays cheap (≈ 0.3 ms per frame p95) because the classifier runs batched and only on ambiguous crops.

## Evidence / benchmark
- Unit tests for ground / hand / neighbour / occlusion / threshold cases: `backend/tests/unit/test_ppe_logic.py`
  (`test_helmet_on_ground_not_credited`, `test_helmet_in_hand_at_chest_not_credited`,
  `test_one_helmet_never_credited_to_two_people`, `test_occluded_or_tiny_or_truncated_is_uncertain`).
- Ablation at 10 × 5 FPS (docs/benchmarks/README.md §4): uncertain observations per frame 1.20 (COCO + geometry) →
  0.72 (pose + geometry) → 1.12 (pose + keypoints) → **0.35** (pose + keypoints + classifier, deployed).
- Field accuracy: not established (no labelled site footage).

## Follow-up actions
Measure association precision/recall on labelled clips that include carried helmets (the stock demo clips contain none).
