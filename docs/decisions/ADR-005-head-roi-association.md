# ADR-005: Helmet-to-person association on a head ROI (keypoints first, geometry fallback), global greedy matching

**Status:** ACCEPTED  
**Date:** 2026-11-02

## Context / problem
FR-DET-003/004/007: a helmet near a person, on the ground or in a hand must not count as compliance; occluded heads must abstain.

## Decision
Head ROI from pose keypoints (nose/eyes/ears) when at least 2 are confident, otherwise the top 25 % of the person box. A helmet
is a candidate only if most of it lies inside the ROI (IoA >= 0.5), its centre is close to the ROI centre and its width is
plausible relative to the person. Helmets and heads are matched globally greedy by score so one helmet is never credited to two
people. A `head` detection is positive evidence of non-compliance; with neither helmet nor head evidence the outcome is
`UNCERTAIN` (never a violation). The optional classifier only resolves ambiguous cases (cascade, spec 6.5).

## Alternatives considered
IoU(helmet, person box) threshold - fails for helmets carried at chest height; a classifier on every person - costly (spec 6.5).

## Evidence / benchmark
Unit tests cover ground / hand / neighbour / occlusion cases (backend/tests/unit/test_association.py). Field accuracy: not established.
