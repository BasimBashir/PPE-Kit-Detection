# ADR-006: Temporal confirmation (N-of-M decisive votes) with hysteresis, cooldown and spatial dedupe

**Status:** ACCEPTED  
**Date:** 2026-11-02

## Context / problem
Interview baseline: a violation must persist ~10-15 frames. Spec 5.1: frame counts mean different durations at different FPS.

## Decision
Per-track FSM (UNKNOWN, COMPLIANT, SUSPECTED_NON_COMPLIANT, CONFIRMED_NON_COMPLIANT, CLEARED). Default: confirm when 12 of the
last 15 **decisive** observations are NO_HELMET (UNCERTAIN never votes); clear after 8 of 10 HELMET (hysteresis).
`mode: duration|both` adds an elapsed-time condition. At 5 processed FPS the minimum confirmation time is (12 - 1) / 5 = 2.2 s
(exposed by the API as `effective_confirm_seconds`). Idempotency key = sha256(camera|track|zone|rule|episode). Re-confirmation of
the same track within `cooldown_s` reopens the existing event; a new track at the same place within `spatial_dedupe_s`
(person-box IoU >= 0.3 or head-position distance <= 0.75 head widths) is merged into the recent event.

## Alternatives considered
Consecutive-frame counter (one missed detection resets it); EMA score threshold (hard to explain to operators).

## Trade-offs
A violation must persist for at least 2.2 s at 5 FPS before it alerts; brief exposures are not reported. Workers who stay
mostly occluded or too small never collect 12 decisive votes and are not assessed (abstention by design, observed on cam03).
Spatial dedupe can merge two different people who stand at the same spot within `spatial_dedupe_s`.

## Consequences
Policy is configured in seconds-aware terms per rule (`configs/rules.example.yaml`: `confirm_n/confirm_m`, `clear_n/clear_m`)
and validated by the API. Cooldown/dedupe memory lives in the worker process, so a worker restart can produce a duplicate alert.
On the looping demo clips the same workers reappear as new tracks every 7-33 s, which inflates events per camera-hour.

## Evidence / benchmark
Measured (docs/benchmarks/README.md): minimum time to confirm (12 votes) 5.5 s / 2.2 s / 1.1 s at 2 / 5 / 10 processed FPS; observed time-to-confirm on events 2.1-3.6 s at 5 FPS. Spatial dedupe extended with a head-distance criterion after live runs showed person-box IoU of only 0.2-0.3 for the same person after ID switches.
Unit tests: `backend/tests/unit/test_ppe_logic.py` (`test_confirm_exactly_at_n_of_m`, `test_single_miss_does_not_flap_and_uncertain_never_votes`, `test_clear_with_hysteresis_and_new_episode`, `test_duration_mode`, `test_both_mode_needs_count_and_duration`); dedupe/cooldown: `backend/tests/integration/test_persistence.py` (`test_cooldown_reopens_and_after_cooldown_creates_new`, `test_spatial_dedupe_merges_new_track_at_same_place`).

## Follow-up actions
Persist dedupe state across worker restarts; tune N-of-M on labelled episodes (event-level recall and false alerts per
camera-hour) instead of demo footage.
