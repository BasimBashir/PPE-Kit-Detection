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
(IoU >= 0.3) is merged into the recent event.

## Alternatives considered
Consecutive-frame counter (one missed detection resets it); EMA score threshold (hard to explain to operators).

## Evidence / benchmark
Measured (docs/benchmarks/README.md): minimum time to confirm (12 votes) 5.5 s / 2.2 s / 1.1 s at 2 / 5 / 10 processed FPS; observed time-to-confirm on events 2.1-3.6 s at 5 FPS. Spatial dedupe extended with a head-distance criterion after live runs showed person-box IoU of only 0.2-0.3 for the same person after ID switches.
