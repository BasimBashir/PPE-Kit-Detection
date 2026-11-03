# ADR-004: ByteTrack (Ultralytics BYTETracker) with one tracker instance per camera

**Status:** ACCEPTED  
**Date:** 2026-11-02 (revised the same day: implementation library changed, see Consequences)

## Context / problem
FR-TRK-001..003: per-camera track IDs, ByteTrack baseline, adapter for alternatives.

## Decision
ByteTrack via `ultralytics.trackers.byte_tracker.BYTETracker`, one instance per camera inside the worker, wrapped by
`app/tracking/tracker.py:ByteTrackAdapter` (keypoints carried through by detection index). Track IDs are counted per
tracker instance and are local to (camera, worker run); the event episode id includes the worker `run_id`, so IDs never
collide across cameras or restarts (unit test: two trackers number independently).

## Alternatives considered
- `supervision.ByteTrack` (MIT): originally chosen, but the installed supervision 0.31 no longer contains the tracker
  (moved to a separate package) - verified in site-packages during implementation.
- BoT-SORT (camera-motion compensation, ReID) - unnecessary for static cameras unless ID switches are measured to hurt.
- Ultralytics `model.track(persist=True)` - tracker state is bound to one predictor and would mix cameras in a batched call.

## Trade-offs
No ReID: a person leaving and re-entering gets a new ID. Mitigated by spatial dedupe on head position (ADR-006); live
runs showed person-box IoU between the same person's old/new track was only 0.2-0.3 under occlusion, so a head-distance
criterion was added.

## Consequences
Tracker code is AGPL-3.0 like the detector (no new licence class).

## Evidence / benchmark
Tracking 1.8-3.0 ms per batch p95 across all load profiles, 2.06 ms at 10 × 5 FPS (`inference_ms.tracking` in
docs/benchmarks/results/*.json). Independent per-camera numbering: `backend/tests/unit/test_tracking_and_mock.py::
test_two_trackers_are_independent`. ID switches are not measured online (no ground truth).

## Follow-up actions
Measure fragmentation on a labelled clip before considering BoT-SORT.
