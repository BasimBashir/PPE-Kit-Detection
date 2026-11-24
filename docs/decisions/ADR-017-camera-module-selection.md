# ADR-017: per-stream module selection — each camera runs only its selected checks

**Status:** ACCEPTED
**Date:** 2026-11-24

## Context / problem
The user's requirement (2026-11-24): "when a stream starts user needs to select which module it needs to detect and
process — 1 stream = 1 class only" for the demo/test scenario, while "in production user can select multiple classes
per stream". Before this change every camera ran the full cascade (person → helmet/head → items → phone), so a
single-check demo stream still paid for every stage and its tile showed unrelated classes (a shoes stream reported
helmets and vests), which muddies both the demo story and the perf numbers.

What already exists and is reused: the camera CRUD + `CameraUpdate` validation against the item/behaviour catalogue
(`PPE_ITEMS`/`BEHAVIOURS`, ADR-014/015, also served by `GET /ppe-items`), the camera source → frame → batched
detector pipeline, and the camera hot-reload path (`PATCH /cameras/{id}` already restarts a camera on config
change).

## Decision
1. **The camera row stores the selection; empty = full cascade.** `cameras.module` (migration 0006) is a nullable
   CSV of check ids; over the API it is `module: list[str] | None`, validated against `PPE_ITEMS ∪ BEHAVIOURS ∪
   {"person"}`. Empty/None keeps today's full cascade, so existing deployments and configs behave exactly as before.
2. **The selection rides the frame; gating happens in the detector.** The camera source stamps each frame with its
   camera's selection (`Frame.module`); `UltralyticsCascadeDetector.detect()` maps the selection to the inference
   stages and detector classes the frame needs (`_module_plan`), batches each stage over only the frames that
   selected it, and drops non-selected classes per frame. Unknown names fail open to the full cascade (a config
   typo degrades to today's behaviour, never to blindness).
3. **UI: multi-select checkboxes on the camera form; the wall shows the selection.** The Cameras admin form lists
   the catalogue (`GET /ppe-items` + "Person (tracking / attendance only)"); the camera wall shows each stream's
   checks next to its name, and its tile's strip reports each check's live verdicts in operator terms (e.g.
   "Hard hat: 1 NO HAT · 2 OK | Safety vest: 1 checking", "Attendance: 1 in view"), never raw model classes; the
   preview keeps one entry per (person, check) so a multi-check stream shows every check. Changing the
   selection hot-restarts just that camera.
4. **Rules are not auto-filtered.** A rule whose check is outside its camera's selection simply never sees
   detections (they are dropped upstream). The wall tile makes the selection visible; keeping rule and module
   scopes aligned is the operator's job (stated in the README limitations).

## Consequences
- **Measured:** with one check per demo stream, every stream reached the 5 FPS processing target (previously
  3.0–4.2 with the full cascade on all 11 streams) — the unselected stages no longer run.
- Unknown-module fail-open means a typo silently widens a stream back to the full cascade; the validated API layer
  makes this unlikely (unknown ids are a 422), and the wall tile shows what actually runs.
- No schema/behaviour change for `module=null` consumers; migration 0006 is additive.

## Alternatives considered
- Per-camera model instances (one detector per selection) — rejected: N models × N cameras of VRAM for no batching
  benefit over per-frame stage grouping inside one process.
- Filtering only at the rule layer (keep the full cascade) — rejected: the user asked for detection/processing to
  be selected per stream, and the wasted stage time was the measured pain.
- Validating rules against the selection on save — deferred; a useful guardrail, not needed for correctness.
