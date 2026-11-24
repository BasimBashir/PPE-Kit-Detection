# ADR-016: Phase 5 - person attendance from door crossings (face identity, enrolled persons)

**Status:** ACCEPTED (revised 2026-11-24: the original "no faces" cut was lifted by the user)
**Date:** 2026-11-24

## Context / problem
The user's Phase 5 scope: an operator selects a door-facing camera, draws an entry/exit line, and the system reports,
per working day, how long an enrolled person spent **outside** against a per-person daily budget. Minutes-outside is
meaningless without saying whose minutes they are, and the user made enrolment-based recognition a hard requirement:
the operator must be able to register employees (reference captures, re-enrol, retire), and faces must still be
recognisable from the angles real mounts produce - side-wall and roof-top both occur.

The master spec's earlier non-goal was "no face recognition". The user has overridden it for attendance only: faces
identify a **crossing**, never an `event`, so the PPE event schema and its privacy story are untouched.

What already exists and is reused: the zone model and draw-on-snapshot editor, ByteTrack ids, the bounded
`EventService` queue (never a DB write on the inference thread), the evidence store, the report/role patterns, and
OpenCV (already a dependency for the detector/decoder path).

## Decision
1. **Identity = operator-enrolled persons, matched primarily on a face embedding.** YuNet detects a face inside the
   crossing person's box; SFace turns it into a unit-length 128-d embedding (`app/attendance/faces.py`). Both models
   come from the OpenCV model zoo (MIT / Apache-2.0) and are fetched into the models dir on first use, like the
   Ultralytics weights - **no new pip dependency**.
2. **Angle-aware matching, several references per person.** YuNet's 5 landmarks give a yaw/pitch estimate; each face
   is bucketed frontal / quarter / profile and matched against the person's *face* references with a threshold that
   loosens as the face turns (0.45 / 0.38 / 0.32 cosine by default) plus a 0.02 lead over the runner-up person.
   Enrolment stores a reference **per capture**, so one person can hold frontal, three-quarter and profile shots.
   Enrolment paths: a detected face on that camera's live frame (with an optional box when several faces are in the
   frame), or any stored crossing (its face and torso).
3. **Fallbacks when the face is not there.** A crossing also carries the old L2-normalised HSV torso histogram,
   matched (0.55) only when no face was found or the face matched nothing. Within one pass the event service first
   reuses the person already matched on that `(camera, track)` for up to 120 s (track continuity) - a back turned for
   two frames does not flip a person to Unknown. Matched crossings record **how** they matched (`matched_by`).
4. **Unknown is a first-class bucket, corrections heal.** An unmatched crossing stores its face, angle, descriptor
   and a crop, so an operator can assign it or enrol it later (`PATCH /crossings/{id}`,
   `POST /persons/{id}/gallery`). Matching is state-constrained: an EXIT can only belong to someone currently
   inside, an ENTER to someone outside. Retiring a person deletes the record; their crossings fall back to Unknown.
5. **A door zone is two normalised points, drawn in the walking direction of an entry.** Positive signed distance
   from the directed line = inside; `s<0 -> s>=0` is an ENTER, the reverse an EXIT. Door lines are drawn and
   redrawn per camera in the zone editor (IN/OUT side shown while drawing), seeded from an editable default line,
   and validated server-side (2 points, inside the frame, at least 0.05 frame-units apart). A deadband (0.004)
   stops foot-point jitter from flapping the side; a track gap > 5 s re-learns the side without emitting.
6. **Presence intervals are derived, not authoritative.** `presence_intervals` is rebuilt from a person's crossings
   on every assignment (`rebuild_intervals`: EXIT opens, ENTER closes, a trailing EXIT stays open). Correcting a
   crossing therefore repairs every report that follows.
7. **Crossings reuse `EventService`.** `submit_crossing` is droppable like an alert; `_crossing` matches, saves the
   crop under `crossings/` through the evidence store, inserts idempotently (sha256 of camera|track|zone|frame time)
   and rebuilds only when the row was new. While the DB breaker is open a crossing is dropped and counted
   (`ppe_crossings_total{result="dropped"}`) - no spool, see cut 2.
8. **Reports are day-clipped in the site time zone.** `outside_by_day` intersects intervals with local day bounds
   and caps open intervals at "now"; `site_tz` defaults to UTC. The range is capped at 92 days.
9. **Names stay out of events.** `events` is unchanged; names live on persons and crossings only.
10. **The door frame is processed before privacy masking.** A privacy mask pixelates faces; crossings (and so the
    face embedding) are computed on the unmasked frame, and only the stored crop is taken from the masked one.

## Mounting assumptions (what is supported, what is not)
Supported: a **side-wall** camera (roughly 1.8-3 m high, oblique to the door, face visible at least in three-quarter
view on approach) and a **roof-top** camera (2.5-4 m, pointed at the threshold, moderate pitch) - the door line must
be placed where people are clearly and completely visible, not at the frame edge or under the camera.

Hard requirements the installer must meet:
- The camera covers the entry/exit so a person going out or coming in is clearly seen at the line; the face is at
  least ~40 px wide at the line (SFace needs detail; overhead-only crowns and steep >60 deg pitch are weak).
- One door line per physical doorway, drawn perpendicular to the walking path, not along it.
- Lighting is stable enough to see a face; strong backlight at the door degrades both YuNet and SFace.

Not supported: extreme top-down (>60 deg pitch), back-of-head-only passes, faces smaller than ~40 px, and cameras
that see the door only in perspective from far away. Those still work on torso and track-continuity fallbacks, at
lower accuracy.

## Expected accuracy limits and how they are measured
SFace is trained on frontal-ish faces: identity is strong when frontal, weaker at three-quarter, weakest at profile;
the loosened thresholds buy recall per angle at the cost of more confusions, which is why the runner-up margin and
the Unknown bucket exist. Overhead views lose the face entirely past a moderate pitch and fall back to torso/track.

Expected, to be replaced by measurements:
- Frontal, well-lit, person alone at the line: high per-crossing precision; the default thresholds are tuned for
  this bucket.
- Profile / heavy overhead: materially lower recall; expect a share of crossings to reach the torso fallback or
  stay Unknown - they are reviewable, not lost.
- Identical uniforms defeat the torso fallback but not the face.
- Tracks resolve repeated passes within 120 s; a person and a colleague entering together can trade a crossing.

Measurement plan (once door footage exists): label crossings on a staged door (10+ people, 3+ passes each, all three
angle buckets, side-wall and roof mounts), then report per bucket - true accept, false accept, false reject,
Unknown share, and the torso/track fallback share - by replaying the clip through the pipeline and diffing the
matching outcome against the labels (`matched_by` is recorded for exactly this). Defaults move only on that evidence.

## Rejected alternatives
- **Torso histogram only** (the original cut): no enrolment quality, useless under identical uniforms, and the user
  requires recognition of enrolled employees.
- **InsightFace / ArcFace**: stronger embeddings, but a new pip dependency, separate model packaging and a heavier
  licence review; SFace already ships inside the installed OpenCV and its angle sensitivity is handled explicitly.
- **A ReID appearance network**: same dependency cost as faces and still loses under uniforms.
- **Track-id only**: no roster, so no per-person budget across days.
- **Writing crossings inline from the worker**: breaks the one-thread-writes-to-DB bottleneck the queue exists for.
- **Storing intervals only**: corrections would need manual surgery; replaying crossings is simpler and self-healing.

## Declared cuts (keep honouring until lifted)
1. No per-hour budget - daily budget only.
2. No spool/replay for crossings on DB outage (dropped + metric, documented).
3. No WebSocket live crossing updates - the crossings table reloads on load/filter change.
4. No bulk roster import; enrolment is one capture at a time (camera frame or stored crossing).
5. No automatic threshold calibration per camera; the angle thresholds are global defaults until measured.

## Consequences
- Two people in identical uniforms and a similar build can trade crossings; the operator re-assigns, which heals the
  intervals, and enrolment of more references per person reduces it.
- Every matched crossing says how it matched, so a bad camera placement is visible in the data rather than silent.
- Faces exist in the person gallery and crossing crops only; no name reaches `events`.
- A crossing happens once per door per pass; someone who never passes a door never leaves an interval.
- Truth is bounded by the door line: loitering inside the deadband is not a crossing, and a camera reconnecting
  mid-pass loses that crossing.
- Thresholds and torso band are still defaults: no labelled site footage existed at acceptance, so the accuracy
  figures above are expectations plus a measurement plan, not results.
