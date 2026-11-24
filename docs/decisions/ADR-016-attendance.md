# ADR-016: Phase 5 - person attendance from door crossings (no faces)

**Status:** ACCEPTED
**Date:** 2026-11-24

## Context / problem
The user's Phase 5 scope (2026-11-24): an operator selects a door-facing camera, draws an entry/exit line, and the
system reports, per working day, how long an enrolled person spent **outside** against a per-person daily budget.
The master spec's hard non-goal is **no face recognition / no biometric profiling**, so the design crux is identity:
minutes-outside is meaningless without saying whose minutes they are, but a face model means a new dependency, new
weights and a gallery of faces.

The existing PPE path is event-shaped (a violation = an event with evidence). Crossings are not violations and occur
far more often, so they cannot ride `events`. What already exists and is reused: the zone model and draw-on-snapshot
editor, ByteTrack ids, the bounded `EventService` queue (never a DB write on the inference thread), the evidence
store, and the report/role patterns.

## Decision
1. **Identity = operator-enrolled persons, matched on torso appearance.** A crossing carries an L2-normalised 8x8x4
   HSV histogram of the upper-torso crop (`app/attendance/identity.py`), matched by cosine similarity (threshold
   0.55) against each person's gallery. No faces, no ReID network, no new dependency. This separates people by
   clothing colour and is weak on identical uniforms - accepted, because every match is reviewable.
2. **Unknown is a first-class bucket, corrections heal.** An unmatched crossing stores its descriptor and a crop, so
   an operator can assign it or enrol it later (`PATCH /crossings/{id}`, `POST /persons/{id}/gallery`). Matching is
   state-constrained: an EXIT can only belong to someone currently inside, an ENTER to someone outside.
3. **A door zone is two normalised points, drawn in the walking direction of an entry.** Positive signed distance
   from the directed line = inside; `s<0 -> s>=0` is an ENTER, the reverse an EXIT. A deadband (0.004) stops
   foot-point jitter from flapping the side; a track gap > 5 s re-learns the side without emitting.
4. **Presence intervals are derived, not authoritative.** `presence_intervals` is rebuilt from a person's crossings
   on every assignment (`rebuild_intervals`: EXIT opens, ENTER closes, a trailing EXIT stays open). Correcting a
   crossing therefore repairs every report that follows.
5. **Crossings reuse `EventService`.** `submit_crossing` is droppable like an alert; `_crossing` matches, saves the
   crop under `crossings/` through the evidence store, inserts idempotently (sha256 of camera|track|zone|frame time)
   and rebuilds only when the row was new. While the DB breaker is open a crossing is dropped and counted
   (`ppe_crossings_total{result="dropped"}`) - no spool, see cut 2.
6. **Reports are day-clipped in the site time zone.** `outside_by_day` intersects intervals with local day bounds
   and caps open intervals at "now"; `site_tz` defaults to UTC. The range is capped at 92 days.
7. **Names stay out of events.** `events` is unchanged; names live on persons and crossings only.

## Rejected alternatives
- **Face embeddings (ArcFace/InsightFace)**: better identity, but a new model, weights, licence and gallery of face
  crops - and the spec's privacy story would need re-arguing. Revisit when site footage shows the cloth histogram is
  not enough.
- **Track-id only**: no roster, so no per-person budget across days.
- **Writing crossings inline from the worker**: breaks the one-thread-writes-to-DB bottleneck the queue exists for.
- **Storing intervals only**: corrections would need manual surgery; replaying crossings is simpler and self-healing.

## Declared cuts (keep honouring until lifted)
1. No per-hour budget - daily budget only.
2. No spool/replay for crossings on DB outage (dropped + metric, documented).
3. No WebSocket live crossing updates - the crossings table reloads on load/filter change.
4. Enrolment is by assigning a crossing crop in the UI; no box-drawing on the live camera or bulk import.

## Consequences
- A clip of the same person on two shifts in the same clothes matches well; two people in identical uniforms will
  both land in the gallery of whoever they match first until the operator re-assigns.
- A crossing happens once per door per pass; someone who never passes a door never leaves an interval.
- Truth is bounded by the door line: loitering past the line without crossing (jitter inside the deadband) is not a
  crossing, and a camera reconnecting mid-pass loses that crossing.
- Measured only on synthetic/unit fixtures so far: threshold 0.55 and torso band are defaults awaiting site footage.
