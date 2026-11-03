# ADR-008: One annotated JPEG per event on local disk, age-based retention, disk guard

**Status:** ACCEPTED  
**Date:** 2026-11-02

## Context / problem
Operators need visual proof for each alert (FR-ALT-003/004) on an offline single node, without storing identity data and without
letting a full disk stop detection.

## Decision
`evidence/YYYY/MM/DD/<event-uuid>.jpg` (opaque ids, privacy-zone masks applied before annotation, no identity data); the path
relative to the evidence root is stored in the DB and files are served only via `GET /api/v1/events/{id}/evidence`. A retention
job (hourly) deletes files older than N days (default 30, editable). Free space is checked before each write (default minimum
1 GB); failures mark the event `evidence_error` but never stop the pipeline. Continuous recording is not implemented (FR-ALT-004).

## Alternatives considered
- Object storage (MinIO) - unnecessary for an offline single node.
- Short clips - deferred until snapshots are stable (spec 7.5).

## Trade-offs
A single frame can miss context that a clip would show. Local disk only: evidence is lost with the disk unless backed up.

## Consequences
Evidence is saved before the DB insert, so spooled events keep their snapshot through a database outage. No direct file URLs:
access is always authenticated through the API.

## Evidence / benchmark
- Live fault injection `disk_full_60s` (docs/benchmarks/fault_injection.md): 17 evidence writes skipped during the fault, events
  still persisted, unaffected cameras at 4.8-5.06 FPS.
- `backend/tests/integration/test_persistence.py::test_disk_low_still_persists_event`.

## Follow-up actions
Optional short clips (FR-ALT-003) once a site requires them; include `evidence/` in the backup runbook schedule.
