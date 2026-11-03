# ADR-008: One annotated JPEG per event on local disk, age-based retention, disk guard

**Status:** ACCEPTED  
**Date:** 2026-11-02

## Decision
`evidence/YYYY/MM/DD/<event-uuid>.jpg` (opaque ids, privacy masks applied, no identity data); the path relative to the evidence root
is stored in the DB and files are served only via `GET /api/v1/events/{id}/evidence`. A retention job deletes files older than N
days (default 30, editable). Free space is checked before each write; failures mark the event `evidence_error` but never stop the
pipeline. Continuous recording is not implemented (FR-ALT-004).

## Alternatives considered
Object storage (MinIO) - unnecessary for an offline single node; short clips - deferred until snapshots are stable (spec 7.5).
