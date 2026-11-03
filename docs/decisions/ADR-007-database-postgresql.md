# ADR-007: PostgreSQL 17 + SQLAlchemy 2 + Alembic; local JSONL spool during outages

**Status:** ACCEPTED  
**Date:** 2026-11-02

## Context / problem
Durable events, outbox and audit trail; two writer processes (API and pipeline worker); restart recovery (NFR-REL-004). A database
outage must neither stall the pipeline nor lose or duplicate alerts.

## Decision
PostgreSQL 17 in docker compose; sync SQLAlchemy 2 (psycopg 3); Alembic migrations applied at API start. The outbox is claimed
with `SELECT … FOR UPDATE SKIP LOCKED`. When the database is unreachable the worker appends events (evidence already on disk) to
an fsync'ed JSONL spool and replays them with the same idempotency key once the database returns - no loss and no duplicates.
A circuit breaker (5 s) stops further connect attempts after a failure, so later events are spooled without waiting on
connect timeouts.

## Alternatives considered
- SQLite (allowed by the spec for a single-process demo; weaker with two writer processes and no SKIP LOCKED outbox claims).
- An in-memory queue during outages - loses events if the worker crashes during the outage.

## Trade-offs
One more service to run (in compose) versus SQLite's zero setup. The spool is local to the worker host; it is not replicated.

## Consequences
Events are written exactly once even across outages and replays (unique idempotency key). `alembic check` guards against
model/migration drift. Backup and restore are documented in docs/operations/backup-and-restore.md.

## Evidence / benchmark
- `backend/tests/integration/test_persistence.py::test_db_outage_spools_then_replays_exactly_once` (spool, breaker, replay,
  duplicate replay rejected).
- Live fault injection `database_outage_60s` (docs/benchmarks/fault_injection.md): 2 events spooled during a 60 s outage,
  0 pending after recovery, unaffected cameras stayed at 4.8-5.05 FPS.

## Follow-up actions
Persist cooldown/dedupe memory (currently in-process) so a worker restart cannot produce a duplicate alert.
