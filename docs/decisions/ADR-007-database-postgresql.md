# ADR-007: PostgreSQL 17 + SQLAlchemy 2 + Alembic; local JSONL spool during outages

**Status:** ACCEPTED  
**Date:** 2026-11-02

## Context / problem
Durable events, outbox and audit trail; two writer processes; restart recovery (NFR-REL-004).

## Decision
PostgreSQL 17 in docker compose; sync SQLAlchemy 2 (psycopg 3); Alembic migrations applied at API start. When the database is
unreachable the worker appends events (evidence already on disk) to an fsync'ed JSONL spool and replays them with the same
idempotency key once the database returns - no loss and no duplicates.

## Alternatives considered
SQLite (allowed by the spec for a single-process demo; weaker with two writer processes and no SKIP LOCKED outbox claims).

## Evidence / benchmark
DB-outage fault test: backend/tests/fault_injection; results in docs/testing/test-report.md.
