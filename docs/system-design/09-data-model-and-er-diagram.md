# 09 - Data model

**Status:** VERIFIED · **Related ADRs:** ADR-007 · **Modules:** `models/tables.py`, `alembic/versions/0001_initial_schema.py`

## Purpose and scope
Tables, keys and indexes (spec 8).

## Assumptions
- PostgreSQL 17; migrations applied at API start.

## Diagram
```mermaid
erDiagram
  users ||--o{ sessions : has
  cameras ||--o{ zones : has
  cameras ||--o{ camera_health : "state history"
  cameras ||--o{ events : produces
  ppe_rules ||--o{ events : "rule_id"
  zones |o--o{ ppe_rules : "scope"
  events ||--o{ alert_outbox : "delivery per channel"
  events ||--o{ event_actions : "ack / resolve / feedback"
  users { uuid id PK; text username UK; text password_hash; text role; bool enabled }
  sessions { text token_hash PK; uuid user_id FK; timestamptz expires_at; timestamptz revoked_at }
  cameras { text id PK; text stream_uri; text credential_ref; text transport; bool enabled; float target_fps }
  zones { text id PK; text camera_id FK; text kind; jsonb polygon }
  ppe_rules { text id PK; jsonb thresholds; jsonb temporal_policy; text severity }
  events { uuid id PK; text idempotency_key UK; text camera_id FK; int track_id; text episode_id; text status; timestamptz confirmed_at; timestamptz persisted_at; text evidence_path; text feedback }
  alert_outbox { bigint id PK; uuid event_id FK; text channel; text status; int attempts; timestamptz next_attempt_at }
  event_actions { bigint id PK; uuid event_id FK; text actor_id; text action }
  camera_health { bigint id PK; text camera_id FK; text state; timestamptz recorded_at }
  audit_logs { bigint id PK; text actor_id; text action; jsonb metadata }
  model_versions { text id PK; text artifact_hash; text license }
  settings { text key PK; jsonb value }
```
Source: [`diagrams/database-er.mmd`](diagrams/database-er.mmd)

## Component responsibilities
Schema per [implementation-plan §4.10](../implementation-plan.md); `alembic check` reports no drift.

## Interfaces / contracts
Repositories in `backend/app/repositories/` are the only SQL users besides the event service/dispatcher.

## Data ownership
No raw detections or frames are stored; `camera_health` stores state transitions only.

## Concurrency / process boundaries
Worker and API share the DB; outbox claims use `FOR UPDATE SKIP LOCKED` plus a 30 s lease.

## Failure modes
Deleting a camera with events is refused (audit trail) - disable instead.

## Security / privacy
Passwords argon2id; session tokens stored as sha256; audit metadata redacted.

## Performance implications
Indexes on events(camera_id, confirmed_at), status, rule_id; outbox(status, next_attempt_at). No partitioning (not justified).

## Alternatives considered
SQLite (ADR-007).
