# 08 - PPE state machine and alert semantics

**Status:** VERIFIED · **Related ADRs:** ADR-006 · **Modules:** `ppe/state_machine.py`, `alerts/event_service.py`, `repositories/events.py`

## Purpose and scope
When a violation becomes an alert, and how alerts are deduplicated (FR-TRK-004..008, FR-ALT-001..009).

## Assumptions
- Baseline policy 12 of 15 decisive observations (interview: 10-15 frames).

## Diagram
```mermaid
stateDiagram-v2
  [*] --> UNKNOWN
  UNKNOWN --> COMPLIANT: ≥3 of last 5 decisive = HELMET and latest = HELMET
  UNKNOWN --> SUSPECTED_NON_COMPLIANT: 2 consecutive NO_HELMET (new episode)
  COMPLIANT --> SUSPECTED_NON_COMPLIANT: 2 consecutive NO_HELMET
  SUSPECTED_NON_COMPLIANT --> CONFIRMED_NON_COMPLIANT: 12 of last 15 decisive NO_HELMET (count) / duration rule
  SUSPECTED_NON_COMPLIANT --> COMPLIANT: helmet verified
  SUSPECTED_NON_COMPLIANT --> UNKNOWN: UNCERTAIN share ≥ 60% (same episode continues)
  CONFIRMED_NON_COMPLIANT --> CONFIRMED_NON_COMPLIANT: violation continues / UNCERTAIN (no flapping)
  CONFIRMED_NON_COMPLIANT --> CLEARED: ≥8 of last 10 decisive HELMET, or track lost > 3 s
  CLEARED --> COMPLIANT: helmet verified
  CLEARED --> SUSPECTED_NON_COMPLIANT: new episode (cooldown may reopen the old event)
```
Source: [`diagrams/ppe-state-machine.mmd`](diagrams/ppe-state-machine.mmd)

## Component responsibilities
- State machine per (camera, rule): states and transitions in the diagram; `UNCERTAIN` never votes; hysteresis on clear.
- Event service: idempotency key `sha256(camera|track|zone|rule|episode)` with `episode = run:camera:track:seq`;
  cooldown (60 s) reopens the same event; spatial dedupe (30 s, person IoU ≥ 0.3 **or** head distance ≤ 0.75 head widths)
  merges ID switches; one transaction inserts event + outbox rows.
- Lifecycle `OPEN → ACKNOWLEDGED → RESOLVED` (resolve also allowed from OPEN), every action audited in `event_actions`.

## Interfaces / contracts
REST lifecycle endpoints in [10](10-api-and-websocket-contracts.md); `effective_confirm_seconds` exposed per rule.

## Data ownership
Episode state in the event-service thread; event rows in PostgreSQL.

## Concurrency / process boundaries
Event service thread with bounded queue and DB circuit breaker (5 s) so the GPU loop never waits on the DB.

## Failure modes
Duplicate replay → no duplicate (unique key, verified by test); DB down → spool; cooldown state lost on worker restart (documented).

## Security / privacy
Events contain no identity; feedback labels are evaluation metadata only.

## Performance implications
Time to confirm ≥ 2.2 s at 5 FPS by design; per-FPS values in [15](15-capacity-planning-and-performance.md).

## Alternatives considered
Consecutive-frame counter; EMA score.
