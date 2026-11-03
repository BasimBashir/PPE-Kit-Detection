# Runbook: Database unavailable

## Symptoms
Banner *Database unavailable*; `/api/v1/ready` 503; `DatabaseUnavailable` alert; spool pending > 0.

## What keeps working
Ingest, inference, live camera status and previews; new events are written to `spool/events.jsonl` (fsync'd) and
the evidence JPEG is kept. Logged-in operators keep read access for 5 minutes (token cache).

## Recover
`docker compose start postgres` (or fix the DB host). The worker replays the spool every 2 s with the original
idempotency keys (no duplicates). Verify spool pending returns to 0 (System health / `ppe_spool_pending`).

## Do not
Delete `spool/events.jsonl` while it has pending lines - those events exist nowhere else.
