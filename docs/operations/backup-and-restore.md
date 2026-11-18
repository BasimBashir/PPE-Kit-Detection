# Runbook: Backup and restore

## Backup
```bash
EVIDENCE_DIR=${PPE_EVIDENCE_DIR:-evidence}   # as configured in .env (relative paths are from the repo root)
docker compose exec -T postgres pg_dump -U ppe -Fc ppe > backup/ppe_$(date +%F).dump
tar czf backup/evidence_$(date +%F).tgz "$EVIDENCE_DIR"
cp configs/*.yaml backup/   # configuration, no secrets (secrets live in .env - back it up separately and securely)
```
## Schedule
Daily, database and evidence together (events reference evidence files by relative path, so restore both from the same
date). Evidence grows with every event; the archive is bounded only by `evidence.retention_days` (default 30) and
the retention cleanup below - size the backup target for that window and prune old archives on the same schedule.
## Restore
```bash
docker compose exec -T postgres pg_restore -U ppe -d ppe --clean < backup/ppe_<date>.dump
tar xzf backup/evidence_<date>.tgz
```
Start the app; Alembic upgrades the schema if the backup is older.

## Retention cleanup
Automatic hourly (`evidence.retention_check_interval_s`) and on demand (Rules & settings).

## Upgrade / rollback
Stop app → backup → `git pull` → `uv sync` → start (migrations run). Rollback: previous git tag + restore the backup.
