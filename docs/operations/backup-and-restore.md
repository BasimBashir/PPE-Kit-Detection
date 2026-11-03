# Runbook: Backup and restore

## Backup
```bash
docker compose exec -T postgres pg_dump -U ppe -Fc ppe > backup/ppe_$(date +%F).dump
tar czf backup/evidence_$(date +%F).tgz evidence/
cp configs/*.yaml backup/   # configuration, no secrets (secrets live in .env - back it up separately and securely)
```
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
