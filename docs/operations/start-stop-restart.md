# Runbook: Start, stop, restart

## Start (demo or site)
```bash
python scripts/start_demo.py            # first run: prerequisites, .env, compose, deps, clips, weights, UI build, app
```
Site deployment (real cameras): set `mode: live` in `configs/app.yaml`, add cameras on the **Cameras** page (or
`configs/cameras.yaml` before first start), put credentials in `.env` as `<PREFIX>_USER` / `<PREFIX>_PASS` and reference
them as `env:<PREFIX>`. Then: `docker compose up -d postgres` and
`uv run --project backend uvicorn app.main:factory --factory --host 127.0.0.1 --port 8000` (from `backend/`).

## Stop
Ctrl+C in the app terminal (stops the worker and the simulator it started); `docker compose down` (data volume kept).

## Restart only the pipeline worker
Dashboard → System health → **Restart worker** (admin), or `POST /api/v1/system/worker/restart`. Expect cameras to show
unknown/offline for ~10-15 s (model load), then ONLINE.

## Verify
`GET /api/v1/ready` → 200 with database/worker/models/disk ok; dashboard banner none; all cameras ONLINE.
