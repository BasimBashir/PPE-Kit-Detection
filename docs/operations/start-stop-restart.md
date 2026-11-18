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

## Run as a service under Linux + systemd (documented, not exercised on this Windows dev host)
The supported form is a systemd unit that keeps Postgres (Docker) and the API up together — full profile:
[deployments/linux/README.md](../../deployments/linux/README.md). Sketch:

```ini
# /etc/systemd/system/ppe-intelligence.service
[Unit]
Description=PPE Intelligence API + pipeline worker
After=docker.service network-online.target
Requires=docker.service

[Service]
User=ppe
WorkingDirectory=/opt/ppe-intelligence/backend
ExecStartPre=/usr/bin/docker compose -f /opt/ppe-intelligence/docker-compose.yml up -d postgres
ExecStart=/usr/local/bin/uv run uvicorn app.main:factory --factory --host 127.0.0.1 --port 8000
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload && sudo systemctl enable --now ppe-intelligence
systemctl status ppe-intelligence      # journalctl -u ppe-intelligence -f for logs
```

The pipeline worker is started by the API process itself (spawn + crash-restart supervision, see
docs/system-design/14-failure-modes-and-recovery.md), so one unit covers both. Put a TLS reverse proxy (nginx /
caddy) in front for LAN access; the API binds 127.0.0.1.

## RTSP over UDP
UDP is a per-camera transport, not a URI scheme: `stream_uri` stays `rtsp://...` (a literal `udp://` URI is
rejected by `validate_stream_uri`), and the camera's `transport` field is set to `udp` (`tcp` default) via the
Cameras page / `PATCH` camera API. It is passed to the decoder as the `rtsp_transport` option
(`backend/app/camera/source.py`). Prefer TCP on unreliable links (UDP drops show up as DEGRADED/OFFLINE
transitions); use UDP only when camera/NAT constraints require it.
