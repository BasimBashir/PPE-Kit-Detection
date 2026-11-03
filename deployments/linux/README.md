# Linux deployment profile (PROPOSED - not benchmarked)

The verified deployment is the Windows 11 + RTX 3090 development host. This profile documents how the same code runs on a
Linux x86-64 box with an NVIDIA GPU; numbers from the dev host do not transfer.

1. NVIDIA driver + CUDA runtime compatible with the torch wheel in `backend/pyproject.toml` (cu130 index; use the cu12x
   index on older drivers), Docker + compose plugin, ffmpeg, uv, Node 24 (build only).
2. `python3 scripts/start_demo.py --no-start` (or the manual steps it prints).
3. Install the service unit below (adjust paths/user), then `systemctl enable --now ppe-intelligence`.
4. Put a TLS reverse proxy (nginx/caddy) in front if the dashboard must be reachable from the LAN (API binds 127.0.0.1).

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
For real cameras set `mode: live` in `configs/app.yaml` (no simulator, no demo controls, fault API disabled).
