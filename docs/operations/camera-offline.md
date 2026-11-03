# Runbook: Camera offline / degraded / error

## Symptoms
Camera badge **Offline / Degraded / Recovering / Error**; Prometheus `CameraOffline` or `StaleFrames`.

## Diagnose
1. Read the state reason (camera card tooltip, Camera detail → *State reason*). Reasons are redacted FFmpeg/PyAV errors.
2. `ERROR` = configuration: `CredentialError` (env vars missing), `HTTPUnauthorizedError/Forbidden` (wrong user/password).
   Fix credentials in `.env` and restart the app (worker reads env at start).
3. `OFFLINE` with `404` / `EOF` = the NVR has no stream for that path (camera down or unplugged).
4. `DEGRADED` = late frames or low decode rate: network congestion, camera bitrate/FPS changed.
5. Admin → Cameras → **Test** performs a one-shot connection probe from the API host.

## Recover
Reconnect is automatic (backoff ≤ 10 s). Nothing to restart unless the configuration was wrong.

## Verify
State ONLINE, decoded FPS ≈ source FPS, reconnect counter stops increasing.
