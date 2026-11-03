# ADR-012: prometheus_client + Prometheus/Grafana; opaque bearer sessions, argon2id, three roles

**Status:** ACCEPTED  
**Date:** 2026-11-02

## Context / problem
Operators need to see throughput, latency and failures across two processes, offline (NFR-OBS-001). The dashboard and API need
local authentication with roles (NFR-SEC-001/002) and must never leak camera credentials.

## Decision
Both processes export Prometheus metrics (API :8000/metrics, worker :9101/metrics) with bounded labels (never track or event
ids); alert rules in `deployments/docker/alert-rules.yml`; a provisioned Grafana dashboard. Structured JSON logs with a redaction
filter for resolved credentials. No OpenTelemetry tracing yet (no debugging need identified). Auth: local users (argon2id),
random opaque bearer tokens stored as sha256 hashes, 12 h expiry, revocable, roles viewer/operator/admin enforced per route,
login rate-limited (5 attempts per 60 s per client).

## Alternatives considered
- JWT (not revocable without a denylist - no benefit locally).
- OpenTelemetry (deferred).

## Trade-offs
Opaque tokens need a DB lookup per request (negligible at this scale). No SSO/LDAP; users are created through the API only
(no users page).

## Consequences
Logout and revocation take effect immediately. Credentials referenced as `env:PREFIX` are resolved only at connect time (worker
ingest and the admin camera **Test** probe), never stored in the database or sent to the browser, and masked in log lines
and error messages (NFR-SEC-001).

## Evidence / benchmark
- `promtool check rules`: SUCCESS, 10 rules (docs/testing/test-report.md).
- `backend/tests/integration/test_api.py::test_roles_and_login_rate_limit`, `test_cameras_never_leak_credentials`;
  `test_persistence.py::test_auth_hash_sessions_expiry_revocation`; `backend/tests/unit/test_redaction_and_uri.py` (20 tests).

## Follow-up actions
Users page in the dashboard; TLS termination guidance for non-localhost deployments.
