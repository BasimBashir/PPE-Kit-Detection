# ADR-012: prometheus_client + Prometheus/Grafana; opaque bearer sessions, argon2id, three roles

**Status:** ACCEPTED  
**Date:** 2026-11-02

## Decision
Both processes export Prometheus metrics (API :8000/metrics, worker :9101/metrics) with bounded labels; alert rules in
`deployments/docker/alert-rules.yml`; a provisioned Grafana dashboard. Structured JSON logs with a redaction filter. No
OpenTelemetry tracing yet (no debugging need identified). Auth: local users (argon2id), random opaque bearer tokens stored as
sha256 hashes, 12 h expiry, roles viewer/operator/admin enforced per route, login rate-limited.

## Alternatives considered
JWT (not revocable without a denylist - no benefit locally); OpenTelemetry (deferred).
