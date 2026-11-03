# ADR-011: Native application on the host for GPU work; supporting services in Docker Compose

**Status:** ACCEPTED (edge PROVISIONAL)  
**Date:** 2026-11-02

## Context / problem
Spec 23.1: if GPU containers are problematic on the host, run inference natively and containerise the rest.

## Decision
Development/benchmark host: Windows 11 + RTX 3090, application native (uv venv). Compose runs PostgreSQL, MediaMTX, Toxiproxy,
Prometheus and Grafana. `deployments/linux` documents a systemd + compose profile; `deployments/edge` the Jetson checklist.

## Consequences
All benchmarks are for an RTX 3090 desktop GPU - not an industrial edge device, not representative of Jetson or Orange Pi.

## Follow-up actions
Measure on the intended edge device before any deployment claim (spec 13.3).
