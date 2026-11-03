# ADR-011: Native application on the host for GPU work; supporting services in Docker Compose

**Status:** ACCEPTED (edge PROVISIONAL)  
**Date:** 2026-11-02

## Context / problem
Spec 23.1: if GPU containers are problematic on the host, run inference natively and containerise the rest. The development host
is Windows 11 with Docker Desktop, where GPU containers add driver/WSL setup risk for reviewers.

## Decision
Development/benchmark host: Windows 11 + RTX 3090, application native (uv venv). Compose runs PostgreSQL, MediaMTX, Toxiproxy,
Prometheus and Grafana. `deployments/linux` documents a systemd + compose profile; `deployments/edge` the Jetson checklist.
`scripts/start_demo.py` falls back to a CPU profile (2 streams × 2 FPS) when no NVIDIA GPU is found.

## Alternatives considered
- Everything in containers with the NVIDIA container toolkit - cleaner on Linux, fragile on Windows/WSL for a reviewer demo.
- DeepStream - edge-oriented, a larger rewrite of the pipeline (not implemented).

## Trade-offs
The host needs Python/uv, Node and ffmpeg installed; the app is not a single image to ship.

## Consequences
All benchmarks are for an RTX 3090 desktop GPU - not an industrial edge device, not representative of Jetson or Orange Pi.
The CPU profile and the Linux profile are documented but not verified.

## Evidence / benchmark
docs/benchmarks/README.md (host: RTX 3090, i5-12400F, Windows 11): 10 × 5 FPS sustained at 50.2 FPS, VRAM 2.7 GB,
worker RSS 2.8 GB; 1 h soak (docs/benchmarks/soak_1h.md) 10/10 cameras online throughout.

## Follow-up actions
Measure on the intended edge device before any deployment claim (spec 13.3); verify the CPU and Linux profiles.
