# ADR-009: REST for state and commands, one WebSocket for live updates, HTTP polling for previews

**Status:** ACCEPTED  
**Date:** 2026-11-02

## Decision
REST `/api/v1` (OpenAPI) for reads and writes; `/ws/live` pushes camera status (1 Hz), events (through the durable outbox) and
metrics, always starting with a full snapshot so reconnecting clients never depend on missed messages. Preview images are polled
(<= 2 Hz on the wall, 4 Hz in detail view); overlays are drawn client-side from JSON so they can be toggled. The dashboard
acknowledges rendering of each new event to measure persist-to-render latency.

## Alternatives considered
Server-Sent Events (one-way, no render acks); WebRTC/HLS (needs a media pipeline per stream for little gain here).
