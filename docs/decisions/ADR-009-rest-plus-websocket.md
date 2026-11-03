# ADR-009: REST for state and commands, one WebSocket for live updates, HTTP polling for previews

**Status:** ACCEPTED  
**Date:** 2026-11-02

## Context / problem
The dashboard needs commands and history (cameras, events, rules), live state (camera health, new events, metrics) and
low-rate video previews for 10 cameras, with an event-to-screen target of < 1 s (NFR-PERF-005), fully offline.

## Decision
REST `/api/v1` (OpenAPI) for reads and writes; `/ws/live` pushes camera status (1 Hz), events (through the durable outbox) and
metrics, always starting with a full snapshot so reconnecting clients never depend on missed messages. Preview images are polled
(2 Hz on the wall, 4 Hz in detail view); overlays are drawn client-side from JSON so they can be toggled. The dashboard
acknowledges rendering of each new event to measure persist-to-render latency.

## Alternatives considered
- Server-Sent Events (one-way, no render acks).
- WebRTC/HLS (needs a media pipeline per stream for little gain here).

## Trade-offs
Polled JPEG previews are not smooth video and cost bandwidth per open client; acceptable for a monitoring wall, not for
forensic review.

## Consequences
One authenticated socket per browser tab; a rejected token closes it. Events reach the UI only after they are committed
(outbox), so the dashboard never shows an alert that was not persisted.

## Evidence / benchmark
- docs/benchmarks/ui_latency.md: persisted → painted p50 24 ms, p95 52 ms, max 240 ms over 195 events (headless browser,
  camera wall open with 10 live previews).
- `backend/tests/integration/test_api.py::test_event_flow_outbox_ws_render_ack_lifecycle`, `test_ws_rejects_bad_token`.

## Follow-up actions
If many concurrent viewers are needed, serve previews from a shared cache or switch to a streaming protocol.
