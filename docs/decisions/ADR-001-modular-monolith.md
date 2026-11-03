# ADR-001: Modular monolith with an isolated pipeline worker process

**Status:** ACCEPTED  
**Date:** 2026-11-02

## Context / problem
Ten RTSP streams, GPU inference, event persistence, REST/WebSocket dashboard, one edge box, one small team. Spec 6.1 asks for a
modular monolith with isolated inference worker(s) and warns against premature microservices.

## Decision
Two OS processes from one codebase: (1) the **API process** (FastAPI: REST, WebSocket, outbox dispatcher, retention, worker
supervisor, static frontend) and (2) the **pipeline worker process** (ingest threads, scheduler, inference, tracking, PPE
state machines, event service). They communicate through bounded `multiprocessing` queues (telemetry, commands) and the database.

## Alternatives considered
- Single process: simplest, but a CUDA OOM or native decoder crash would take the API and dashboard down with it.
- Microservices (ingest / inference / events / API) with a broker: independent scaling, but adds Kafka/Redis/gRPC and
  distributed failure modes for a single-node deployment that does not need them (spec 1.4 non-goals).
- One process per camera: strongest isolation, but ten CUDA contexts and no cross-camera batching.

## Trade-offs
A worker crash stops inference for all cameras during the restart window - accepted; the API, history and health reporting
stay up and show it. Cross-process traffic is limited to small JSON and preview JPEGs.

## Consequences
Clear module seams (camera, streaming, inference, tracking, ppe, alerts, evidence) allow later extraction if a second node is needed.

## Evidence / benchmark
Measured 2026-11-02 (docs/benchmarks/fault_injection.md): injected worker crash detected by the API in 0.6 s, all 10 cameras ONLINE again 14.7 s later; the API, history and dashboard stayed available throughout. A DB outage did not slow inference after the circuit-breaker fix.

## Follow-up actions
Re-evaluate when more than one inference node or more than ~30 cameras are required.
