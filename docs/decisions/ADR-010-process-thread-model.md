# ADR-010: Threads inside the worker; worker supervised by the API process

**Status:** ACCEPTED  
**Date:** 2026-11-02

## Context / problem
10 blocking RTSP decoders, one GPU, per-camera trackers and state machines, and an API that must stay up when the CV pipeline
crashes, then recover on its own (NFR-REL-001 isolation, NFR-REL-002 automatic recovery).

## Decision
Inside the worker: one decode thread per camera (PyAV releases the GIL while decoding), one watchdog thread, one inference
thread that owns the GPU, all trackers and all PPE state machines (no locks on CV state), and one event-service thread (event
persistence and spool replay). The API spawns the worker with the `spawn` start method, watches 1 Hz heartbeats (dead after
10 s without one) and restarts it with bounded exponential backoff (1-30 s); the worker exits when its parent disappears.

## Alternatives considered
- asyncio ingest (PyAV is blocking).
- One process per camera (ADR-001) - 10 model copies in VRAM and cross-process batching.

## Trade-offs
A single Python inference thread is the throughput ceiling: saturation reached 72.6 processed FPS with the GPU only 37 % busy.
A worker crash takes all cameras down until the restart completes.

## Consequences
CV state needs no locks. The dashboard and API keep serving during a worker crash and show the outage.

## Evidence / benchmark
- docs/benchmarks/README.md §3: saturation 72.6 FPS, GPU mean 37 % (bottleneck = Python inference thread + CPU decode).
- Live fault injection `worker_process_crash` (docs/benchmarks/fault_injection.md): detected 0.61 s, all cameras back 14.69 s
  (dominated by model load).

## Follow-up actions
Move tracking/association off the GPU thread, NVDEC decode and TensorRT engines (docs/benchmarks/README.md §3 next steps).
