# ADR-010: Threads inside the worker; worker supervised by the API process

**Status:** ACCEPTED  
**Date:** 2026-11-02

## Decision
Inside the worker: one decode thread per camera (PyAV releases the GIL while decoding), one watchdog thread, one inference
thread that owns the GPU, all trackers and all PPE state machines (no locks on CV state), and one event-replay thread. The API
spawns the worker with the `spawn` start method, watches 1 Hz heartbeats and restarts it with exponential backoff; the worker
exits when its parent disappears.

## Alternatives considered
asyncio ingest (PyAV is blocking); one process per camera (ADR-001).

## Evidence / benchmark
CPU cost of decode threads and inference throughput: docs/benchmarks.
