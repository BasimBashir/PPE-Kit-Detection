# 02 - Component architecture

**Status:** VERIFIED · **Related ADRs:** ADR-002..006 · **Modules:** `backend/app/*`

## Purpose and scope
Module-level structure inside the two processes.

## Assumptions
- Replaceable adapters: detector (`inference/types.Detector`), tracker (`tracking.PersonTracker`), source (`camera.source`), storage (repositories).

## Diagram
```mermaid
flowchart TB
  subgraph worker["Pipeline worker"]
    SRC["camera.source.CameraSource x N<br/>(PyAV, timeouts, backoff)"] --> SLOT["streaming.slot.LatestFrameSlot x N<br/>(depth 1, drop stale)"]
    HEALTH["camera.health.CameraHealth"] --- SRC
    WD["camera.source.CameraWatchdog"] --> HEALTH
    SLOT --> SCH["streaming.scheduler.InferenceScheduler<br/>(fair, FPS-paced batches)"]
    SCH --> DET["inference.UltralyticsCascadeDetector<br/>pose person → helmet/head"]
    DET --> TRK["tracking.ByteTrackAdapter (per camera)"]
    TRK --> ASSOC["ppe.association.Associator<br/>head ROI · greedy matching · classifier cascade"]
    ASSOC --> FSM["ppe.state_machine.PPEStateMachine<br/>(per camera × rule)"]
    FSM --> EVS["alerts.event_service.EventService<br/>(own thread, circuit breaker)"]
    EVS --> SPOOL["alerts.spool.EventSpool"] & EVID["evidence.store.EvidenceStore"]
  end
  subgraph api["API process"]
    SUP["workers.supervisor.WorkerSupervisor"] --> LIVE["services.live_state.LiveState"]
    DISP["alerts.dispatcher.OutboxDispatcher"] --> HUB["services.ws_hub.WebSocketHub"]
    ROUTES["api.routes_*"] --> REPO["repositories.*"]
    DEMO["services.simulator.SimulatorClient"]
  end
  EVS --> DB[("PostgreSQL")]
  REPO --> DB
  DISP --> DB
```
Source: [`diagrams/component-architecture.mmd`](diagrams/component-architecture.mmd)

## Component responsibilities
See the diagram; each box names its module. Contracts are listed in [implementation-plan §4](../implementation-plan.md).

## Interfaces / contracts
`Frame`, `Detection`, `FrameDetections`, `TrackedPerson`, `Observation`, `Transition` (dataclasses), `ConfirmedViolation` (event service input).

## Data ownership
Trackers and state machines are owned by the inference thread (no locks); episode/dedupe state by the event-service thread.

## Concurrency / process boundaries
Cross-thread hand-offs only through `LatestFrameSlot` (Condition), the event queue (bounded) and telemetry queue (bounded).

## Failure modes
A failed batch is skipped and counted; a failed frame is logged and skipped; nothing kills the inference thread.

## Security / privacy
Privacy masks are applied before evidence/preview encoding. No credentials leave the worker: stream URIs are stored credential-free and resolved from `env:` references at connect time; every stream error passes `core.redaction.redact()`.

## Performance implications
Detector stages batched across cameras; association is pure Python (≈1 ms/frame).

## Alternatives considered
`model.track(persist=True)` (rejected - one tracker state per predictor would mix cameras, ADR-004).
