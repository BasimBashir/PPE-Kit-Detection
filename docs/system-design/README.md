# System design

Every document lists purpose, assumptions, diagram, responsibilities, interfaces, data ownership, concurrency, failure
modes, security/privacy, performance and alternatives, with a status of `PROPOSED`, `IMPLEMENTED` or `VERIFIED`.
Diagrams are Mermaid (rendered by GitHub); sources in [`diagrams/`](diagrams/).

| # | Document | Status |
|---|---|---|
| 00 | [System context](00-system-context.md) | VERIFIED |
| 01 | [Container architecture](01-container-architecture.md) | VERIFIED |
| 02 | [Component architecture](02-component-architecture.md) | VERIFIED |
| 03 | [Runtime data flow](03-runtime-data-flow.md) | VERIFIED |
| 04 | [Camera ingestion and reconnection](04-camera-ingestion-and-reconnection.md) | VERIFIED |
| 05 | [Scheduling and backpressure](05-multi-camera-scheduling-and-backpressure.md) | VERIFIED |
| 06 | [Computer-vision pipeline](06-computer-vision-pipeline.md) | VERIFIED |
| 07 | [Tracking and PPE association](07-person-tracking-and-ppe-association.md) | VERIFIED (unit level) |
| 08 | [PPE state machine and alert semantics](08-ppe-state-machine-and-alert-semantics.md) | VERIFIED |
| 09 | [Data model](09-data-model-and-er-diagram.md) | VERIFIED |
| 10 | [API and WebSocket contracts](10-api-and-websocket-contracts.md) | VERIFIED (integration tests) |
| 11 | [Deployment topology](11-deployment-topology.md) | dev VERIFIED / edge PROPOSED |
| 12 | [Security and privacy](12-security-and-privacy-boundaries.md) | VERIFIED (tests) |
| 13 | [Observability and SLOs](13-observability-and-slo-design.md) | VERIFIED |
| 14 | [Failure modes and recovery](14-failure-modes-and-recovery.md) | VERIFIED |
| 15 | [Capacity and performance](15-capacity-planning-and-performance.md) | VERIFIED on RTX 3090 |
| 16 | [Threat model](16-threat-model.md) | PROPOSED |
