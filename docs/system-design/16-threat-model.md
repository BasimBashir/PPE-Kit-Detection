# 16 - Threat model (STRIDE, lightweight)

**Status:** PROPOSED (reviewed, not pen-tested) · **Related ADRs:** ADR-012 · **Modules:** whole system

## Purpose and scope
Main threats for a LAN-only deployment and their controls.

## Assumptions
- Attacker on the ship/plant LAN or a curious insider; no internet exposure.

## Diagram

| Threat | Example | Control | Residual |
|---|---|---|---|
| Spoofing | stolen dashboard token | 12 h expiry, revocation, sha256 at rest, role checks | no MFA |
| Tampering | rogue RTSP stream / fake camera | cameras configured by admin only, audit log | RTSP itself unauthenticated on many NVRs |
| Repudiation | "I never resolved it" | `event_actions` + `audit_logs` with actor and time | logs on same host |
| Information disclosure | camera passwords in logs / API | credential refs, redaction, tests | `.env` readable by host admin |
| Information disclosure | evidence images | auth route, no path params, retention | no encryption at rest |
| Denial of service | login brute force; WS flood | rate limit, bounded queues, slow-client drop | no global rate limiting |
| Elevation of privilege | viewer calling admin routes | `require(role)` on every write route (tests) | single seeded admin |
| Privacy | identifying workers | no face/identity models, privacy zones, opaque ids | stock footage shows faces (demo only) |
