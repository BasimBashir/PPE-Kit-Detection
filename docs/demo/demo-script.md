# Demo video script (5-8 min) and recording checklist

## Before recording
- [ ] `python scripts/start_demo.py` → all 10 cameras ONLINE, `/api/v1/ready` 200.
- [ ] Browser at http://127.0.0.1:8000, logged in as admin; second tab on Grafana (http://127.0.0.1:3000) optional.
- [ ] GPU otherwise idle (no training/benchmarks running) so live numbers are representative.
- [ ] Banner "DEMO MODE — SIMULATED RTSP SOURCES" visible (do not hide it).

## Script
1. **Problem & architecture (40 s)** - docs/system-design/01 diagram: offline, 10 RTSP cameras, API + supervised worker.
2. **Camera wall (50 s)** - 10 streams, state badges, source/decoded/processed FPS, frame age. Toggle overlays.
3. **Detection (60 s)** - Camera detail on cam05/cam09: person boxes, track ids, dashed head ROI, helmet (blue) vs bare
   head (orange), state colours. Point out `UNKNOWN` for small/occluded people (cam03 far workers).
4. **Correct association (45 s)** - explain head ROI + global matching; show the unit tests for helmet-on-ground /
   helmet-in-hand / neighbour (`backend/tests/unit/test_ppe_logic.py`). (Stock clips contain no carried helmet - be explicit.)
5. **Temporal alerts (45 s)** - Rules page: 12 of 15 decisive votes ≈ 2.2 s at 5 FPS; Events page: new event appears
   live, evidence snapshot, latency breakdown, acknowledge → resolve, mark true/false alert.
6. **Resilience (60 s)** - Demo control: Unplug cam03 → OFFLINE within ~1 s → Plug in → ONLINE; Crash worker →
   banner + cameras unknown → automatic restart ~15 s; set streams 10 → 4 → 10. Other cameras keep 5 FPS.
7. **Performance (60 s)** - System health: aggregate FPS, GPU/VRAM, worker RSS, stage latency table; then
   docs/benchmarks/results.md (2/5/10 FPS, ablations) and soak results.
8. **Engineering leadership (45 s)** - docs/product/interview-baseline.md (proposal → correction → evidence), ADRs,
   risk register, runbooks, BUILD_STATUS.md.
9. **Close (15 s)** - README quick start; limitations section.

## Integrity rules
Say "simulated cameras / stock footage" out loud; do not quote detector metrics as system recall; state the GPU.
