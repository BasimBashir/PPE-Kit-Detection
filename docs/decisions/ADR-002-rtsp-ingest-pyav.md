# ADR-002: RTSP ingest with PyAV (FFmpeg) on the development host; GStreamer/DeepStream for edge

**Status:** ACCEPTED (edge path PROVISIONAL)  
**Date:** 2026-11-02

## Context / problem
Spec 6.3 prefers GStreamer where supported and warns that `cv2.VideoCapture` alone is not sufficient for reliability.
The development host is Windows 11, where GStreamer Python bindings are fragile to install and hard to reproduce.

## Decision
PyAV 19 (FFmpeg libraries), one decode thread per camera, explicit open/read timeouts (`av.open(..., timeout=(5, 5))`),
`rtsp_transport=tcp` by default (UDP selectable per camera), our own watchdog for stall detection based on frame age rather
than socket state (FR-CAM-004), and bounded exponential backoff with jitter. Frames are decoded continuously (H.264
inter-frame dependency) but converted to arrays only at 2x the processed FPS to save CPU.

## Alternatives considered
- OpenCV VideoCapture: no per-read timeout control, opaque reconnect behaviour, no timestamps - rejected.
- GStreamer (rtspsrc + decodebin / nvv4l2decoder): best on Linux/Jetson with hardware decode; not reproducible on this host.
- NVIDIA DeepStream: multi-stream GPU pipeline, Linux only; a later optimisation (spec 13.3).

## Trade-offs
CPU decode costs CPU per stream (measured in docs/benchmarks). Breaking a blocked read relies on PyAV's timeout callback plus
closing the container from the watchdog.

## Consequences
`app/camera/source.py` is the only module that knows about PyAV; an edge build can add a GStreamer source with the same contract.

## Evidence / benchmark
Verified 2026-11-02: MediaMTX -> Toxiproxy -> PyAV read, 720p H.264, open 1.34 s, 30 fps decode; wrong credentials -> 401.
PyAV's 401 message embeds `user:password@` in the URL - all stream errors pass through `app/core/redaction.py`.

## Follow-up actions
Evaluate NVDEC (`ingest.hwaccel: cuda`) in Phase 9; GStreamer source for the Linux/Jetson profile.
