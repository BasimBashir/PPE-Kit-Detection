# Edge (Jetson / mini-PC) readiness checklist - PROPOSED, nothing here is measured

The interview proposed "Jetson or Orange Pi is sufficient". This project does **not** claim that. Before any edge claim,
measure with `scripts/benchmark/run_benchmark.py` on the device itself.

| Item | Why it matters | Status |
|---|---|---|
| JetPack / L4T version ↔ PyTorch / Ultralytics wheels | Jetson uses NVIDIA-built wheels, not PyPI CUDA wheels | to verify on device |
| TensorRT export of the three models (`yolo export format=engine half=True`) | 2-5x inference speed-up typical on Jetson; accuracy must be re-checked | not done |
| Hardware decode (nvv4l2decoder via GStreamer, or NVDEC) | 10 × 720p30 software decode is a large share of a Jetson's CPU | not implemented (ADR-002) |
| DeepStream pipeline | multi-stream batching + decode on GPU | out of scope for v0.1 |
| Memory: models + 10 decoder contexts | Orin Nano 8 GB is tight | to measure |
| Thermal / power mode (`nvpmodel`, `jetson_clocks`) | sustained throughput drops when throttled | to measure in an 8 h soak |
| Orange Pi (Rockchip NPU) | needs RKNN conversion; CUDA path unavailable | not supported |

Sizing method (from docs/system-design/15): aggregate inference load = cameras × processed FPS (baseline 50 frames/s,
two detector stages each); decode load = cameras × source FPS. Pick the device whose measured processed FPS at the
baseline profile keeps frame age at inference p95 < 0.5 s with ≥ 30 % headroom.
