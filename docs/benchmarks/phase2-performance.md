# Phase 2 performance: TensorRT FP16, NVDEC probe, CPU-only profile, tracking placement

Measured 2026-11-18 on the dev host. All numbers in sections 1-4 are **measured on this host** with no compute
processes on the GPU (desktop compositor only; NVML shows momentary desktop load). Scripts live outside the repo in
`C:/Users/mianx/ppe-work/trt/` (`bench_trt.py`, `bench_cpu.py`, raw JSON next to them); TensorRT engines are also
outside the repo. TensorRT and NVDEC are treated as **Linux/Docker deployment features**; they worked on this
Windows host, so the numbers below are measured, and the Linux/Docker enablement notes live in
[deployments/linux/README.md](../../deployments/linux/README.md).

**Host:** NVIDIA GeForce RTX 3090 24 GB (driver 621.24) · Intel Core i5-12400F (6C/12T) · 64 GB RAM ·
Windows 11 Pro 10.0.26300 · Python 3.12.10 · torch 2.14.1+cu130 (CUDA 13.0) · Ultralytics 8.4.198 ·
TensorRT 11.5.0.63 (venv only) · NVIDIA ModelOpt (AutoCast, venv only) · ffmpeg 8.1.2 (gyan.dev full build,
includes `cuda` hwaccel and `h264_cuvid`).

## 1. TensorRT FP16 vs PyTorch FP16 (measured)

Installed into the ppe-intelligence venv only (never in `backend/pyproject.toml` / `uv.lock`):

```bash
uv pip install --python C:/Users/mianx/.venvs/ppe-intelligence/Scripts/python.exe tensorrt "nvidia-modelopt[onnx]" onnx onnxslim
```

Exports (weights copied to `C:/Users/mianx/ppe-work/trt/`, engines written there; nothing under `models/` changed):

```python
YOLO(".../ppe_yolo26s.pt").export(format="engine", half=True, imgsz=640, batch=10, dynamic=True, device=0, workspace=4)
YOLO(".../head_cls_yolo26n.pt").export(format="engine", half=True, imgsz=96, batch=32, dynamic=True, device=0, workspace=2)
```

Export took 139 s (detector, 23.2 MB engine) and 79 s (classifier, 5 MB engine). TensorRT 11 removed the FP16
builder flags, so Ultralytics bakes FP16 into the ONNX graph with NVIDIA ModelOpt AutoCast before building the
engine — that is why `nvidia-modelopt[onnx]` is required for `half=True` on TRT 11.

Method: real frames from the cam01 demo clip (1280×720 BGR), the production `YOLO.predict` call signature
(`imgsz=640`/`96`, `conf=0.25`, `quantize=16`, `device=0`), batches 1/4/10 (production `max_batch=10`, median
batch 4). Cascade = detector → one classifier call per batch with 0.29 crops per frame (the measured share of
classifier invocations per processed frame, `docs/benchmarks/results/cams10_fps5.json`). The production person
stage (`yolo26s-pose.pt`) was **not** exported or measured — see section 6.

| model set | backend | batch | per-frame ms p50 | p95 | max sustainable fps |
|---|---|---|---|---|---|
| detector (`ppe_yolo26s`) | PyTorch FP16 | 1 | 11.3 | 16.5 | 83.6 |
| detector | **TensorRT FP16** | 1 | **4.7** | 7.8 | **196.6** |
| detector | PyTorch FP16 | 4 | 3.9 | 5.7 | 243.9 |
| detector | **TensorRT FP16** | 4 | **2.4** | 3.1 | **403.9** |
| detector | PyTorch FP16 | 10 | 2.5 | 3.0 | 389.0 |
| detector | **TensorRT FP16** | 10 | **2.0** | 2.4 | **498.1** |
| cascade (det → classifier) | PyTorch FP16 | 1 | 15.8 | 22.5 | 59.6 |
| cascade | **TensorRT FP16** | 1 | **6.7** | 10.4 | **142.3** |
| cascade | PyTorch FP16 | 4 | 5.1 | 8.1 | 180.8 |
| cascade | **TensorRT FP16** | 4 | **3.0** | 4.4 | **319.3** |
| cascade | PyTorch FP16 | 10 | 3.0 | 3.6 | 325.6 |
| cascade | **TensorRT FP16** | 10 | **2.2** | 3.0 | **439.5** |

Reading:

- **Single-stream (batch 1): TRT is 2.4× faster** end-to-end per call (cascade p50 15.8 → 6.7 ms; detector p50
  11.3 → 4.7 ms). This is the regime a per-camera path would see if inference were per-stream.
- **At production batch sizes (4-10) the gap shrinks to 1.25-1.35×**: per-call Python pre/post-processing
  (letterbox, NMS/boxes, tensor→numpy) dominates; the GPU kernels are no longer the bottleneck. TRT alone will
  not lift the ~72.6 FPS single-thread saturation ceiling much.
- The person-pose stage is roughly half of the GPU time per frame (6.9 ms p50 vs 6.5 ms ppe + 2.8 ms classifier
  at baseline); exporting it too and running the full pipeline on engines is the step that could lift saturation
  meaningfully — not measured here.

## 2. NVDEC hardware decode probe (measured)

Production decodes RTSP via PyAV/OpenCV on CPU. Probe: a local 1080p h264 file (`probe_1080p.mp4`, created by
upscaling the 720p cam01 clip with libx264 — production sources are 720p, so CPU numbers below are conservative).
ffmpeg 8.1.2 (gyan.dev full build) with `cuda` hwaccel + `h264_cuvid`; pip `opencv-python` was also probed.

| probe (1080p h264) | fps |
|---|---|
| CPU software decode only (`-f null`) | 941-952 |
| NVDEC decode only (`-hwaccel cuda -c:v h264_cuvid`, frames downloaded) | 581 |
| NVDEC decode only, frames stay on GPU (`-hwaccel_output_format cuda`) | 604 |
| decode + convert to BGR + pipe out, CPU | 296 |
| decode + convert to BGR, NVDEC (`h264_cuvid`) | 296 |
| `cv2.VideoCapture` + `OPENCV_FFMPEG_CAPTURE_OPTIONS=hwaccel;cuda\|video_codec;h264` | 293 (env var silently ignored: pip OpenCV bundles an FFmpeg without CUDA) |

Recommendation: **not worth integrating now.** CPU FFmpeg software decode of 1080p h264 is ~945 fps decode-only
and ~296 fps through the to-BGR path the pipeline actually needs — an order of magnitude above the worst required
load (~300 decoded fps for 10 cameras × 30 source fps) and far above the production load (50 processed fps at 48%
host CPU). NVDEC does not beat CPU here (581-604 vs 941-952 decode-only) because this CPU is fast and h264 is
cheap; end-to-end both paths equalize at ~296 fps where the download + swscale-to-BGR conversion dominates.
Integration would cost a CUDA-enabled FFmpeg/OpenCV build (pip OpenCV cannot reach NVDEC) or an ffmpeg-CLI
subprocess, plus per-camera GPU session management — for zero measured gain at this camera count. Revisit for
high-resolution / high-camera-count Linux deployments, and then with GPU-side scaling (`-hwaccel_output_format
cuda` + GPU resize) so frames never copy back per frame. `ingest.hwaccel` stays reserved (ADR-002).

## 3. CPU-only profile (measured)

Production inference cascade on `device="cpu"` (same classes as production: `yolo26s-pose` 640 → `ppe_yolo26s`
640 → head classifier 96), real 720p frames, torch on 6 threads (default: physical cores). Boundary: **inference
cascade only** — decode, tracking, association, FSM, previews excluded; frames pre-decoded from one clip.

| batch | per-frame ms p50 | p95 | cascade fps | cameras at 2 FPS |
|---|---|---|---|---|
| 1 (per-stream) | 115.9 | 122.6 | 8.6 | 4 |
| 4 | 95.7 | 100.1 | 10.4 | 5 |
| 10 | 95.1 | 98.4 | 10.5 | 5 |

Conclusion: on this i5-12400F a CPU-only site sustains **~5 cameras at 2 FPS per camera** in inference terms, and
a single stream runs at 8.6 fps. The documented CPU fallback profile (2 streams × 2 FPS — README, `--cpu`) is
comfortably inside this capacity and is now measured rather than asserted. Real deployments must also fit decode,
tracking and previews into the same cores; treat 5 cameras as the inference-only upper bound, 2-3 as the safe
count with the rest of the pipeline sharing the CPU.

## 4. Tracking / association placement (measured; no code change)

Confirmed placement: tracker update, head-ROI association, classifier batching, FSM and event submission all run
**sequentially on the single inference thread** with GPU inference
(`backend/app/workers/pipeline_worker.py`, `_inference_loop` / `_track_assess`).

Measured cost per frame (`docs/benchmarks/results/cams10_fps5.json`, 10×5 baseline):

| stage | p50 | p95 |
|---|---|---|
| tracking | 0.64 ms | 2.06 ms |
| association | 0.12 ms | 0.31 ms |
| combined | 0.76 ms (3.8% of the 20 ms per-frame budget at 50 fps aggregate) | 2.37 ms (11.9%) |

Decision: **no code change.** The median share is below the 5% threshold; the p95 share looks higher but the
total p95 (44.8 ms vs 19.8 ms p50) shows the tail is batch stalls, not tracking. Moving tracking off the GPU
thread is invasive: the tracker holds per-camera sequential state consumed by the FSM on the same thread, and the
classifier is batched across the whole inference batch — that batching was itself a measured +38% saturation win
(`docs/benchmarks/README.md` §3), and threading would either break it or require per-camera cross-thread frame
ordering. Recommendation: revisit only for saturation-bound deployments after the person stage is batched/trimmed
or engines adopted; measure again at the saturation profile before touching the loop.

## 5. UDP transport and Linux/systemd runbook (verified / docs added)

Verified at runtime on this host:

- `validate_stream_uri` accepts `rtsp://` and `rtsps://`; a **literal `udp://` URI is rejected** ("unsupported
  scheme 'udp'; allowed: rtsp, rtsps, file") — by design, UDP is a transport knob, not a URI scheme.
- UDP RTSP transport is the per-camera `transport` field (`"tcp" | "udp"`, `CameraSeed`/`CameraUpdate` in
  `backend/app/config/settings.py` and `backend/app/schemas/requests.py`), passed to PyAV as the
  `rtsp_transport` open option in `backend/app/camera/source.py`. A camera can be switched to UDP via the Cameras
  API/page without touching `validate_stream_uri`.

Docs added: [docs/operations/start-stop-restart.md](../operations/start-stop-restart.md) now has a Linux + systemd
section (unit from [deployments/linux/README.md](../../deployments/linux/README.md)) and a UDP transport note.

## 6. Not measured

- **Person-pose stage TensorRT engine** (`yolo26s-pose.pt`) and a full-pipeline saturation run on engines — out of
  the requested scope (`ppe_yolo26s.pt` + classifier only); this is the step that could raise the ~72.6 FPS ceiling.
- **TRT engine accuracy** (mAP vs PyTorch FP16) — latency only, no eval run on the engines.
- **Real RTSP-over-UDP camera** — the demo simulator publishes over TCP; transport=udp is code-verified and
  documented, not exercised against a live UDP source on this host.
- **Full application on `device=cpu`** (`start_demo.py --cpu` end-to-end) — the CPU profile covers the inference
  cascade only.
- **Multi-resolution sources** — all pipeline benchmarks use 720p; the NVDEC probe used a local 1080p file, not
  RTSP.
- **8 h soak** — completed after this document was written: clean run ([soak_8h.md](soak_8h.md), 2874 samples,
  0 errors: 50.2/50 FPS mean, 10/10 cameras online throughout, 0 worker restarts / 0 reconnects, worker RSS drift
  +0.9 MB/h, VRAM -18.7 MB/h, worst per-camera uptime 0.9998; teacher disabled during the run).
