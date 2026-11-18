# Linux deployment profile (PROPOSED - not benchmarked)

The verified deployment is the Windows 11 + RTX 3090 development host. This profile documents how the same code runs on a
Linux x86-64 box with an NVIDIA GPU; numbers from the dev host do not transfer.

1. NVIDIA driver + CUDA runtime compatible with the torch wheel in `backend/pyproject.toml` (cu130 index; use the cu12x
   index on older drivers), Docker + compose plugin, ffmpeg, uv, Node 24 (build only).
2. `python3 scripts/start_demo.py --no-start` (or the manual steps it prints).
3. Install the service unit below (adjust paths/user), then `systemctl enable --now ppe-intelligence`.
4. Put a TLS reverse proxy (nginx/caddy) in front if the dashboard must be reachable from the LAN (API binds 127.0.0.1).

For real cameras set `mode: live` in `configs/app.yaml` (no simulator, no demo controls, fault API disabled).

## GPU deployment items: TensorRT FP16 and NVDEC (Linux/Docker targets)

Both were probed on the Windows dev host (RTX 3090, TensorRT 11.5) and are measured there — see
[docs/benchmarks/phase2-performance.md](../../docs/benchmarks/phase2-performance.md). They are deployment
optimisations for a Linux/GPU host, not Windows requirements; the PyTorch FP16 stack is the supported default.

**TensorRT FP16 engines** (optional; measured 2.4× per-call latency at batch 1, 1.25-1.35× at batch 10 for the
PPE detector; never add these packages to `backend/pyproject.toml` — experiment venv only):

```bash
uv pip install tensorrt "nvidia-modelopt[onnx]" onnx onnxslim   # TRT 11 needs ModelOpt to bake FP16 into ONNX
python -c "from ultralytics import YOLO; \
  YOLO('models/ppe_yolo26s.pt').export(format='engine', half=True, imgsz=640, batch=10, dynamic=True, device=0, workspace=4); \
  YOLO('models/head_cls_yolo26n.pt').export(format='engine', half=True, imgsz=96, batch=32, dynamic=True, device=0, workspace=2)"
```

Store engines outside the repo (they are host/TRT-version specific). Wiring engines into the pipeline is not
implemented — `ModelsConfig.weights` would need to point at `.engine` files and the model-version bookkeeping
(`weights_version`) updated; until then this is a measured feasibility result only.

**NVDEC hardware decode**: probed, **not recommended** — on the dev host CPU FFmpeg software-decodes 1080p h264
faster than NVDEC (945 vs 590 fps decode-only; end-to-end to-BGR equal at ~296 fps), and pip OpenCV's bundled
FFmpeg cannot reach NVDEC at all. If a future Linux deployment needs it (e.g. many 4K sources), use an
NVDEC-enabled FFmpeg build and keep frames on the GPU (`-hwaccel cuda -hwaccel_output_format cuda -c:v h264_cuvid`
+ GPU-side resize); `ingest.hwaccel: cuda` in `configs/app.yaml` is a reserved knob, not an implementation.
