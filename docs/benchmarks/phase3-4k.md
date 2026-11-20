# Phase 3: 4K streams - measurements, fixes, and deployment sizing (10-50 cameras)

All measured numbers come from `scripts/benchmark/run_benchmark.py` (full pipeline, JSON in `results/p3_*.json`)
and `scripts/benchmark/decode_probe.py` (`results/decode_4k.json`) on the dev host: **RTX 3090 24 GB, Intel
i5-12400F (6 cores / 12 threads), Windows 11**. Sources: the 10 demo clips transcoded to **3840x2160 H.264 25 fps**
(`fetch_demo_videos.py --height 2160`; 6 native 4K, 4 upscaled from 1080p/720p - same decode load), served over
RTSP/TCP by MediaMTX. Every Phase 3 run has **all five PPE rules active** (construction preset) and both detectors
(helmet stack + items detector yolo26m @ 1280).

## Why the cameras stream 25 fps but the tests infer at 5 fps
Yes - industry cameras stream 25-30 fps, and this app **decodes every frame at that rate** (H.264 requires it; see
the `decoded` column in the tables below). *Inference* runs on the newest frame `target_fps` times per second per
camera - a per-camera or global setting (`pipeline.target_fps`, up to 30; set 25 and every frame is inferred).
The tests measure 5 fps because it is the default, not a limit: PPE state changes slowly, frames 40 ms apart are
near-duplicates, and the alert rule is "12 missing of the last 15 observations" (~2.2 s at 5 fps, ADR-006).
Running inference at the full 25 fps is **supported today** - it costs ~5x the compute for near-duplicate
evidence and buys lower alert latency - and section 3 sizes exactly that. If you move to 25 fps, re-tune the
temporal policy (e.g. confirm 60/75 keeps ~2.4 s of evidence at 25 fps instead of 0.5 s at 12/15).

## 1. What we measured

| run | cams x fps | processed fps | decoded fps / cam | frame age p95 | host CPU | GPU | note |
|---|---|---|---|---|---|---|---|
| p3_uhd_cams1_fps5 | 1 x 5 | 5.0 / 5 | 25.5 | 0.14 s | 33 % | 40 % | |
| p3_uhd_cams4_fps5 | 4 x 5 | 19.8 / 20 | 24.5-25.5 | 0.14 s | 56 % | 29 % | real time, CPU decode |
| p3_uhd_cams10_fps2 | 10 x 2 | 17.1 / 20 | 24-25.5 | 0.29 s | 86 % (max 100) | 11 % | CPU-bound (decode) |
| p3_uhd_cams10_fps5 (before fixes) | 10 x 5 | 12.6 / 50 | 24-25.5 | 0.14 s | 92 % | 8 % | CPU-bound: 10 x 4K25 software decode |
| p3_uhd_cams10_fps5_nvdec (after fixes) | 10 x 5 | **20.7 / 50** | 20-23.5 | 0.14 s | 57 % | 31 % | NVDEC saturated (~215 fps); one inference worker ~20 fps |

Decode capacity alone (`decode_probe.py`, 4K H.264 25 fps, every 3rd frame converted to BGR, no inference):

| decoder | 1 stream | 4 streams | 10 streams | host CPU at 10 |
|---|---|---|---|---|
| CPU (PyAV/FFmpeg, 12 threads) | 226 fps | 276 fps | 270 fps total (24.7 / stream) | 100 % |
| NVDEC (PyAV `hwaccel=cuda`, RTX 3090) | 190 fps | 207 fps | 194 fps total (18.2 / stream) | 46 % |

**Initial plan -> what failed -> what replaced it**
1. *Plan:* run 4K exactly like 720p. *Failed at 10 cameras:* 10 x 25 fps x 4K = 250 decoded fps needs the whole
   CPU (ceiling 270 fps) before any inference; GPU idle at 8 %.
2. *Fix 1 - NVDEC* (`ingest.hwaccel: cuda`, now wired in `camera/source.py`, software fallback): CPU 92 % -> 57 %.
   Ceiling moves to the decoder engine (~200-215 4K fps on one RTX 3090), so cameras decode ~21-23 of 25 fps.
3. *Fix 2 - downscale once* (`UltralyticsCascadeDetector._fit`): Ultralytics letterboxed the full 4K frame
   separately for each of the 3 models; now one resize to the largest model input (1280), boxes/keypoints mapped
   back, crops/evidence stay full resolution. Models see the same pixels.
4. *Fix 3 - preview after shrink:* privacy-masking and copying a full 4K frame 4x/s per camera for the live
   preview moved to the 640 px preview image.
5. *Rejected:* converting frames at exactly the inference rate (`convert_fps_factor 1`) - aliases with the
   scheduler and leaves only ~2 fps per camera. Kept 2x.
6. *Remaining limit:* one inference worker process handles **~20 processed 4K frames/s** (3 models, ~50 ms of
   per-frame work on one Python thread) while the GPU is at ~31 %. 10 cameras x 5 fps needs 50.

**What 4K buys:** the detectors see at most 1280 px (whole frame), so 4K mainly gives full-resolution crops
(head classifier, teacher, evidence) today. For gloves/goggles on distant workers the stronger lever is a
person-crop second stage (detect items on full-res person crops) or close-angle cameras - see
[phase3-log.md](../ml/phase3-log.md).

## 2. Deployment sizing: 10-50 4K cameras, processed at 5 fps each

**Unit costs (measured on the dev host):** NVDEC ~210 4K H.264 frames/s per RTX 3090; one inference worker
~20 processed 4K frames/s at ~31 % GPU, ~7 CPU threads (incl. BGR conversion of 10 streams), ~4-6.6 GB RAM,
~3.3 GB VRAM + ~0.7 GB per NVDEC 4K stream; 4K H.264 bitrate 7-27 Mbit/s per camera (our clips).

**Estimate rules (NOT measured - validate on the target hardware):** up to 3 workers per 24 GB GPU (31 % each);
0.5 CPU thread per processed fps; workers = processed fps / 20; GPUs = max(workers / 3, decoded fps / 210);
camera bitrate 16 Mbit/s for planning.

**Recommended camera configuration:** 4K main stream, H.264 (H.265 halves bandwidth; NVDEC decodes both),
**10-12 fps**, GOP 1-2 s. Decode cost follows the *camera* fps, inference cost follows `target_fps`; 25-30 fps
cameras cost 2.5-3x the decode for no accuracy gain at 5 fps inference.

| cameras | processed fps | workers | GPUs (24 GB, 3090/4090/L40S class) - cameras at 10 fps | GPUs - cameras at 25 fps | CPU threads | RAM | network (ingress) |
|---|---|---|---|---|---|---|---|
| 10 | 50 | 3 | 1 | 2 (NVDEC-bound) | 24-32 (e.g. 16-core) | 32 GB | ~160 Mbit/s |
| 20 | 100 | 5 | 2 | 3 | 48-64 (e.g. 32-core) | 64 GB | ~320 Mbit/s |
| 30 | 150 | 8 | 3 | 4 | 75-96 (e.g. 2 x 24-core) | 96 GB | ~480 Mbit/s |
| 40 | 200 | 10 | 4 | 5 | 100-128 | 128 GB | ~640 Mbit/s (10 GbE) |
| 50 | 250 | 13 | 5 | 6 | 125-160 | 160 GB | ~800 Mbit/s (10 GbE) |

Rule of thumb: **one 24 GB RTX-class GPU + ~16 CPU cores per 10 4K cameras at 5 fps**, with cameras set to
10-12 fps. Measured TensorRT FP16 speed-ups (Phase 2: 1.25-1.35x at production batches) would lower the worker
count by about a quarter. Data-centre GPUs with several NVDEC engines (e.g. L4, L40S) raise the decode ceiling per
card - not measured here. At 2 fps inference (10 x 4K x 2 fps = 20 fps) a single worker is enough per 10 cameras.

**Software prerequisite for > ~4 x 4K @ 5 fps per deployment:** one deployment runs **one** inference worker
(~20 processed 4K fps). Scale out with **network sharding** (section 4) or lower `target_fps`.

## 3. Full-rate inference: 4K @ 25 fps analysed per camera (hypothetical, NOT measured)

You asked for the full-25 fps numbers. Everything below is **extrapolated from the measured unit costs in section 2**
- no 25 fps run was executed on this host. Treat as sizing input, validate on target hardware.

**Formulas** (measured units in parentheses):
- inference workers = 25 x cameras / 18 (one worker ≈ 18-20 processed 4K fps, measured 20.7)
- GPUs (24 GB) = max(workers / 6, 25 x cameras / 210) (VRAM ≈ 3.3 GB per worker + ~0.7 GB per NVDEC stream;
  NVDEC ≈ 210 4K fps per GPU, measured 194-207)
- CPU threads ≈ 0.3 x processed fps + 8 (measured: ~7 of 12 threads at 20.7 processed fps incl. decode/convert)
- RAM ≈ 6 GB x workers + 16 GB OS/PostgreSQL (worker RSS measured 4-6.6 GB at 10 cameras)
- network = 16 Mbit/s per camera (planning figure; measured clips 7-27 Mbit/s H.264)

| cameras | processed fps | workers | GPUs (24 GB) | CPU threads (cores) | RAM | network ingress |
|---|---|---|---|---|---|---|
| 10 | 250 | 14 | 3 | ~83 (e.g. 2 x 24-core) | ~100 GB | ~160 Mbit/s |
| 20 | 500 | 28 | 5 | ~158 (e.g. 4 x 24-core) | ~184 GB | ~320 Mbit/s (10 GbE) |
| 30 | 750 | 42 | 7 | ~233 | ~268 GB | ~480 Mbit/s (10 GbE) |
| 40 | 1000 | 56 | 10 | ~308 | ~352 GB | ~640 Mbit/s (10 GbE) |
| 50 | 1250 | 70 | 12 | ~383 | ~436 GB | ~800 Mbit/s (10 GbE) |

Compare section 2: the same 50 cameras at the recommended 5 fps need **5 GPUs** instead of 12 and ~1/5 of the CPU -
the price of inferring every 40 ms frame is ~2.4x the hardware here (5x inference vs decode-only, partly offset by
batching efficiency). Measured TensorRT FP16 gains (Phase 2: 1.25-1.35x at production batches) would cut the worker
count ~25% (50 cams: 70 -> ~52 workers, 12 -> ~9 GPUs). Data-centre GPUs (L4/L40S: multiple NVDEC engines + more
VRAM) change both terms - re-measure before buying.

**Accuracy note:** at 25 fps inference the per-frame evidence is nearly identical to 5 fps (same people, same PPE
state); the gain is alert latency (confirm 12/15 lands in ~0.5 s instead of ~2.2 s) - re-tune the temporal policy
(counts scale with fps) or you trade latency for false alarms.

## 4. Network sharding (on/off)

`PPE_WORKER_SHARD=i/N` (env, off when unset; `.env.example`). A deployment with the shard set handles only the
cameras whose sorted id index mod N == i: `0/2` takes cam01, cam03, ...; `1/2` takes cam02, cam04, ... One app
instance per machine, **all against the same PostgreSQL** - events, dedupe re-seeding, spool and the outbox are
already per-camera and idempotent, so nodes can share the database unchanged; run the API on each node (its
dashboard shows that node's cameras; every event from every node is in the shared DB).

- `0/4` on node A, `1/4` on B, `2/4` on C, `3/4` on D = 4 nodes, one GPU each → ~72 processed 4K fps total.
- Adding capacity = adding a node with the next shard index and restarting with a larger N (assignment is
  deterministic; in-flight episodes end cleanly, the same alert does not re-fire thanks to DB dedupe seeding).
- Not needed below ~4 x 4K @ 5 fps per GPU; leave unset (off).
- In-process multi-worker (several inference workers inside ONE app instance) is deliberately not implemented:
  it hits the same GPU and NVDEC ceilings as sharding while duplicating the telemetry/live-state layer.
