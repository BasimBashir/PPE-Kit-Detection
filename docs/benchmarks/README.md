# Benchmarks - analysis

Raw, generated reports (never hand-edited): [results.md](results.md) + `results/*.json` (load and ablation),
[fault_injection.md](fault_injection.md) + `.json`, [soak_1h.md](soak_1h.md) + `.csv`, [ui_latency.md](ui_latency.md).
Earlier run kept for the before/after comparison: [results_before_classifier_batching/](results_before_classifier_batching/results.md).

**Host:** RTX 3090 24 GB, Intel i5-12400F (6C/12T), 64 GB RAM, Windows 11, torch 2.14.1+cu130, Ultralytics 8.4.198.
**Sources:** 10 simulated RTSP cameras (720p H.264, 24-30 FPS, looped stock clips via MediaMTX + Toxiproxy, TCP).
These are desktop-GPU results; they are **not** edge-device results.

## 1. Baseline profile (10 cameras × 5 processed FPS) - meets target
| Metric | Value | Boundary |
|---|---|---|
| Aggregate processed FPS | 50.2 / 50 | detection + tracking + association completed |
| Frame age at inference, worst camera p95 | 0.126 s | decode → inference start (host clock) |
| Person / PPE stage p95 per frame | 12.1 / 12.0 ms | batch time ÷ batch size |
| Association p95 per frame | 0.31 ms | head ROI + matching (classifier batched separately) |
| GPU utilisation mean / max | 37 % / 69 % | NVML |
| VRAM | 2.7 GB | whole device |
| Worker RSS | 2.8 GB, +23 MB over 60 s | process |
| Host CPU | 48 % | all 12 threads (decode of ~280 FPS + Python pipeline) |

## 2. Sampling rate trade-off (spec 5.1, NFR-PERF-002)
| Processed FPS / camera | Aggregate achieved | Frame age p95 | Min. time to confirm (12 votes) | Comment |
|---|---|---|---|---|
| 2 | 20.1 / 20 | 0.27 s | 5.5 s | lowest load; slow alerts |
| 5 (baseline) | 50.2 / 50 | 0.13 s | 2.2 s | chosen: ≈ interview's 10-15 samples within ~2-3 s |
| 10 | 81.5 / 100 | 0.08 s | 1.1 s | **not sustained** for 10 cameras on this host |

Frame age is *lower* at higher FPS because a frame is taken sooner after decode; the cost is throughput.

## 3. Capacity and the bottleneck
Saturation run (10 × 30 FPS requested): **72.6 processed FPS** aggregate while the GPU averaged only **37 %**.
The ceiling is the single Python inference thread plus CPU decode, not the GPU.

**Optimisation applied and measured:** the head classifier was first called once per frame (≈12 ms per call, ten small
GPU calls per batch). Batching all ambiguous crops of an inference batch into one call:

| Profile | Before | After |
|---|---|---|
| 10 × 30 (saturation) | 52.5 FPS | **72.6 FPS (+38 %)** |
| 10 × 10 | 58.7 FPS | **81.5 FPS (+39 %)** |
| association p95 per frame | 12-14 ms | 0.3 ms |

Next steps, not implemented: TensorRT FP16 engines, NVDEC decode, moving tracking/association off the GPU thread,
DeepStream for the edge.

## 4. Architecture ablation (10 × 5) - is every model justified?
| Variant | Person model | Head ROI | Classifier | Uncertain obs. / frame | Events / cam-h* |
|---|---|---|---|---|---|
| A | COCO yolo26s | geometry | off | 1.20 | 108 |
| B | yolo26s-pose | geometry | off | 0.72 | 108 |
| C | yolo26s-pose | keypoints | off | 1.12 | 138 |
| D (deployed) | yolo26s-pose | keypoints | on (cascade) | **0.35** | 126 |

Reading: the classifier cascade resolves ~70 % of the cases the geometric association cannot decide (fewer abstentions,
i.e. more people actually assessed), at a cost of ~0.3 ms/frame after batching. Keypoint ROIs alone produce more
`UNCERTAIN` than geometry because the keypoint box is tighter (more helmets fail the IoA gate) - they pay off together
with the classifier. Without labelled episodes this shows *coverage*, not *accuracy*; per-variant precision/recall
needs ground truth (see limitations).

\*Events per camera-hour are dominated by the **looping** clips: every 7-33 s the same workers reappear at their start
positions as new tracks, which defeats spatial deduplication. Real cameras do not loop; this number is not an operational
alert rate. Compliant cameras (cam04, cam07, cam08, cam10) produced no events across the runs reviewed.

## 5. Resilience (live fault injection) - summary
All 13 fault scenarios recovered without restarting the application; unaffected cameras stayed at ≈5 FPS (min 4.4-5.0)
during single-camera faults. Highlights: camera unplug detected 0.6 s / recovered 7.8 s after replug; worker crash
detected 0.6 s / all cameras back 14.7 s; DB outage: 2 events spooled and replayed, 0 lost, pipeline unaffected. Full
table and the bugs these runs exposed: [fault_injection.md](fault_injection.md), [test-report](../testing/test-report.md).

## 6. Observed missed violations (qualitative, demo footage)
Cumulative events per camera over all runs on 2026-11-02: cam09 garment floor 876, cam05 assembly line 340, cam02 mixed
site 15, **every other camera 0**. Compliant scenes (cam04, cam07, cam08, cam10) correctly produced no alerts, but two
scenes chosen *because* workers have no hard hats also produced none:
- **cam06 metal workshop - caps and head wraps judged COMPLIANT** (visible as green "OK" boxes on the camera wall). The
  training set has no non-hard-hat headwear labelled as `head`; the helmet detector / classifier generalise "something on
  the head" to helmet. This is a **false-compliance failure mode** - the exact hard-negative gap spec 12.1 warns about.
- **cam03 masonry wall - distant, small, often occluded workers** rarely accumulate 12 decisive votes (abstention by design).

Remedy (not done): add hard negatives (caps, hoods, turbans, scarves, hair) labelled as `head` and helmets in hand / on
the ground, fine-tune, and evaluate event-level recall per scene on labelled site footage. This is also why no recall
claim is made, and why "5,000 images ⇒ 80 % recall" cannot be assumed.

## 7. What is not measured
Event-level recall and false alarms per camera-hour on labelled footage; edge devices; 8 h / 24 h soak; multi-resolution
sources (all 720p); UDP transport.
