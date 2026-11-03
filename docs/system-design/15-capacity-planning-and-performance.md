# 15 - Capacity planning and performance

**Status:** VERIFIED on RTX 3090 / PROPOSED for edge · **Related ADRs:** ADR-003, ADR-011 · **Modules:** `scripts/benchmark/*`

## Purpose and scope
Workload model and measured capacity (spec 11, interview gap #11).

## Assumptions
- 720p H.264 sources; numbers are for the stated host only.

## Diagram

## Workload model
| Parameter | Baseline |
|---|---|
| Cameras | 10 |
| Source | 720p, 24-30 FPS H.264 |
| Processed (inference) FPS per camera | 5 (2 and 10 benchmarked) |
| Aggregate inference load | 50 frames/s (two detector stages each) |
| Decode load | all source frames (~280 frames/s) on CPU |

## Measured results
Generated tables: [benchmarks/results.md](../benchmarks/results.md) (throughput, stage latency, GPU/CPU/RSS, events) and
[benchmarks/soak_1h.md](../benchmarks/soak_1h.md) (drift). Fault recovery: [benchmarks/fault_injection.md](../benchmarks/fault_injection.md).

## Interpretation rules
FPS is reported per boundary (source, decoded, processed); "processed" means detection + tracking + association done.
Stage latency is per frame (batch time / batch size). No edge-device claim is made.

## Alternatives considered
TensorRT/ONNX export, NVDEC decode, DeepStream - Phase 9 options, only after these baselines.
