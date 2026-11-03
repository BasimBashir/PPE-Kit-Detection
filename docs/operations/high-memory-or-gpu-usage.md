# Runbook: High memory or GPU usage / GPU OOM

## Symptoms
`GpuMemoryHigh`, `InferenceErrors`, worker restarts after repeated OOM, RSS trend rising in System health.

## Diagnose
1. System health: worker RSS, VRAM, torch allocated. Compare with `docs/benchmarks/soak_1h.md` baselines.
2. Logs: `GPU OOM during inference` lines (worker), restart count.

## Recover
The worker empties the CUDA cache after an OOM and restarts itself after 3 OOMs in 60 s. Persistent pressure: reduce
`pipeline.max_batch`, `imgsz`, camera count or model size; ensure no other process is using the GPU.

## Verify
VRAM stable, no new inference errors for 15 minutes.
