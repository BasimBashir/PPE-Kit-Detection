# Runbook: High latency / low processed FPS

## Symptoms
`HighFrameAgeAtInference`, `LowProcessedFps`; System health frame-age chart rising; dropped-frame counters rising fast.

## Diagnose
1. System health → *Inference stages*: which stage dominates (person / ppe / classifier / tracking / association)?
2. GPU util near 100 % → GPU bound; worker CPU near (cores × 100 %) → decode bound.
3. Check other GPU consumers (`nvidia-smi`): the benchmark numbers assume an otherwise idle GPU.

## Recover
Lower `pipeline.target_fps` or the number of cameras, disable the classifier (`association.use_classifier: false`),
or use a smaller person model (`yolo26n-pose.pt`). Hardware decode (NVDEC) is not implemented yet (`ingest.hwaccel` is reserved, Phase 9).

## Verify
Frame age at inference p95 < 0.5 s and processed FPS ≈ target in System health.
