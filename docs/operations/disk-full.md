# Runbook: Disk full / low disk

## Symptoms
`DiskFreeLow`; events with `evidence_error = disk_low | write_failed`; `ppe_evidence_failures_total` rising.

## What keeps working
Events are still persisted (without image); the pipeline never stops for disk errors.

## Recover
Lower retention (Rules & settings → Evidence retention) and **Run retention now**, or free space on the evidence volume.
`evidence.min_free_bytes` (default 1 GB) is the guard threshold.

## Verify
New events have evidence again; disk free above threshold.
