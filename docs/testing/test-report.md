# Test report

**Run date:** 2026-11-02 · **Host:** Windows 11, i5-12400F, RTX 3090, Python 3.12.10, Node 24.16 ·
**Commit state:** local runs before the first commit; GitHub Actions CI re-runs backend tests (minus the GPU test), ruff, and the frontend type-check/tests/build on every push to `main` (promtool, alembic check and the live benchmarks are local only).

## Results (exact commands, actual output)
| Suite | Command | Result |
|---|---|---|
| Backend unit + integration + GPU | `cd backend && uv run pytest tests -q` (Docker services up) | **77 passed** in 34.0 s |
| Backend lint | `uv run ruff check backend/app backend/tests scripts` | clean |
| Frontend types | `cd frontend && npx tsc --noEmit -p .` | clean |
| Frontend unit | `npx vitest run` | **14 passed** |
| Frontend build | `npm run build` | built (one >500 kB chunk warning from recharts; served locally) |
| Prometheus rules | `promtool check rules deployments/docker/alert-rules.yml` | SUCCESS: 10 rules |
| Migrations | `alembic check` | no drift between models and migration |
| Fault injection (live system) | `python scripts/benchmark/fault_injection.py` | 13 scenarios recovered - [results](../benchmarks/fault_injection.md) |
| Load (live pipeline) | `python scripts/benchmark/run_benchmark.py --all` | 10 profiles - [results](../benchmarks/results.md) |
| Soak | `python scripts/benchmark/soak.py --hours 1` | [results](../benchmarks/soak_1h.md) |

## Coverage against spec 19.1 (unit)
| Spec item | Test |
|---|---|
| bounding-box validation, IoU/intersection, polygon | `test_ppe_logic.py::test_geometry_basics`, `test_zones` |
| head ROI geometry (both modes, fallback, clipping) | `test_geometry_roi`, `test_keypoint_roi_and_fallback` |
| helmet-person association incl. ground / hand / neighbour / occlusion | `test_helmet_on_ground_not_credited`, `test_helmet_in_hand_at_chest_not_credited`, `test_one_helmet_never_credited_to_two_people`, `test_occluded_or_tiny_or_truncated_is_uncertain` |
| confidence threshold boundaries | `test_threshold_boundaries` |
| classifier cascade (only ambiguous; batched path ≡ direct) | `test_conflict_goes_to_classifier_only_when_ambiguous`, `test_deferred_classification_matches_direct_path` |
| temporal voting, PPE transitions, hysteresis, track expiry, duration/both modes | `test_confirm_exactly_at_n_of_m`, `test_single_miss_does_not_flap_and_uncertain_never_votes`, `test_clear_with_hysteresis_and_new_episode`, `test_track_expiry`, `test_duration_mode`, `test_both_mode_needs_count_and_duration` |
| alert idempotency / dedup / cooldown / re-arm | `integration/test_persistence.py::test_one_episode_one_event...`, `test_cooldown_reopens...`, `test_spatial_dedupe...` |
| event state transitions | `test_event_lifecycle_filters_outbox_and_render_ack`, `test_api.py::test_event_flow...` |
| camera health transitions, stall-by-trickle, reconnect backoff | `test_ingest_logic.py` (7 tests) |
| slot drops, scheduler fairness and pacing | `test_slot_drops_and_depth`, `test_fairness...`, `test_fps_pacing_and_no_burst_after_stall` |
| config validation | `test_policy_validation`, `test_api.py::test_rules_validation...` |
| credential redaction | `test_redaction_and_uri.py` (20 tests incl. PyAV 401 message, logging filter) |

## Integration (spec 19.2)
Video → detection pipeline on real frames (GPU test), DB persistence, outbox retry/backoff/FAILED, REST camera/event
workflows, WebSocket snapshot + live event + render ack, evidence snapshot creation and download, spool replay after DB
outage, restart persistence (migrations + seed idempotency). RTSP ingest is exercised end-to-end by the live fault and
load campaigns rather than a dedicated pytest (gap noted).

## Bugs found by the tests and fixed (selection)
| Found by | Defect | Fix |
|---|---|---|
| unit | SUSPECTED ↔ UNKNOWN flapping (no-helmet run survived UNCERTAIN frames) | entering SUSPECTED requires a current NO_HELMET; UNKNOWN only on an UNCERTAIN observation |
| unit | scheduler: never-served cameras sorted like cameras served at t=0; catch-up burst after stalls | `-inf` default; reschedule from now when late |
| integration | `ON CONFLICT DO NOTHING` reported success via `rowcount` → orphan outbox insert | `RETURNING id` decides novelty |
| integration | `notify or default` replaced an empty-list recorder | explicit `is not None` |
| live fault injection | 404 from the NVR treated as fatal → 29 s recovery | 404 transient; backoff cap 10 s → 4.8 s |
| live fault injection | `InvalidDataError` (a `ValueError`) shown as ERROR | explicit config-error type |
| live fault injection | stale telemetry resurrected cameras as ONLINE after a worker crash | ignore telemetry while worker down |
| live fault injection | DB outage slowed inference to 2.6 FPS and the API to 9-12 s/request | event-service circuit breaker, throttled updates, token cache, DB-free metrics |
| live fault injection | a trickle of straggler frames kept a stalled stream DEGRADED | decode rate < 10 % nominal → OFFLINE |
| live run | alert duplicates from ID switches (person IoU only 0.2-0.3) | head-position spatial dedupe |
| benchmark | per-frame classifier calls capped throughput at ~53 FPS | one batched call per inference batch → 72.6 FPS |

## Not covered / known gaps
Frontend end-to-end browser test (Playwright) not implemented. The dashboard was exercised through its API and
WebSocket contracts (integration tests) and unit tests; visual verification status is recorded in BUILD_STATUS.md.
No labelled site footage → no system-level recall / false-alarm rate. CPU-only profile not tested.
