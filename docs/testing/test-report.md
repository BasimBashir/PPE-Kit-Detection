# Test report

**Run date:** 2026-11-23 (Phase 4; Phase 3 and Phase 2 details below) · **Host:** Windows 11, i5-12400F, RTX 3090, Python 3.12.10, Node 24.16 ·
**Commit state:** local runs; GitHub Actions CI re-runs backend tests (minus the GPU test), ruff, and the frontend type-check/tests/build on every push to `main` (promtool, alembic check and the live benchmarks are local only).

## Results (exact commands, actual output)
| Suite | Command | Result |
|---|---|---|
| Backend unit + integration | `cd backend && uv run pytest tests -q -m "not gpu"` (Docker services up) | **148 passed** in 22 s (Phase 4 adds fall/stalling/phone assessor tests, schedule gating, behaviour worker wiring) |
| Backend GPU (real frames through the cascade incl. the Phase 3 items stage) | `uv run pytest tests -q -m gpu` | **1 passed** |
| Backend lint | `uv run ruff check backend/app backend/tests scripts` | clean |
| Frontend types | `cd frontend && npx tsc --noEmit -p .` | clean |
| Frontend unit | `npx vitest run` | **17 passed** |
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
| teacher fusion/verdict policy incl. never-suppress over the full grid | `test_teacher.py` (11 tests) |
| teacher providers (Gemini Decisions / SystemOne wire formats, circuit breaker, rate limiter) | `test_teacher.py` |
| restart re-seed of cooldown/dedupe state (incl. DB-down fallback) | `integration/test_persistence.py` (3 tests) |
| teacher review end-to-end (escalation visible; false-alert tag issues no UPDATE/DELETE) | `integration/test_teacher_review.py` (3 tests) |
| camera-source ingest loop (flow, backoff/offline/recovery, watchdog) | `test_camera_source.py` (3 tests, local files) |
| Users admin page (create, self-lockout guard) | `app.test.tsx` (frontend) |
| Phase 3: body-region association (bare hand beats gloved, ground vest not credited, no wrist keypoints = UNCERTAIN, feet cut off = UNCERTAIN, other items ignored) | `test_multi_ppe.py` (9 tests) |
| Phase 3: config-only new class end to end (rule + association + teacher question) | `test_multi_ppe.py::test_new_class_is_config_only` |
| Phase 3: network sharding parse + deterministic camera split | `test_multi_ppe.py::test_shard_parsing_and_camera_split` |
| Phase 3: site presets validate | `test_multi_ppe.py::test_site_presets_validate` |
| Phase 3: zone polygons are resolution-independent (SD/FHD/4K foot-point + mask fraction) | `test_ppe_logic.py::test_zone_polygon_is_resolution_independent` |
| Phase 4: fall = fast transition only (fast fall confirms after hold, getting up ends it, slow lying / bending / no pose / gated person never fire) | `test_behaviour.py::test_fast_fall_then_lying_is_absent_and_confirms_after_hold`, `test_lying_down_slowly_never_starts_an_episode`, `test_bending_over_is_not_a_fall`, `test_no_pose_is_uncertain_never_a_fall`, `test_small_or_low_confidence_person_is_not_assessed` |
| Phase 4: stalling timer (warming up, still = absent, working hands = moving, confirms after the rule's idle time) | `test_behaviour.py::test_still_person_is_absent_after_warmup`, `test_working_hands_count_as_moving`, `test_stalling_confirms_after_the_rules_idle_time` |
| Phase 4: phone evidence over hand/head ROI, dropout bridging, far/low-conf phone = normal | `test_behaviour.py::test_phone_in_hand_is_absent_with_the_phone_as_evidence`, `test_short_detector_dropouts_are_bridged`, `test_phone_far_from_the_body_or_low_confidence_is_normal` |
| Phase 4: behaviour config validation, schedule gating (off shift = no observations), worker wiring | `test_behaviour_config.py`, `test_behaviour_worker.py` |
| Phase 3: secondary rule records the event with NO outbox row (no alert delivery) | `integration/test_persistence.py::test_secondary_rule_records_event_without_alert_delivery` |
| Phase 3: preset API flow (create, apply idempotent, 422s, delete) | `integration/test_api.py::test_operator_presets_create_apply_and_validate` |

## Integration (spec 19.2)
Video → detection pipeline on real frames (GPU test), DB persistence, outbox retry/backoff/FAILED, REST camera/event
workflows, WebSocket snapshot + live event + render ack, evidence snapshot creation and download, spool replay after DB
outage, restart persistence (migrations + seed idempotency), teacher review against a fake provider (escalation via
the normal event path; LIKELY_FALSE_ALERT verified to issue no UPDATE/DELETE against the original event). RTSP ingest
is exercised end-to-end by the live fault and load campaigns plus the file-based camera-source unit tests (a live
RTSP pytest remains a gap).

## Live smoke test (2026-11-20, Phase 3)
Full app on 10 simulated 4K@25fps cameras: startup seed reports the two presets; `POST /api/v1/presets/construction/apply`
creates 5 rules (3 mandatory, 2 secondary) and the running worker hot-reloads them ("rules reloaded: 6 enabled rules");
dual-detector inference on 3840x2160 streams produces OPEN hard-hat events with evidence and the live overlay carries
per-track `ppe_class` + `item_roi`; the gloves rule correctly stays UNKNOWN (no wrist keypoints on wide factory clips -
never a false alert). Vest/shoes/goggles confirmations did not fire on this stock footage (footage limitation;
accuracy gap documented in docs/ml/phase3-log.md).

## Live smoke test (2026-11-18, Phase 2)
Complete app started with 10/10 simulated cameras online: teacher review verified with the deterministic fake
provider (all four verdicts stored; one compliant miss escalated as a visible OPEN event, reason "raised by teacher
review", fused 0.824) and then with live Gemini Decisions on the stock demo streams (~80 reviews, zero failures once
the timeout default was raised 5 s → 30 s — live p95 ≈ 8 s; fix shipped in the default config). Teacher disabled for
normal runs afterwards.

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
Frontend end-to-end browser test (Playwright) not implemented; the Phase 3 zone drawer and presets card were verified by tsc + vitest + API-level integration tests, not a scripted browser run. The dashboard was exercised through its API and
WebSocket contracts (integration tests) and unit tests; visual verification status is recorded in BUILD_STATUS.md.
No labelled site footage → no system-level recall / false-alarm rate. CPU-only profile not tested.
