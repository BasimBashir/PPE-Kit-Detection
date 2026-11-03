"""Pipeline worker process: ingest -> schedule -> detect -> track -> associate -> PPE FSM -> events (ADR-001/010).

Threads: one CameraSource per camera, CameraWatchdog, inference (owns GPU, trackers and state machines),
EventService, telemetry (1 Hz). The main thread handles commands and watches the parent API process.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import psutil
from sqlalchemy import select

from app.alerts.event_service import ConfirmedViolation, EventService
from app.alerts.spool import EventSpool
from app.camera.health import CameraHealth
from app.camera.source import CameraRuntimeConfig, CameraSource, CameraWatchdog, FaultHooks
from app.config.settings import (
    AppConfig,
    ModelsConfig,
    RuleSeed,
    Settings,
    get_settings,
    load_app_config,
    load_models_config,
)
from app.core.logging import setup_logging
from app.db.base import make_engine, make_session_factory, session_scope
from app.evidence.store import EvidenceStore
from app.inference.factory import build_classifier, build_detector
from app.inference.types import Detector, HeadClassifier
from app.models import Camera
from app.observability import metrics as m
from app.observability.resources import ResourceSampler
from app.observability.stats import RollingWindow
from app.ppe.association import Associator, apply_classification
from app.ppe.geometry import scale
from app.ppe.state_machine import PPEStateMachine
from app.ppe.types import Outcome, PPEState
from app.ppe.zones import ZoneRuntime, apply_privacy_masks, filter_ignored, zone_of
from app.repositories import rules as rules_repo
from app.repositories.misc import record_health, record_model_versions
from app.streaming.frame import Frame
from app.streaming.scheduler import InferenceScheduler
from app.streaming.slot import LatestFrameSlot
from app.tracking.tracker import PersonTracker, make_tracker

log = logging.getLogger("worker")
S = PPEState


@dataclass
class RuleRuntime:
    id: str
    severity: str
    zone_id: str | None
    associator: Associator
    fsm: PPEStateMachine


@dataclass
class CamRuntime:
    cfg: CameraRuntimeConfig
    name: str
    slot: LatestFrameSlot | None = None
    health: CameraHealth | None = None
    source: CameraSource | None = None
    tracker: PersonTracker | None = None
    hooks: FaultHooks = field(default_factory=FaultHooks)
    zones: list[ZoneRuntime] = field(default_factory=list)
    rules: list[RuleRuntime] = field(default_factory=list)
    processed: RollingWindow = field(default_factory=lambda: RollingWindow(5.0))
    frame_age: RollingWindow = field(default_factory=lambda: RollingWindow(60.0))
    last_preview: float = 0.0
    last_state: str = ""


class PipelineWorker:
    def __init__(self, settings: Settings, app_cfg: AppConfig, models_cfg: ModelsConfig,
                 telemetry: Any, commands: Any, detector: Detector | None = None,
                 classifier: HeadClassifier | None = None, parent_pid: int | None = None):
        self.s, self.cfg, self.models_cfg = settings, app_cfg, models_cfg
        self.telemetry, self.commands, self.parent_pid = telemetry, commands, parent_pid
        self.run_id = uuid.uuid4().hex[:12]
        self.started_at = time.time()
        self.engine = make_engine(settings.database_url)
        self.sf = make_session_factory(self.engine)
        self.detector = detector or build_detector(models_cfg, settings.resolve(settings.ppe_models_dir))
        self.classifier = classifier if detector is not None else build_classifier(
            models_cfg, settings.resolve(settings.ppe_models_dir))
        self.model_versions = dict(getattr(self.detector, "model_versions", lambda: {})())
        if self.classifier is not None:
            self.model_versions["classifier"] = self.classifier.version
        self.evidence = EvidenceStore(settings.resolve(settings.ppe_evidence_dir), app_cfg.evidence)
        self.spool = EventSpool(settings.resolve(settings.ppe_spool_dir))
        channels = ["dashboard"] + (["local_sound"] if app_cfg.alerts.local_sound_enabled else [])
        self.events = EventService(self.sf, self.evidence, self.spool, app_cfg.alerts, self.run_id, channels,
                                   notify=self._send)
        self.cond = threading.Condition()
        self.cams: dict[str, CamRuntime] = {}
        self.lock = threading.RLock()  # guards self.cams structure during reloads vs. the inference loop
        self.scheduler = InferenceScheduler({}, {}, app_cfg.pipeline.max_batch, app_cfg.pipeline.max_batch_wait_ms,
                                            self.cond)
        self.watchdog = CameraWatchdog()
        self.sampler = ResourceSampler("worker", {"evidence": self.evidence.root})
        self.stage_ms: dict[str, RollingWindow] = {k: RollingWindow(60.0) for k in
                                                   ("person", "ppe", "classifier", "tracking", "association", "total")}
        self.batch_sizes = RollingWindow(60.0)
        self.stop_evt = threading.Event()
        self.fault: dict[str, Any] = {}
        self.oom_times: deque[float] = deque(maxlen=10)
        self.preview_dropped = 0
        self._still_sent: dict[tuple[str, int, str], float] = {}

    # ------------------------------------------------------------ telemetry
    def _send(self, msg: dict[str, Any], droppable: bool = False) -> None:
        try:
            if droppable:
                self.telemetry.put_nowait(msg)
            else:
                self.telemetry.put(msg, timeout=0.5)
        except queue.Full:
            if droppable:
                self.preview_dropped += 1
            else:
                log.warning("telemetry queue full; %s message dropped", msg.get("type"))
        except (OSError, ValueError, EOFError, BrokenPipeError):
            self.stop_evt.set()  # parent gone

    # ------------------------------------------------------------ config
    def _load(self) -> tuple[list[Camera], list[RuleSeed], list[ZoneRuntime]]:
        with session_scope(self.sf) as s:
            cams = list(s.scalars(select(Camera).order_by(Camera.id)))
            rules = [rules_repo.rule_to_seed(r) for r in rules_repo.list_rules(s) if r.enabled]
            zones = [ZoneRuntime(z.id, z.camera_id, z.name, z.kind, [tuple(p) for p in z.polygon], z.enabled)
                     for z in rules_repo.list_zones(s)]
        return cams, rules, zones

    def _target_fps(self, cam: Camera) -> float:
        return float(cam.target_fps or self.cfg.pipeline.target_fps)

    def _build_rules(self, cr: CamRuntime, rules: list[RuleSeed], old: list[RuleRuntime] | None = None) -> None:
        prev = {r.id: r for r in old or []}
        out = []
        for r in rules:
            if r.camera_id not in (None, cr.cfg.id):
                continue
            if r.id in prev:  # hot reload keeps track states
                rr = prev[r.id]
                rr.severity, rr.zone_id = r.severity, r.zone_id
                rr.associator.set_thresholds(r.thresholds)
                rr.fsm.set_policy(r.temporal_policy)
            else:
                rr = RuleRuntime(r.id, r.severity, r.zone_id,
                                 Associator(self.cfg.association, r.thresholds, self.classifier),
                                 PPEStateMachine(cr.cfg.id, r.temporal_policy, self.cfg.tracking.track_expiry_s))
            out.append(rr)
        cr.rules = out

    def _start_camera(self, cam: Camera, rules: list[RuleSeed], zones: list[ZoneRuntime]) -> CamRuntime:
        fps = self._target_fps(cam)
        cfg = CameraRuntimeConfig(cam.id, cam.stream_uri, cam.credential_ref, cam.transport, cam.enabled, fps)
        cr = CamRuntime(cfg, cam.name, zones=[z for z in zones if z.camera_id == cam.id])
        m.camera_target_fps.labels(cam.id).set(fps if cam.enabled else 0)
        cr.health = CameraHealth(cam.id, self.cfg.ingest)
        if not cam.enabled:
            cr.health.on_disabled()
            return cr
        cr.slot = LatestFrameSlot(cam.id, self.cond)
        cr.tracker = make_tracker(self.cfg.tracking, fps)
        self._build_rules(cr, rules)
        cr.source = CameraSource(cfg, self.cfg.ingest, cr.slot, cr.health,
                                 convert_fps=self.cfg.pipeline.convert_fps_factor * fps, fault_hooks=cr.hooks)
        self.scheduler.add_camera(cam.id, cr.slot, fps)
        self.watchdog.register(cr.health, cr.source)
        cr.source.start()
        return cr

    def _stop_camera(self, cr: CamRuntime) -> None:
        self.scheduler.remove_camera(cr.cfg.id)
        self.watchdog.unregister(cr.cfg.id)
        if cr.source:
            cr.source.stop()

    def reload_cameras(self) -> None:
        cams, rules, zones = self._load()
        with self.lock:
            wanted = {c.id: c for c in cams}
            for cid in list(self.cams):
                cur, new = self.cams[cid], wanted.get(cid)
                changed = new is None or (new.stream_uri, new.credential_ref, new.transport, new.enabled,
                                          self._target_fps(new)) != (cur.cfg.stream_uri, cur.cfg.credential_ref,
                                                                     cur.cfg.transport, cur.cfg.enabled,
                                                                     cur.cfg.target_fps)
                if changed:
                    self._stop_camera(cur)
                    del self.cams[cid]
                else:
                    cur.name = new.name  # type: ignore[union-attr]
            for cid, cam in wanted.items():
                if cid not in self.cams:
                    self.cams[cid] = self._start_camera(cam, rules, zones)
        log.info("cameras loaded: %d (%d enabled)", len(self.cams), sum(c.cfg.enabled for c in self.cams.values()))

    def reload_rules(self) -> None:
        _, rules, zones = self._load()
        with self.lock:
            for cr in self.cams.values():
                cr.zones = [z for z in zones if z.camera_id == cr.cfg.id]
                if cr.cfg.enabled:
                    self._build_rules(cr, rules, cr.rules)
        log.info("rules reloaded: %d enabled rules", len(rules))

    # ------------------------------------------------------------ inference
    def _inference_loop(self) -> None:
        pcfg = self.cfg.pipeline
        while not self.stop_evt.is_set():
            batch = self.scheduler.next_batch(0.5)
            if not batch:
                continue
            t0 = time.monotonic()
            for f in batch:
                age = t0 - f.captured_mono
                m.camera_frame_age_at_inference_seconds.labels(f.camera_id).observe(age)
                cr = self.cams.get(f.camera_id)
                if cr:
                    cr.frame_age.add(age)
            try:
                self._maybe_inject_inference_fault()
                results = self.detector.detect(batch)
            except Exception as e:  # noqa: BLE001 - a failed batch must never stop the loop
                self._on_inference_error(e)
                continue
            self.batch_sizes.add(len(batch))
            for fd in results:
                for k, v in fd.stage_ms.items():
                    self.stage_ms[k].add(v)
            with self.lock:
                staged = []
                for fd in results:  # phase 1: track + associate (ambiguous cases queued, not classified yet)
                    cr = self.cams.get(fd.frame.camera_id)
                    if cr is None or cr.tracker is None:
                        continue
                    try:
                        staged.append((cr, fd, self._track_assess(cr, fd.frame, fd.detections)))
                    except Exception:
                        m.inference_errors_total.labels("association").inc()
                        log.exception("frame processing failed for %s", cr.cfg.id)
                self._classify_batch([x for _, _, st in staged for x in st["pending"]])  # phase 2: one GPU call
                for cr, fd, st in staged:  # phase 3: temporal state, events, previews
                    try:
                        self._decide(cr, fd.frame, fd.detections, st, pcfg)
                    except Exception:
                        m.inference_errors_total.labels("association").inc()
                        log.exception("frame decision failed for %s", cr.cfg.id)
            total_ms = 1000 * (time.monotonic() - t0)
            self.stage_ms["total"].add(total_ms / len(batch))
            m.inference_duration_seconds.labels("total").observe(total_ms / 1000)

    def _maybe_inject_inference_fault(self) -> None:
        kind = self.fault.pop("inference", None)
        if kind == "inference_exception":
            raise RuntimeError("injected inference exception")
        if kind == "gpu_oom":
            import torch

            raise torch.cuda.OutOfMemoryError("injected CUDA out of memory")

    def _on_inference_error(self, e: Exception) -> None:
        name = type(e).__name__
        m.inference_errors_total.labels("total").inc()
        if "OutOfMemory" in name:
            now = time.monotonic()
            self.oom_times.append(now)
            log.error("GPU OOM during inference; freeing cache (%d in last 60 s)",
                      sum(1 for t in self.oom_times if now - t < 60))
            try:
                import torch

                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001, S110
                pass
            if sum(1 for t in self.oom_times if now - t < 60) >= 3:
                log.critical("repeated GPU OOM - exiting so the supervisor restarts a clean worker")
                self.stop_evt.set()
                os._exit(75)
        else:
            log.exception("inference batch failed: %s", name)

    def _track_assess(self, cr: CamRuntime, frame: Frame, dets: list) -> dict[str, Any]:
        t0 = time.perf_counter()
        persons = [d for d in dets if d.cls == "person"]
        tracked = cr.tracker.update(persons, frame)  # type: ignore[union-attr]
        tracked = filter_ignored(tracked, cr.zones, frame.width, frame.height)
        t1 = time.perf_counter()
        self.stage_ms["tracking"].add(1000 * (t1 - t0))
        m.inference_duration_seconds.labels("tracking").observe(t1 - t0)
        ppe_dets = [d for d in dets if d.cls in ("helmet", "head")]
        per_rule, pending = [], []
        for rr in cr.rules:
            scoped = tracked
            if rr.zone_id:
                scoped = [p for p in tracked if zone_of(p.box, cr.zones, frame.width, frame.height) == rr.zone_id]
            obs = rr.associator.assess_deferred(scoped, ppe_dets, frame)
            pending += rr.associator.take_pending()
            per_rule.append((rr, obs))
        t2 = time.perf_counter()
        self.stage_ms["association"].add(1000 * (t2 - t1))
        m.inference_duration_seconds.labels("association").observe(t2 - t1)
        return {"per_rule": per_rule, "pending": pending}

    def _classify_batch(self, pending: list) -> None:
        if not pending or self.classifier is None:
            return
        t0 = time.perf_counter()
        try:
            apply_classification(pending, self.classifier.classify([crop for _, crop in pending]))
        except Exception:  # noqa: BLE001 - classifier failure leaves the cases UNCERTAIN (never a violation)
            m.inference_errors_total.labels("classifier").inc()
            log.exception("head classifier batch failed")
        self.stage_ms["classifier"].add(1000 * (time.perf_counter() - t0) / len(pending))

    def _decide(self, cr: CamRuntime, frame: Frame, dets: list, st: dict[str, Any], pcfg) -> None:
        overlay_state: dict[int, tuple[str, Any]] = {}
        masked: np.ndarray | None = None
        for rr, obs in st["per_rule"]:
            for o in obs:
                o.zone_id = rr.zone_id
                if o.outcome is Outcome.UNCERTAIN:
                    m.ppe_uncertain_observations_total.labels(cr.cfg.id).inc()
            for tr in rr.fsm.update(obs, frame.captured_at) + rr.fsm.expire(frame.captured_at):
                if tr.new is S.CONFIRMED_NON_COMPLIANT:
                    t = rr.fsm.tracks[tr.track_id]
                    o = t.last_observation
                    if masked is None:
                        masked = apply_privacy_masks(frame.image, cr.zones)
                    p = rr.fsm.policy
                    self.events.submit_confirmed(ConfirmedViolation(
                        cr.cfg.id, tr.track_id, rr.zone_id, rr.id, rr.severity, tr.episode_seq, o.person_box,
                        o.head_roi, tr.reason, t.votes(p.confirm_m, Outcome.NO_HELMET), len(t.decisive(p.confirm_m)),
                        t.no_helmet_confidence(p.confirm_m), o.roi_source, o.classifier_used, masked,
                        frame.captured_at, t.first_violation_at or frame.captured_at, tr.ts, self.model_versions,
                        cr.name))
                elif tr.prev is S.CONFIRMED_NON_COMPLIANT and tr.new is S.CLEARED:
                    self.events.submit_cleared(cr.cfg.id, tr.track_id, rr.id, tr.ts)
            for o in obs:
                state = rr.fsm.state_of(o.track_id)
                if state is S.CONFIRMED_NON_COMPLIANT:
                    k = (cr.cfg.id, o.track_id, rr.id)
                    if frame.captured_mono - self._still_sent.get(k, -1e9) >= 1.0:  # bounded queue load per track
                        self._still_sent[k] = frame.captured_mono
                        if len(self._still_sent) > 5000:  # bounded: forget tracks not refreshed for a minute
                            cut = frame.captured_mono - 60
                            self._still_sent = {kk: t for kk, t in self._still_sent.items() if t >= cut}
                        self.events.submit_still_confirmed(cr.cfg.id, o.track_id, rr.id, o.person_box, o.head_roi)
                if o.track_id not in overlay_state or state is S.CONFIRMED_NON_COMPLIANT:
                    overlay_state[o.track_id] = (state.value, o)
        cr.processed.add(1.0)
        m.tracker_active_tracks.labels(cr.cfg.id).set(cr.tracker.active_count)  # type: ignore[union-attr]
        now = time.monotonic()
        if now - cr.last_preview >= 1.0 / pcfg.preview_fps:
            cr.last_preview = now
            self._send_preview(cr, frame, dets, overlay_state, masked, pcfg)

    def _send_preview(self, cr: CamRuntime, frame: Frame, dets: list, overlay: dict, masked, pcfg) -> None:
        img = masked if masked is not None else apply_privacy_masks(frame.image, cr.zones)
        sx = pcfg.preview_width / frame.width
        h = max(1, round(frame.height * sx))
        small = cv2.resize(img, (pcfg.preview_width, h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, pcfg.preview_jpeg_quality])
        if not ok:
            return

        def sc(b):
            return None if b is None else [round(v, 1) for v in scale(b, sx, sx)]

        tracks = [{"track_id": tid, "box": sc(o.person_box), "ppe_state": st, "outcome": o.outcome.value,
                   "head_roi": sc(o.head_roi), "helmet_box": sc(o.helmet_box), "head_box": sc(o.head_box)}
                  for tid, (st, o) in overlay.items()]
        det_list = [{"cls": d.cls, "conf": round(d.conf, 3), "box": sc(d.xyxy)} for d in dets]
        self._send({"type": "preview", "camera_id": cr.cfg.id, "frame_id": frame.frame_id,
                    "captured_at": frame.captured_at, "width": pcfg.preview_width, "height": h,
                    "jpeg": buf.tobytes(), "tracks": tracks, "detections": det_list}, droppable=True)

    # ------------------------------------------------------------ status
    def camera_status(self) -> list[dict[str, Any]]:
        out = []
        now = time.monotonic()
        uptime_base = max(1e-6, time.time() - self.started_at)
        with self.lock:
            items = list(self.cams.values())
        for cr in items:
            h = cr.health
            assert h is not None
            age = h.last_frame_age()
            q = cr.frame_age.quantiles(now)
            pfps = cr.processed.rate(now)
            m.camera_processed_fps.labels(cr.cfg.id).set(pfps)
            src = cr.source
            out.append({
                "camera_id": cr.cfg.id, "state": h.state.value, "state_reason": h.reason,
                "state_since": h.since_wall, "last_frame_at": h.last_frame_wall, "last_frame_age_s": age,
                "source_fps": h.nominal_fps, "decoded_fps": round(h.decoded_fps, 2), "processed_fps": round(pfps, 2),
                "target_fps": cr.cfg.target_fps if cr.cfg.enabled else 0.0, "reconnect_count": h.reconnect_count,
                "decode_errors": h.decode_errors, "frames_dropped": cr.slot.dropped if cr.slot else 0,
                "queue_depth": cr.slot.depth if cr.slot else 0,
                "frame_age_at_inference_p95_s": round(q["p95"], 4) if q else None,
                "uptime_ratio": round(min(1.0, h.online_seconds / uptime_base), 4),
                "last_recovery_s": round(h.last_recovery_s, 3) if h.last_recovery_s is not None else None,
                "active_tracks": cr.tracker.active_count if cr.tracker else 0,
                "width": src.width if src else None, "height": src.height if src else None,
            })
            if h.state.value != cr.last_state and self.events.db_ok:
                cr.last_state = h.state.value
                try:
                    with session_scope(self.sf) as s:
                        record_health(s, cr.cfg.id, h.state.value, h.reason)
                except Exception:  # noqa: BLE001 - history is best effort
                    log.debug("camera_health history write failed", exc_info=True)
        return out

    def _telemetry_loop(self) -> None:
        while not self.stop_evt.wait(1.0):
            now = time.time()
            cams = self.camera_status()
            self._send({"type": "camera_status", "ts": now, "cameras": cams})
            res = self.sampler.sample()
            inf = {k: {"p50_ms": q["p50"], "p95_ms": q["p95"], "p99_ms": q["p99"], "count": q["count"]}
                   for k, w in self.stage_ms.items() if (q := w.quantiles())}
            bq = self.batch_sizes.quantiles()
            self._send({"type": "stats", "ts": now, "aggregate_processed_fps": round(sum(c["processed_fps"]
                        for c in cams), 2), "inference": inf, "batch_size_mean": bq["p50"] if bq else 0.0,
                        "resources": res, "spool_pending": len(self.spool.pending()), "db_ok": self.events.db_ok,
                        "preview_dropped": self.preview_dropped})
            self._send({"type": "heartbeat", "ts": now, "run_id": self.run_id})
            self._expire_faults()

    def _expire_faults(self) -> None:
        until = self.fault.get("disk_full_until")
        if until and time.monotonic() > until:
            self.fault.pop("disk_full_until", None)
            self.evidence.disk_usage = shutil.disk_usage
            log.info("disk_full fault cleared")

    # ------------------------------------------------------------ commands
    def handle(self, cmd: dict[str, Any]) -> None:
        t = cmd.get("type")
        if t == "reload_cameras":
            self.reload_cameras()
        elif t == "reload_rules":
            self.reload_rules()
        elif t == "shutdown":
            self.stop_evt.set()
        elif t == "fault":
            self._fault(cmd)

    def _fault(self, cmd: dict[str, Any]) -> None:
        kind, cam_id, dur = cmd.get("kind"), cmd.get("camera_id"), float(cmd.get("duration_s") or 10.0)
        log.warning("FAULT INJECTION: %s camera=%s duration=%.1fs", kind, cam_id, dur)
        if kind in ("inference_exception", "gpu_oom"):
            self.fault["inference"] = kind
        elif kind == "crash":
            os._exit(3)
        elif kind == "disk_full":
            self.evidence.disk_usage = lambda _p: SimpleNamespace(free=0)
            self.fault["disk_full_until"] = time.monotonic() + dur
        elif kind in ("decoder_error", "stall") and cam_id in self.cams:
            hooks = self.cams[cam_id].hooks
            if kind == "decoder_error":
                hooks.decoder_error_once = True
            else:
                hooks.stall_until_mono = time.monotonic() + dur

    def _parent_alive(self) -> bool:
        if self.parent_pid is None:
            return True
        try:
            return psutil.pid_exists(self.parent_pid) and psutil.Process(self.parent_pid).is_running()
        except psutil.Error:
            return False

    # ------------------------------------------------------------ lifecycle
    def run(self) -> int:
        from prometheus_client import start_http_server

        try:
            start_http_server(self.s.worker_metrics_port, addr="127.0.0.1")
        except OSError as e:
            log.warning("worker metrics port %d unavailable: %s", self.s.worker_metrics_port, e)
        self.detector.warmup()
        try:
            with session_scope(self.sf) as s:
                record_model_versions(s, self.model_versions)
        except Exception:  # noqa: BLE001
            log.warning("could not record model versions (database unavailable?)")
        self._send({"type": "hello", "run_id": self.run_id, "pid": os.getpid(), "started_at": self.started_at,
                    "model_versions": self.model_versions, "device": self.models_cfg.person.device})
        self.events.start()
        self.reload_cameras()
        self.watchdog.start()
        threads = [threading.Thread(target=self._inference_loop, name="inference", daemon=True),
                   threading.Thread(target=self._telemetry_loop, name="telemetry", daemon=True)]
        for t in threads:
            t.start()
        log.info("worker %s running (pid %d)", self.run_id, os.getpid())
        while not self.stop_evt.is_set():
            try:
                cmd = self.commands.get(timeout=1.0)
                self.handle(cmd)
            except queue.Empty:
                pass
            except (EOFError, OSError):
                break
            if not self._parent_alive():
                log.warning("parent process gone - shutting down")
                break
        self.shutdown()
        return 0

    def shutdown(self) -> None:
        self.stop_evt.set()
        self.watchdog.stop()
        with self.lock:
            for cr in self.cams.values():
                if cr.source:
                    cr.source.stop(timeout=3)
        self.events.stop()
        self.engine.dispose()
        # Buffered telemetry must not keep a dying worker alive when the reader is gone (mp feeder-thread hang).
        if hasattr(self.telemetry, "cancel_join_thread"):
            self.telemetry.cancel_join_thread()
        log.info("worker %s stopped", self.run_id)


def run_worker(telemetry: mp.Queue, commands: mp.Queue, parent_pid: int) -> None:
    """Process entry point (spawn start method)."""
    setup_logging("worker")
    s = get_settings()
    w = PipelineWorker(s, load_app_config(s), load_models_config(s), telemetry, commands, parent_pid=parent_pid)
    raise SystemExit(w.run())
