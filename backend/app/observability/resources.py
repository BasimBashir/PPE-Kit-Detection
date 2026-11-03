"""Process / host / GPU resource sampling (NFR-OBS-001, spec 10.1 Resources). Safe when no NVIDIA GPU exists."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import psutil

from app.observability import metrics as m

log = logging.getLogger(__name__)


class ResourceSampler:
    def __init__(self, process_name: str, disk_paths: dict[str, Path] | None = None):
        self.process_name = process_name
        self.proc = psutil.Process()
        self.proc.cpu_percent(None)
        psutil.cpu_percent(None)
        self.disk_paths = disk_paths or {}
        self._nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception as e:  # noqa: BLE001 - no GPU / driver: GPU metrics stay absent
            log.info("NVML unavailable (%s); GPU metrics disabled", type(e).__name__)

    def sample(self) -> dict[str, float | None]:
        out: dict[str, float | None] = {}
        out["rss_bytes"] = float(self.proc.memory_info().rss)
        out["cpu_percent"] = self.proc.cpu_percent(None)
        out["system_cpu_percent"] = psutil.cpu_percent(None)
        out["system_memory_used_bytes"] = float(psutil.virtual_memory().used)
        m.app_process_rss_bytes.labels(self.process_name).set(out["rss_bytes"])
        m.app_process_cpu_percent.labels(self.process_name).set(out["cpu_percent"])
        m.system_cpu_percent.set(out["system_cpu_percent"])
        m.system_memory_used_bytes.set(out["system_memory_used_bytes"])
        for label, p in self.disk_paths.items():
            try:
                free = float(shutil.disk_usage(p).free)
            except OSError:
                free = None
            out[f"disk_free_bytes_{label}"] = free
            if free is not None:
                m.disk_free_bytes.labels(label).set(free)
        if self._nvml is not None:
            nv = self._nvml
            try:
                util = nv.nvmlDeviceGetUtilizationRates(self._gpu)
                mem = nv.nvmlDeviceGetMemoryInfo(self._gpu)
                out.update(gpu_util_percent=float(util.gpu), gpu_memory_used_bytes=float(mem.used),
                           gpu_memory_total_bytes=float(mem.total),
                           gpu_temperature_c=float(nv.nvmlDeviceGetTemperature(self._gpu, nv.NVML_TEMPERATURE_GPU)),
                           gpu_power_w=nv.nvmlDeviceGetPowerUsage(self._gpu) / 1000.0)
                m.gpu_utilization_percent.labels("0").set(util.gpu)
                m.gpu_memory_used_bytes.labels("0").set(mem.used)
                m.gpu_memory_total_bytes.labels("0").set(mem.total)
                m.gpu_temperature_celsius.labels("0").set(out["gpu_temperature_c"])  # type: ignore[arg-type]
                m.gpu_power_watts.labels("0").set(out["gpu_power_w"])  # type: ignore[arg-type]
            except Exception:  # noqa: BLE001
                log.debug("NVML sample failed", exc_info=True)
        try:
            import torch

            if torch.cuda.is_available() and torch.cuda.is_initialized():
                out["torch_cuda_allocated_bytes"] = float(torch.cuda.memory_allocated())
                m.torch_cuda_memory_allocated_bytes.set(out["torch_cuda_allocated_bytes"])  # type: ignore[arg-type]
        except Exception:  # noqa: BLE001
            log.debug("torch CUDA memory sample failed", exc_info=True)
        return out
