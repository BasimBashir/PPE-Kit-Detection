"""Durable local spool for events created while the database is unreachable (FR-ALT-009, ADR-007)."""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.observability import metrics as m

log = logging.getLogger(__name__)


class EventSpool:
    def __init__(self, directory: Path):
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "events.jsonl"
        self._lock = threading.Lock()
        m.ppe_spool_pending.set(len(self.pending()))

    def append(self, payload: dict[str, Any]) -> None:
        line = json.dumps(payload, separators=(",", ":"), default=str) + "\n"
        with self._lock, open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
            m.ppe_events_spooled_total.inc()
        m.ppe_spool_pending.set(len(self.pending()))

    def pending(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    log.error("corrupt spool line skipped")  # a torn final write after power loss
        return out

    def replay(self, insert_fn: Callable[[dict[str, Any]], None]) -> int:
        """Insert every spooled payload; keep the ones that fail; rewrite atomically. Returns inserted count."""
        with self._lock:
            items = self.pending()
            if not items:
                return 0
            keep, done = [], 0
            for i, item in enumerate(items):
                try:
                    insert_fn(item)
                    done += 1
                except Exception as e:  # noqa: BLE001 - keep for the next replay round
                    log.warning("spool replay failed (%s); %d items kept", type(e).__name__, len(items) - i)
                    keep = items[i:]
                    break
            tmp = self.path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                fh.writelines(json.dumps(k, separators=(",", ":"), default=str) + "\n" for k in keep)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, self.path)
            m.ppe_spool_pending.set(len(keep))
            return done
