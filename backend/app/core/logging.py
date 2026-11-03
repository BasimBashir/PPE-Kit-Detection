"""Structured JSON logs with credential redaction (NFR-OBS-001, NFR-SEC-001). No per-frame logging anywhere."""

from __future__ import annotations

import json
import logging
import sys
import time

from app.core.redaction import RedactingFilter


class JsonFormatter(logging.Formatter):
    def __init__(self, process_name: str):
        super().__init__()
        self.process_name = process_name

    def format(self, r: logging.LogRecord) -> str:
        out = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(r.created)) + f".{int(r.msecs):03d}Z",
            "level": r.levelname, "proc": self.process_name, "logger": r.name, "msg": r.getMessage(),
        }
        if r.exc_text:
            out["exc"] = r.exc_text
        return json.dumps(out, ensure_ascii=False)


def setup_logging(process_name: str, level: int = logging.INFO) -> None:
    h = logging.StreamHandler(sys.stderr)
    h.addFilter(RedactingFilter())
    h.setFormatter(JsonFormatter(process_name))
    root = logging.getLogger()
    root.handlers[:] = [h]
    root.setLevel(level)
    for noisy in ("ultralytics", "httpx", "uvicorn.access", "libav"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
