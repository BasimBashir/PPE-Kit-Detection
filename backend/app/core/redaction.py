"""Credential redaction for logs, exceptions and API payloads (FR-CAM-008, NFR-SEC-001).

FFmpeg/PyAV error messages embed the full URL including user:password (observed: HTTPUnauthorizedError
"... 'rtsp://viewer:secret@host/cam01'"), so every message derived from a stream error must pass here.
"""

from __future__ import annotations

import logging
import re

# scheme://user:pass@  or  scheme://user@
_USERINFO = re.compile(r"(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)(?P<userinfo>[^/@\s'\"]+)@")
_KV_SECRET = re.compile(r"(?i)\b(password|passwd|pass|pwd|token|secret)=([^&\s'\"]+)")

_known_secrets: set[str] = set()


def register_secret(value: str | None) -> None:
    """Values registered here are masked wherever they appear (e.g. resolved camera passwords)."""
    if value and len(value) >= 4:
        _known_secrets.add(value)


def redact(text: object) -> str:
    s = str(text)
    s = _USERINFO.sub(lambda m: f"{m.group('scheme')}***@", s)
    s = _KV_SECRET.sub(lambda m: f"{m.group(1)}=***", s)
    for secret in _known_secrets:
        if secret in s:
            s = s.replace(secret, "***")
    return s


class RedactingFilter(logging.Filter):
    """Attach to handlers: redacts the formatted message and exception text."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = None
        if record.exc_info and record.exc_info[1] is not None:
            record.exc_text = redact(logging.Formatter().formatException(record.exc_info))
            record.exc_info = None
        return True
