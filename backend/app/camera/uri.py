"""Stream URI validation and credential resolution (FR-CAM-001/008).

Stored URIs never contain credentials. A camera's `credential_ref` ("env:PREFIX") names environment
variables PREFIX_USER / PREFIX_PASS that are injected only at connect time, inside the worker.
"""

from __future__ import annotations

import os
from urllib.parse import quote, urlsplit, urlunsplit

from app.core.redaction import register_secret

ALLOWED_SCHEMES = ("rtsp", "rtsps", "file")


class CredentialError(ValueError):
    pass


def validate_stream_uri(uri: str) -> str:
    uri = uri.strip()
    parts = urlsplit(uri)
    if parts.scheme not in ALLOWED_SCHEMES:
        raise ValueError(f"unsupported scheme {parts.scheme!r}; allowed: {', '.join(ALLOWED_SCHEMES)}")
    if parts.username is not None or parts.password is not None or "@" in parts.netloc:
        raise ValueError("credentials must not be embedded in stream_uri; use credential_ref")
    if parts.scheme != "file" and not parts.hostname:
        raise ValueError("stream_uri must include a host")
    if parts.scheme == "file" and not parts.path:
        raise ValueError("file URI must include a path")
    return uri


def validate_credential_ref(ref: str | None) -> str | None:
    if ref is None or ref == "":
        return None
    kind, _, name = ref.partition(":")
    if kind != "env" or not name or not name.replace("_", "").isalnum():
        raise ValueError("credential_ref must look like 'env:PREFIX' (PREFIX_USER / PREFIX_PASS)")
    return ref


def resolve_uri(uri: str, credential_ref: str | None) -> str:
    """Return the connectable URI with credentials injected. Never log the result unredacted."""
    validate_stream_uri(uri)
    if not credential_ref:
        return uri
    validate_credential_ref(credential_ref)
    prefix = credential_ref.split(":", 1)[1]
    user, pw = os.environ.get(f"{prefix}_USER"), os.environ.get(f"{prefix}_PASS")
    if not user or pw is None:
        raise CredentialError(f"credential_ref {credential_ref!r}: {prefix}_USER/{prefix}_PASS not set")
    register_secret(pw)
    p = urlsplit(uri)
    netloc = f"{quote(user, safe='')}:{quote(pw, safe='')}@{p.netloc}"
    return urlunsplit((p.scheme, netloc, p.path, p.query, p.fragment))
