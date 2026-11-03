import logging

import pytest

from app.camera.uri import CredentialError, resolve_uri, validate_credential_ref, validate_stream_uri
from app.core.redaction import RedactingFilter, redact


def test_redact_userinfo_in_pyav_style_message():
    msg = "Server returned 401 Unauthorized: 'rtsp://viewer:s3cr3t@localhost:18501/cam01'"
    out = redact(msg)
    assert "s3cr3t" not in out and "viewer" not in out
    assert "rtsp://***@localhost:18501/cam01" in out


def test_redact_query_secrets():
    assert redact("http://x/y?token=abc&password=def") == "http://x/y?token=***&password=***"


def test_redact_leaves_clean_text():
    assert redact("rtsp://localhost:8554/cam01 opened") == "rtsp://localhost:8554/cam01 opened"


@pytest.mark.parametrize("uri", ["rtsp://u:p@host/cam", "rtsp://u@host/cam", "http://host/x", "rtsp:///nohost"])
def test_validate_rejects(uri):
    with pytest.raises(ValueError):
        validate_stream_uri(uri)


@pytest.mark.parametrize("uri", ["rtsp://localhost:18501/cam01", "rtsps://10.0.0.5/s1", "file:///C:/v/cam01.mp4"])
def test_validate_accepts(uri):
    assert validate_stream_uri(uri) == uri


@pytest.mark.parametrize("ref", ["env:RTSP_READ", None, ""])
def test_credential_ref_ok(ref):
    validate_credential_ref(ref)


@pytest.mark.parametrize("ref", ["RTSP_READ", "vault:x", "env:", "env:A-B"])
def test_credential_ref_bad(ref):
    with pytest.raises(ValueError):
        validate_credential_ref(ref)


def test_resolve_injects_and_registers_secret(monkeypatch):
    monkeypatch.setenv("CAMX_USER", "viewer")
    monkeypatch.setenv("CAMX_PASS", "p@ss/word!")
    uri = resolve_uri("rtsp://localhost:18501/cam01", "env:CAMX")
    assert uri == "rtsp://viewer:p%40ss%2Fword%21@localhost:18501/cam01"
    assert "p@ss/word!" not in redact("boom p@ss/word! boom")  # registered raw secret masked anywhere


def test_resolve_missing_env(monkeypatch):
    monkeypatch.delenv("NOPE_USER", raising=False)
    with pytest.raises(CredentialError):
        resolve_uri("rtsp://h/c", "env:NOPE")


def test_logging_filter_redacts_message_and_exception(caplog):
    logger = logging.getLogger("t.redact")
    handler = logging.Handler()
    records = []
    handler.emit = records.append  # type: ignore[method-assign]
    handler.addFilter(RedactingFilter())
    logger.addHandler(handler)
    try:
        raise RuntimeError("open failed rtsp://a:b@h/c")
    except RuntimeError:
        logger.exception("camera %s failed at %s", "cam01", "rtsp://a:b@h/c")
    logger.removeHandler(handler)
    rec = records[0]
    assert "a:b@" not in rec.getMessage()
    assert "a:b@" not in (rec.exc_text or "")
