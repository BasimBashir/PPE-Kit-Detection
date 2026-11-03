"""REST + WebSocket integration (spec 19.2): auth/roles, camera & rule workflows, events lifecycle, outbox -> WS,
render-ack latency, structured errors, no credential leakage. Runs the real app without the worker process."""

import time
import uuid

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.alerts.event_service import ConfirmedViolation, EventService
from app.alerts.spool import EventSpool
from app.config.settings import AlertsConfig, EvidenceConfig, Settings, load_app_config
from app.db.base import make_engine, make_session_factory
from app.evidence.store import EvidenceStore
from app.services.live_state import LiveState
from app.services.ws_hub import WebSocketHub

pytestmark = pytest.mark.integration


@pytest.fixture()
def client(test_db_url, sf, tmp_path, monkeypatch):
    from app.api.deps import AppContext
    from app.main import create_app

    monkeypatch.setenv("RTSP_READ_USER", "viewer")
    monkeypatch.setenv("RTSP_READ_PASS", "s3cret-pass")
    s = Settings(PPE_DATABASE_URL=test_db_url, ppe_admin_password="admin-password-1",
                 ppe_enable_fault_injection=True)
    cfg = load_app_config(s)
    eng = make_engine(test_db_url)
    ctx = AppContext(settings=s, cfg=cfg, engine=eng, sf=make_session_factory(eng), live=LiveState(),
                     hub=WebSocketHub(),
                     evidence=EvidenceStore(tmp_path / "evidence", EvidenceConfig(min_free_bytes=0)))
    with TestClient(create_app(start_worker=False, context=ctx)) as c:
        c.ctx = ctx
        yield c
    eng.dispose()


def login(c, user="admin", pw="admin-password-1"):
    r = c.post("/api/v1/auth/login", json={"username": user, "password": pw})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_health_ready_and_structured_errors(client):
    assert client.get("/api/v1/health").json()["status"] == "ok"
    r = client.get("/api/v1/ready")
    assert r.status_code == 503 and r.json()["checks"]["database"]["ok"] is True  # worker not running
    r = client.get("/api/v1/cameras")
    assert r.status_code == 401 and r.json()["error"]["code"] == "unauthorized" and "request_id" in r.json()
    r = client.post("/api/v1/auth/login", json={"username": "admin"})
    assert r.status_code == 422 and r.json()["error"]["code"] == "validation_error"
    assert "camera_connection_state" in client.get("/metrics").text


def test_roles_and_login_rate_limit(client):
    admin = login(client)
    r = client.post("/api/v1/users", json={"username": "viewer1", "password": "viewer-pass-1", "role": "viewer"},
                    headers=admin)
    assert r.status_code == 201
    viewer = login(client, "viewer1", "viewer-pass-1")
    assert client.get("/api/v1/cameras", headers=viewer).status_code == 200
    r = client.patch("/api/v1/cameras/cam01", json={"enabled": False}, headers=viewer)
    assert r.status_code == 403 and r.json()["error"]["code"] == "forbidden"
    for _ in range(5):
        client.post("/api/v1/auth/login", json={"username": "x", "password": "y"})
    assert client.post("/api/v1/auth/login", json={"username": "x", "password": "y"}).status_code == 429


def test_cameras_never_leak_credentials(client):
    admin = login(client)
    cams = client.get("/api/v1/cameras", headers=admin).json()
    assert len(cams) == 10 and cams[0]["credential_ref"] == "env:RTSP_READ" and cams[0]["has_credentials"] is True
    body = client.get("/api/v1/cameras", headers=admin).text
    assert "s3cret-pass" not in body and "viewer:" not in body
    bad = client.post("/api/v1/cameras", headers=admin,
                      json={"id": "cam99", "name": "x", "stream_uri": "rtsp://u:p@h/c"})
    assert bad.status_code == 422
    ok = client.post("/api/v1/cameras", headers=admin,
                     json={"id": "cam99", "name": "Yard", "stream_uri": "rtsp://localhost:9/cam99",
                           "credential_ref": "env:RTSP_READ", "target_fps": 2})
    assert ok.status_code == 201 and ok.json()["target_fps"] == 2
    # connectivity test to an unreachable port: fails cleanly and the error is redacted
    t = client.post("/api/v1/cameras/cam99/test", headers=admin).json()
    assert t["ok"] is False and "s3cret-pass" not in (t["error"] or "")
    assert client.delete("/api/v1/cameras/cam99", headers=admin).status_code == 204


def test_rules_validation_and_effective_seconds(client):
    admin = login(client)
    rules = client.get("/api/v1/rules", headers=admin).json()
    assert rules[0]["id"] == "hardhat-required" and rules[0]["effective_confirm_seconds"] == pytest.approx(11 / 5)
    bad = client.patch("/api/v1/rules/hardhat-required", headers=admin,
                       json={"temporal_policy": {"confirm_n": 20, "confirm_m": 15}})
    assert bad.status_code == 422
    ok = client.patch("/api/v1/rules/hardhat-required", headers=admin,
                      json={"temporal_policy": {"mode": "duration", "min_duration_s": 3.0}})
    assert ok.status_code == 200 and ok.json()["effective_confirm_seconds"] == 3.0


def _confirm(ctx, tmp_path):
    svc = EventService(ctx.sf, ctx.evidence, EventSpool(tmp_path / "spool"), AlertsConfig(), "apitest",
                       ["dashboard"], notify=lambda _m: ctx.dispatcher.wake(), replay_interval_s=3600)
    svc.start()
    now = time.time()
    svc.submit_confirmed(ConfirmedViolation("cam03", 4, None, "hardhat-required", "high", 1, (10.0, 10.0, 90.0, 300.0),
                                            (25.0, 0.0, 75.0, 60.0), "temporal_policy_met:count", 12, 14, 0.81,
                                            "keypoints", False, np.zeros((360, 640, 3), np.uint8), now - 0.2,
                                            now - 3, now))
    svc.drain()
    svc.stop()


def test_event_flow_outbox_ws_render_ack_lifecycle(client, tmp_path):
    admin = login(client)
    token = admin["Authorization"][7:]
    with client.websocket_connect(f"/ws/live?token={token}") as ws:
        snap = ws.receive_json()
        assert snap["type"] == "snapshot" and len(snap["cameras"]) == 10 and snap["open_events"] == []
        _confirm(client.ctx, tmp_path)
        deadline = time.time() + 10
        msg = None
        while time.time() < deadline:
            msg = ws.receive_json()
            if msg["type"] == "event_created":
                break
        assert msg and msg["type"] == "event_created"
        ev = msg["event"]
        assert ev["camera_id"] == "cam03" and ev["has_evidence"] and ev["latency"]["time_to_confirm_s"] > 2.5
        ws.send_json({"type": "event_rendered", "event_id": ev["id"], "rendered_at": time.strftime(
            "%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time() % 1 * 1000):03d}Z"})
        time.sleep(0.5)
    e = client.get(f"/api/v1/events/{ev['id']}", headers=admin).json()
    assert e["latency"]["persist_to_dashboard_s"] is not None
    img = client.get(f"/api/v1/events/{ev['id']}/evidence", headers=admin)
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert client.post(f"/api/v1/events/{ev['id']}/acknowledge", headers=admin, json={"note": "seen"}).json()[
        "status"] == "ACKNOWLEDGED"
    assert client.post(f"/api/v1/events/{ev['id']}/acknowledge", headers=admin).status_code == 409
    assert client.post(f"/api/v1/events/{ev['id']}/resolve", headers=admin).json()["status"] == "RESOLVED"
    fb = client.post(f"/api/v1/events/{ev['id']}/feedback", headers=admin, json={"label": "TRUE_POSITIVE"}).json()
    assert fb["feedback"] == "TRUE_POSITIVE"
    page = client.get("/api/v1/events?camera_id=cam03&status=RESOLVED&limit=10", headers=admin).json()
    assert page["total"] == 1 and page["items"][0]["id"] == ev["id"]
    detail = client.get(f"/api/v1/events/{ev['id']}", headers=admin).json()
    assert [a["action"] for a in detail["actions"]] == ["ACKNOWLEDGE", "RESOLVE", "FEEDBACK"]
    assert client.get(f"/api/v1/events/{uuid.uuid4()}", headers=admin).status_code == 404
    rep = client.get("/api/v1/evaluation/report", headers=admin).json()
    assert rep["summary"]["events"] == 1 and rep["summary"]["labelled"] == 1


def test_ws_rejects_bad_token(client):
    from starlette.websockets import WebSocketDisconnect

    with pytest.raises(WebSocketDisconnect), client.websocket_connect("/ws/live?token=nope") as ws:
        ws.receive_json()


def test_metrics_summary_shape_without_worker(client):
    admin = login(client)
    mtr = client.get("/api/v1/metrics", headers=admin).json()
    assert mtr["worker"]["up"] is False and mtr["aggregate"]["cameras_total"] == 10
    assert mtr["aggregate"]["online"] == 0  # never report cameras ONLINE without a worker
    r = client.post("/api/v1/system/faults", headers=admin, json={"kind": "stall", "camera_id": "cam01"})
    assert r.status_code == 202
