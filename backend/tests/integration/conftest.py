import pytest
from sqlalchemy import text

from app.config.settings import Settings
from app.db.base import make_engine, make_session_factory, session_scope
from app.db.migrate import upgrade_to_head


@pytest.fixture(scope="session")
def test_db_url(dotenv) -> str:
    s = Settings()
    url = s.database_url.rsplit("/", 1)[0] + "/ppe_test"
    eng = make_engine(url)
    try:
        with eng.begin() as c:
            c.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"PostgreSQL ppe_test not reachable: {type(e).__name__}")
    eng.dispose()
    upgrade_to_head(url)
    return url


@pytest.fixture(scope="session")
def engine(test_db_url):
    e = make_engine(test_db_url)
    yield e
    e.dispose()


@pytest.fixture()
def sf(engine):
    """Session factory on a clean set of tables (truncated per test)."""
    with engine.begin() as c:
        c.execute(text("TRUNCATE users, sessions, cameras, camera_health, zones, ppe_rules, model_versions, events, "
                       "alert_outbox, event_actions, audit_logs, system_metrics_snapshots, settings, teacher_reviews, "
                       "persons, crossings, presence_intervals "
                       "CASCADE"))
    return make_session_factory(engine)


@pytest.fixture()
def client(test_db_url, sf, tmp_path, monkeypatch):
    """The real app without the worker (same as test_api's local fixture, shared with other modules)."""
    from fastapi.testclient import TestClient

    from app.api.deps import AppContext
    from app.config.settings import EvidenceConfig, load_app_config
    from app.evidence.store import EvidenceStore
    from app.main import create_app
    from app.services.live_state import LiveState
    from app.services.ws_hub import WebSocketHub

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


@pytest.fixture()
def seeded(sf):
    from app.config.settings import CameraSeed, RuleSeed
    from app.repositories.misc import seed_if_empty

    with session_scope(sf) as s:
        seed_if_empty(s, [CameraSeed(id="cam01", name="Cam 1", stream_uri="rtsp://localhost:18501/cam01",
                                     credential_ref="env:RTSP_READ"),
                          CameraSeed(id="cam02", name="Cam 2", stream_uri="rtsp://localhost:18502/cam02")],
                      [RuleSeed(id="hardhat-required", name="Hard hat")], [], "admin-password-1")
    return sf
