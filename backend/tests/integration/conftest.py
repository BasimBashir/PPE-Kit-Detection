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
                       "alert_outbox, event_actions, audit_logs, system_metrics_snapshots, settings CASCADE"))
    return make_session_factory(engine)


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
