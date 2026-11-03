"""Engine / session helpers (sync SQLAlchemy 2 + psycopg 3)."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

SessionFactory = Callable[[], Session]


def make_engine(url: str, pool_size: int = 5) -> Engine:
    return create_engine(url, pool_pre_ping=True, pool_size=pool_size, max_overflow=5, pool_recycle=1800,
                         connect_args={"connect_timeout": 3})


def make_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: SessionFactory) -> Iterator[Session]:
    s = factory()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def db_healthcheck(engine: Engine) -> bool:
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001 - health probe reports, never raises
        return False
