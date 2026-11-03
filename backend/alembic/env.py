from __future__ import annotations

import os

from alembic import context
from sqlalchemy import create_engine

from app.config.settings import get_settings
from app.models import Base

config = context.config
target_metadata = Base.metadata


def _url() -> str:
    return (config.attributes.get("url") or context.get_x_argument(as_dictionary=True).get("url")
            or os.environ.get("PPE_DATABASE_URL") or get_settings().database_url)


def run_migrations_offline() -> None:
    context.configure(url=_url(), target_metadata=target_metadata, literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_url())
    with engine.connect() as conn:
        context.configure(connection=conn, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()
    engine.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
