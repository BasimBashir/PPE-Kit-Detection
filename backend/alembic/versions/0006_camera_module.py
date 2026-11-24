"""camera module selection: per-stream check list (1 stream = 1..n classes; null = full cascade)

Revision ID: 0006
Revises: 0005
Create Date: 2026-11-24 20:00:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("cameras", sa.Column("module", sa.String(256), nullable=True))


def downgrade() -> None:
    op.drop_column("cameras", "module")
