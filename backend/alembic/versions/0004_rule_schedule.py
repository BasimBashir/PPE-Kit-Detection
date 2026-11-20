"""rule shift schedule (Phase 4): ppe_rules.schedule JSONB, NULL = always enforced

Revision ID: 0004
Revises: 0003
Create Date: 2026-11-20 18:00:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("ppe_rules", sa.Column("schedule", JSONB(), nullable=True))


def downgrade() -> None:
    op.drop_column("ppe_rules", "schedule")
