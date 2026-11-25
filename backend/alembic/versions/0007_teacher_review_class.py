"""teacher_reviews.ppe_class: which check a reviewed crop belongs to, so the retraining flywheel (ADR-018) can sort
mined mistakes per check. Rows before this migration were hard-hat head crops.

Revision ID: 0007
Revises: 0006
Create Date: 2026-11-25 10:00:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("teacher_reviews",
                  sa.Column("ppe_class", sa.String(32), nullable=False, server_default="hard_hat"))


def downgrade() -> None:
    op.drop_column("teacher_reviews", "ppe_class")
