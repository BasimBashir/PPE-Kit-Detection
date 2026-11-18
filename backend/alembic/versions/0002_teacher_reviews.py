"""teacher reviews

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-18 12:00:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "teacher_reviews",
        sa.Column("id", sa.BigInteger(), autoincrement=True, nullable=False),
        sa.Column("event_id", sa.UUID(), nullable=True),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("camera_id", sa.String(length=32), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("crop_path", sa.Text(), nullable=False),
        sa.Column("p_model", sa.Float(), nullable=False),
        sa.Column("p_teacher", sa.Float(), nullable=False),
        sa.Column("fused", sa.Float(), nullable=False),
        sa.Column("verdict", sa.String(length=24), nullable=False),
        sa.Column("provider", sa.String(length=32), nullable=False),
        sa.Column("latency_ms", sa.Float(), nullable=False),
        sa.Column("raw", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(["event_id"], ["events.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_teacher_reviews_event_id"), "teacher_reviews", ["event_id"], unique=False)
    op.create_index(op.f("ix_teacher_reviews_verdict"), "teacher_reviews", ["verdict"], unique=False)
    op.create_index(op.f("ix_teacher_reviews_created_at"), "teacher_reviews", ["created_at"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_teacher_reviews_created_at"), table_name="teacher_reviews")
    op.drop_index(op.f("ix_teacher_reviews_verdict"), table_name="teacher_reviews")
    op.drop_index(op.f("ix_teacher_reviews_event_id"), table_name="teacher_reviews")
    op.drop_table("teacher_reviews")
