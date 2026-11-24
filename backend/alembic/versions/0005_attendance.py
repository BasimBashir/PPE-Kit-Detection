"""attendance (Phase 5): door crossings attributed to operator-enrolled persons, presence intervals

Revision ID: 0005
Revises: 0004
Create Date: 2026-11-24 12:00:00.000000
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB, UUID

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "persons",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(128), nullable=False, unique=True),
        sa.Column("allowed_minutes_per_day", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("gallery", JSONB(), nullable=False, server_default="[]"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_table(
        "crossings",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("idempotency_key", sa.String(64), nullable=False, unique=True),
        sa.Column("camera_id", sa.String(32), sa.ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False),
        sa.Column("zone_id", sa.String(64), sa.ForeignKey("zones.id", ondelete="CASCADE"), nullable=False),
        sa.Column("track_id", sa.Integer(), nullable=False),
        sa.Column("direction", sa.String(5), nullable=False),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("person_id", UUID(as_uuid=True), sa.ForeignKey("persons.id", ondelete="SET NULL")),
        sa.Column("match_score", sa.Float()),
        sa.Column("matched_by", sa.String(8)),
        sa.Column("descriptor", JSONB()),
        sa.Column("face", JSONB()),
        sa.Column("face_angle", sa.String(8)),
        sa.Column("image_path", sa.Text()),
        sa.Column("image_error", sa.String(32)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )
    op.create_index("ix_crossings_camera_id", "crossings", ["camera_id"])
    op.create_index("ix_crossings_zone_id", "crossings", ["zone_id"])
    op.create_index("ix_crossings_occurred_at", "crossings", ["occurred_at"])
    op.create_index("ix_crossings_person_id", "crossings", ["person_id"])
    op.create_index("ix_crossings_person_occurred", "crossings", ["person_id", "occurred_at"])
    op.create_table(
        "presence_intervals",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("person_id", UUID(as_uuid=True), sa.ForeignKey("persons.id", ondelete="CASCADE"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True)),
    )
    op.create_index("ix_presence_intervals_person_id", "presence_intervals", ["person_id"])
    op.create_index("ix_presence_person_started", "presence_intervals", ["person_id", "started_at"])


def downgrade() -> None:
    op.drop_table("presence_intervals")
    op.drop_table("crossings")
    op.drop_table("persons")
