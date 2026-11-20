"""multi-PPE rules (Phase 3): mandatory flag, item-generic threshold names

Revision ID: 0003
Revises: 0002
Create Date: 2026-11-20 12:00:00.000000
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

_RENAME = (("helmet_conf", "present_conf"), ("head_conf", "absent_conf"), ("min_head_px", "min_roi_px"))


def _rename(pairs) -> None:
    for old, new in pairs:
        op.execute(f"UPDATE ppe_rules SET thresholds = (thresholds - '{old}') || "
                   f"jsonb_build_object('{new}', thresholds->'{old}') WHERE thresholds ? '{old}'")


def upgrade() -> None:
    op.add_column("ppe_rules", sa.Column("mandatory", sa.Boolean(), server_default=sa.true(), nullable=False))
    _rename(_RENAME)


def downgrade() -> None:
    _rename((new, old) for old, new in _RENAME)
    op.drop_column("ppe_rules", "mandatory")
