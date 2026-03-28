"""Add response_feedback table.

Revision ID: 002
Revises: 001
Create Date: 2026-03-27
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "002"
down_revision: str | None = "001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "response_feedback",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("thread_id", sa.String(36), nullable=False),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("response", sa.Text(), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("timestamp", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index(
        "ix_feedback_thread_ts",
        "response_feedback",
        ["thread_id", "timestamp"],
    )


def downgrade() -> None:
    op.drop_index("ix_feedback_thread_ts", table_name="response_feedback")
    op.drop_table("response_feedback")
