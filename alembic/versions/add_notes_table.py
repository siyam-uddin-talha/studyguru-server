"""Add notes table

Revision ID: add_notes_table
Revises: add_goals_tables
Create Date: 2025-01-27 15:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


# revision identifiers, used by Alembic.
revision: str = "add_notes_table"
down_revision: Union[str, Sequence[str], None] = "streamline_rag_v1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create note table
    op.create_table(
        "note",
        sa.Column("id", sa.String(191), primary_key=True),
        sa.Column("user_id", sa.String(191), nullable=False),
        sa.Column("title", sa.String(500), nullable=False, server_default="Untitled"),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("preview", sa.Text(), nullable=True),
        sa.Column("color", sa.String(50), nullable=False, server_default="#FDF2F8"),
        sa.Column("text_color", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    # Create indexes
    op.create_index("idx_note_user_id", "note", ["user_id"])
    op.create_index("idx_note_updated_at", "note", ["updated_at"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_note_updated_at", table_name="note")
    op.drop_index("idx_note_user_id", table_name="note")
    op.drop_table("note")
