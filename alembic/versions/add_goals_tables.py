"""Add goals and goal_progress tables

Revision ID: add_goals_tables
Revises: add_guest_account_type
Create Date: 2025-01-27 14:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import mysql


# revision identifiers, used by Alembic.
revision: str = "add_goals_tables"
down_revision: Union[str, Sequence[str], None] = "add_guest_account_type"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Create goal table
    op.create_table(
        "goal",
        sa.Column("id", sa.String(191), primary_key=True),
        sa.Column("user_id", sa.String(191), nullable=False),
        sa.Column("name", sa.String(500), nullable=False),
        sa.Column("icon", sa.JSON, nullable=False),
        sa.Column("color", sa.String(50), nullable=False),
        sa.Column(
            "type",
            sa.Enum("Target", "Build", "Quit", name="goaltype"),
            nullable=False,
            server_default="Target",
        ),
        sa.Column(
            "period",
            sa.Enum("daily", "weekly", name="goalperiod"),
            nullable=False,
            server_default="daily",
        ),
        sa.Column("target_value", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("unit", sa.String(50), nullable=False, server_default="times"),
        sa.Column("task_days", sa.JSON, nullable=False),
        sa.Column("reminder_enabled", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("reminder_time", sa.String(10), nullable=True),
        sa.Column("reminder_message", sa.Text(), nullable=True),
        sa.Column("show_memo", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column(
            "chart_type",
            sa.Enum("line", "bar", "pie", name="charttype"),
            nullable=False,
            server_default="line",
        ),
        sa.Column("start_date", sa.DateTime(), nullable=False),
        sa.Column("end_date", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"], ondelete="CASCADE"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    # Create goal_progress table
    op.create_table(
        "goal_progress",
        sa.Column("id", sa.String(191), primary_key=True),
        sa.Column("goal_id", sa.String(191), nullable=False),
        sa.Column("date", sa.String(10), nullable=False),  # YYYY-MM-DD format
        sa.Column("value", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("completed", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("memo", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["goal_id"], ["goal.id"], ondelete="CASCADE"),
        mysql_engine="InnoDB",
        mysql_charset="utf8mb4",
    )

    # Create indexes
    op.create_index("idx_goal_user_id", "goal", ["user_id"])
    op.create_index("idx_goal_start_date", "goal", ["start_date"])
    op.create_index("idx_goal_progress_goal_id", "goal_progress", ["goal_id"])
    op.create_index("idx_goal_progress_date", "goal_progress", ["date"])
    op.create_index(
        "idx_goal_progress_goal_date", "goal_progress", ["goal_id", "date"], unique=True
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("idx_goal_progress_goal_date", table_name="goal_progress")
    op.drop_index("idx_goal_progress_date", table_name="goal_progress")
    op.drop_index("idx_goal_progress_goal_id", table_name="goal_progress")
    op.drop_index("idx_goal_start_date", table_name="goal")
    op.drop_index("idx_goal_user_id", table_name="goal")
    op.drop_table("goal_progress")
    op.drop_table("goal")
