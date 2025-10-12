"""Add is_pinned column and interaction share table

Revision ID: add_pin_and_share_tables
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = "add_pin_and_share_tables"
down_revision = "71fae022759e"  # References the latest migration
branch_labels = None
depends_on = None

def upgrade():
    # Add is_pinned column to interaction table
    op.add_column(
        "interaction",
        sa.Column("is_pinned", sa.Boolean(), nullable=False, server_default="0"),
    )

    # Create interaction_share table
    op.create_table(
        "interaction_share",
        sa.Column("id", sa.String(191), nullable=False),
        sa.Column("original_interaction_id", sa.String(191), nullable=False),
        sa.Column("share_id", sa.String(191), nullable=False),
        sa.Column("is_public", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("visit_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_visited_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["original_interaction_id"],
            ["interaction.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("share_id"),
    )

    # Create interaction_share_visitor table
    op.create_table(
        "interaction_share_visitor",
        sa.Column("id", sa.String(191), nullable=False),
        sa.Column("interaction_share_id", sa.String(191), nullable=False),
        sa.Column("visitor_user_id", sa.String(191), nullable=True),
        sa.Column("visitor_ip", sa.String(45), nullable=True),
        sa.Column("visitor_fingerprint", sa.String(191), nullable=True),
        sa.Column("reward_given", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("visited_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(
            ["interaction_share_id"],
            ["interaction_share.id"],
        ),
        sa.ForeignKeyConstraint(
            ["visitor_user_id"],
            ["user.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "interaction_share_id", "visitor_user_id", name="unique_user_visit"
        ),
        sa.UniqueConstraint(
            "interaction_share_id", "visitor_ip", name="unique_ip_visit"
        ),
        sa.UniqueConstraint(
            "interaction_share_id",
            "visitor_fingerprint",
            name="unique_fingerprint_visit",
        ),
    )

def downgrade():
    # Drop interaction_share_visitor table
    op.drop_table("interaction_share_visitor")

    # Drop interaction_share table
    op.drop_table("interaction_share")

    # Remove is_pinned column from interaction table
    op.drop_column("interaction", "is_pinned")
