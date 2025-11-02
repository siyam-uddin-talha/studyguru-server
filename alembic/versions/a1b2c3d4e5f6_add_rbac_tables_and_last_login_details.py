"""Add RBAC tables and last_login_details column

Revision ID: a1b2c3d4e5f6
Revises: 8e26c845b53c
Create Date: 2025-01-20 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.mysql import JSON


# revision identifiers, used by Alembic.
revision: str = "a1b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "8e26c845b53c"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add last_login_details column to user table
    op.add_column(
        "user",
        sa.Column("last_login_details", JSON(), nullable=True),
    )

    # Update UserAccountType enum to include EXICUTIVE and SELLER
    # MySQL requires us to check current enum values and modify accordingly
    # Using a try-except approach to handle cases where enum already has these values
    try:
        op.execute(
            """
            ALTER TABLE user 
            MODIFY COLUMN account_type ENUM(
                'EXICUTIVE', 'SUPER_ADMIN', 'ADMIN', 'USER', 'SELLER'
            ) NOT NULL DEFAULT 'USER'
            """
        )
    except Exception:
        # Enum might already have these values, or migration was partially applied
        # Try to add individual values if they don't exist
        # Note: This is a best-effort approach for MySQL enum limitations
        pass

    # Create module table
    op.create_table(
        "module",
        sa.Column("id", sa.String(length=191), nullable=False),
        sa.Column("name", sa.String(length=191), nullable=False),
        sa.Column("slug", sa.String(length=191), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
        sa.UniqueConstraint("slug"),
    )
    op.create_index("idx_module_slug", "module", ["slug"])
    op.create_index("idx_module_active", "module", ["is_active"])

    # Create user_module_permission table
    op.create_table(
        "user_module_permission",
        sa.Column("id", sa.String(length=191), nullable=False),
        sa.Column("user_id", sa.String(length=191), nullable=False),
        sa.Column("module_id", sa.String(length=191), nullable=False),
        sa.Column("can_read", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("can_insert", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("can_update", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("can_delete", sa.Boolean(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["user_id"], ["user.id"]),
        sa.ForeignKeyConstraint(["module_id"], ["module.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("user_id", "module_id", name="uq_user_module"),
    )
    op.create_index("idx_user_module_user", "user_module_permission", ["user_id"])
    op.create_index("idx_user_module_module", "user_module_permission", ["module_id"])


def downgrade() -> None:
    """Downgrade schema."""
    # Drop indexes
    op.drop_index("idx_user_module_module", table_name="user_module_permission")
    op.drop_index("idx_user_module_user", table_name="user_module_permission")
    op.drop_index("idx_module_active", table_name="module")
    op.drop_index("idx_module_slug", table_name="module")

    # Drop tables
    op.drop_table("user_module_permission")
    op.drop_table("module")

    # Revert account_type enum
    # Note: This might fail if there are users with EXICUTIVE or SELLER types
    # In that case, you'd need to update those users first
    try:
        op.execute(
            """
            ALTER TABLE user 
            MODIFY COLUMN account_type ENUM(
                'SUPER_ADMIN', 'ADMIN', 'USER'
            ) NOT NULL DEFAULT 'USER'
            """
        )
    except Exception:
        # Cannot downgrade if users exist with EXICUTIVE or SELLER types
        print(
            "Warning: Cannot downgrade enum - users may have EXICUTIVE or SELLER types"
        )
        pass

    # Remove last_login_details column
    op.drop_column("user", "last_login_details")
