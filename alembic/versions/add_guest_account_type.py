"""Add GUEST account type to UserAccountType enum

Revision ID: add_guest_account_type
Revises: 8e26c845b53c
Create Date: 2025-01-27 12:00:00.000000

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "add_guest_account_type"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    # Add GUEST to UserAccountType enum
    # For MySQL, we need to alter the column to include the new enum value
    # MySQL doesn't support adding enum values directly, so we modify the column
    op.execute(
        "ALTER TABLE user MODIFY COLUMN account_type ENUM('EXICUTIVE', 'SUPER_ADMIN', 'ADMIN', 'USER', 'SELLER', 'GUEST') NOT NULL DEFAULT 'USER'"
    )


def downgrade() -> None:
    """Downgrade schema."""
    # Note: PostgreSQL doesn't support removing enum values directly
    # This would require recreating the enum type, which is complex
    # For now, we'll leave it as a no-op or implement a more complex solution if needed
    pass
