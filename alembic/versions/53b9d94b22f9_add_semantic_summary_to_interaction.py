"""add_semantic_summary_to_interaction

Revision ID: 53b9d94b22f9
Revises: add_pin_and_share_tables
Create Date: 2025-10-12 00:30:45.467943

"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = "53b9d94b22f9"
down_revision: Union[str, Sequence[str], None] = "add_pin_and_share_tables"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

def upgrade() -> None:
    """Upgrade schema."""
    # Add semantic_summary JSON column to interaction table
    op.add_column(
        "interaction", sa.Column("semantic_summary", sa.JSON(), nullable=True)
    )

def downgrade() -> None:
    """Downgrade schema."""
    # Remove semantic_summary column from interaction table
    op.drop_column("interaction", "semantic_summary")
