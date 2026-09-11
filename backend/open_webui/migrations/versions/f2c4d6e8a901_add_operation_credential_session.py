"""Bind durable chat operations to their originating key session.

Revision ID: f2c4d6e8a901
Revises: e7b2c8d4f901
Create Date: 2026-09-11 11:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'f2c4d6e8a901'
down_revision: str | None = 'e7b2c8d4f901'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column('chat_operation', sa.Column('credential_session_id', sa.String(), nullable=True))


def downgrade() -> None:
    op.drop_column('chat_operation', 'credential_session_id')
