"""Add durable chat operations and tool executions.

Revision ID: e7b2c8d4f901
Revises: d4c1a8e37b62
Create Date: 2026-09-09 19:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = 'e7b2c8d4f901'
down_revision: str | None = 'd4c1a8e37b62'
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        'chat_operation',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('client_key', sa.String(), nullable=False),
        sa.Column('request_fingerprint', sa.String(), nullable=False),
        sa.Column('chat_id', sa.String(), nullable=False),
        sa.Column('user_message_id', sa.String(), nullable=True),
        sa.Column('assistant_message_ids', sa.JSON(), nullable=False),
        sa.Column('completed_lanes', sa.JSON(), nullable=False),
        sa.Column('task_ids', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('result', sa.JSON(), nullable=True),
        sa.Column('lease_owner', sa.String(), nullable=True),
        sa.Column('lease_expires_at', sa.BigInteger(), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.BigInteger(), nullable=False),
        sa.Column('updated_at', sa.BigInteger(), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('user_id', 'client_key', name='uq_chat_operation_user_key'),
    )
    op.create_index('ix_chat_operation_user_id', 'chat_operation', ['user_id'])
    op.create_index('ix_chat_operation_chat_id', 'chat_operation', ['chat_id'])
    op.create_index('ix_chat_operation_status', 'chat_operation', ['status'])
    op.create_index('ix_chat_operation_lease_expires_at', 'chat_operation', ['lease_expires_at'])
    op.create_index('chat_operation_status_lease_idx', 'chat_operation', ['status', 'lease_expires_at'])

    op.create_table(
        'tool_execution',
        sa.Column('id', sa.String(), nullable=False),
        sa.Column('operation_id', sa.String(), nullable=False),
        sa.Column('user_id', sa.String(), nullable=False),
        sa.Column('lane_id', sa.String(), nullable=False),
        sa.Column('command_key', sa.String(), nullable=False),
        sa.Column('raw_tool_call_id', sa.String(), nullable=True),
        sa.Column('tool_name', sa.String(), nullable=False),
        sa.Column('parameters', sa.JSON(), nullable=False),
        sa.Column('source_file_refs', sa.JSON(), nullable=False),
        sa.Column('status', sa.String(), nullable=False),
        sa.Column('result', sa.JSON(), nullable=True),
        sa.Column('result_files', sa.JSON(), nullable=False),
        sa.Column('sent_at', sa.BigInteger(), nullable=True),
        sa.Column('lease_owner', sa.String(), nullable=True),
        sa.Column('lease_expires_at', sa.BigInteger(), nullable=True),
        sa.Column('version', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.BigInteger(), nullable=False),
        sa.Column('updated_at', sa.BigInteger(), nullable=False),
        sa.ForeignKeyConstraint(['operation_id'], ['chat_operation.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('operation_id', 'lane_id', 'command_key', name='uq_tool_execution_command'),
    )
    op.create_index('ix_tool_execution_operation_id', 'tool_execution', ['operation_id'])
    op.create_index('ix_tool_execution_user_id', 'tool_execution', ['user_id'])
    op.create_index('ix_tool_execution_status', 'tool_execution', ['status'])
    op.create_index('ix_tool_execution_lease_expires_at', 'tool_execution', ['lease_expires_at'])
    op.create_index(
        'uq_tool_execution_active_lane',
        'tool_execution',
        ['operation_id', 'lane_id'],
        unique=True,
        sqlite_where=sa.text("status IN ('SENDING', 'UNKNOWN', 'DELIVERY_PENDING')"),
        postgresql_where=sa.text("status IN ('SENDING', 'UNKNOWN', 'DELIVERY_PENDING')"),
    )
    op.create_index(
        'tool_execution_operation_lane_status_idx',
        'tool_execution',
        ['operation_id', 'lane_id', 'status'],
    )
    op.create_index(
        'tool_execution_operation_lane_raw_idx',
        'tool_execution',
        ['operation_id', 'lane_id', 'raw_tool_call_id'],
    )


def downgrade() -> None:
    op.drop_index('tool_execution_operation_lane_raw_idx', table_name='tool_execution')
    op.drop_index('tool_execution_operation_lane_status_idx', table_name='tool_execution')
    op.drop_index('uq_tool_execution_active_lane', table_name='tool_execution')
    op.drop_index('ix_tool_execution_lease_expires_at', table_name='tool_execution')
    op.drop_index('ix_tool_execution_status', table_name='tool_execution')
    op.drop_index('ix_tool_execution_user_id', table_name='tool_execution')
    op.drop_index('ix_tool_execution_operation_id', table_name='tool_execution')
    op.drop_table('tool_execution')
    op.drop_index('chat_operation_status_lease_idx', table_name='chat_operation')
    op.drop_index('ix_chat_operation_lease_expires_at', table_name='chat_operation')
    op.drop_index('ix_chat_operation_status', table_name='chat_operation')
    op.drop_index('ix_chat_operation_chat_id', table_name='chat_operation')
    op.drop_index('ix_chat_operation_user_id', table_name='chat_operation')
    op.drop_table('chat_operation')
