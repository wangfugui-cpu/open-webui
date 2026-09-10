"""Durable user operations and chargeable tool execution records."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from typing import Any

from open_webui.internal.db import Base, get_async_db_context
from sqlalchemy import JSON, BigInteger, Column, ForeignKey, Index, Integer, String, UniqueConstraint, select, update
from sqlalchemy.exc import IntegrityError

OPERATION_READY = 'READY'
OPERATION_RUNNING = 'RUNNING'
OPERATION_COMPLETED = 'COMPLETED'
OPERATION_UNKNOWN = 'UNKNOWN'
OPERATION_DELIVERY_PENDING = 'DELIVERY_PENDING'
OPERATION_FAILED = 'FAILED'

EXECUTION_NOT_SENT = 'NOT_SENT'
EXECUTION_SENDING = 'SENDING'
EXECUTION_UNKNOWN = 'UNKNOWN'
EXECUTION_DELIVERY_PENDING = 'DELIVERY_PENDING'
EXECUTION_COMPLETED = 'COMPLETED'
BLOCKING_EXECUTION_STATES = (EXECUTION_SENDING, EXECUTION_UNKNOWN, EXECUTION_DELIVERY_PENDING)


class ChatOperation(Base):
    __tablename__ = 'chat_operation'

    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    client_key = Column(String, nullable=False)
    request_fingerprint = Column(String, nullable=False)
    chat_id = Column(String, nullable=False, index=True)
    user_message_id = Column(String, nullable=True)
    assistant_message_ids = Column(JSON, nullable=False, default=list)
    completed_lanes = Column(JSON, nullable=False, default=list)
    task_ids = Column(JSON, nullable=False, default=list)
    status = Column(String, nullable=False, default=OPERATION_READY, index=True)
    result = Column(JSON, nullable=True)
    lease_owner = Column(String, nullable=True)
    lease_expires_at = Column(BigInteger, nullable=True, index=True)
    version = Column(Integer, nullable=False, default=1)
    created_at = Column(BigInteger, nullable=False, index=True)
    updated_at = Column(BigInteger, nullable=False)

    __table_args__ = (
        UniqueConstraint('user_id', 'client_key', name='uq_chat_operation_user_key'),
        Index('chat_operation_status_lease_idx', 'status', 'lease_expires_at'),
    )


class ToolExecution(Base):
    __tablename__ = 'tool_execution'

    id = Column(String, primary_key=True)
    operation_id = Column(String, ForeignKey('chat_operation.id', ondelete='CASCADE'), nullable=False, index=True)
    user_id = Column(String, nullable=False, index=True)
    lane_id = Column(String, nullable=False)
    command_key = Column(String, nullable=False)
    raw_tool_call_id = Column(String, nullable=True)
    tool_name = Column(String, nullable=False)
    parameters = Column(JSON, nullable=False, default=dict)
    source_file_refs = Column(JSON, nullable=False, default=list)
    status = Column(String, nullable=False, default=EXECUTION_NOT_SENT, index=True)
    result = Column(JSON, nullable=True)
    result_files = Column(JSON, nullable=False, default=list)
    sent_at = Column(BigInteger, nullable=True)
    lease_owner = Column(String, nullable=True)
    lease_expires_at = Column(BigInteger, nullable=True, index=True)
    version = Column(Integer, nullable=False, default=1)
    created_at = Column(BigInteger, nullable=False, index=True)
    updated_at = Column(BigInteger, nullable=False)

    __table_args__ = (
        UniqueConstraint('operation_id', 'lane_id', 'command_key', name='uq_tool_execution_command'),
        # A lane can have only one potentially-paid execution.  This is a
        # partial unique index so completed/not-sent commands do not prevent a
        # later, distinct legitimate tool call in the same model answer.
        Index(
            'uq_tool_execution_active_lane',
            'operation_id',
            'lane_id',
            unique=True,
            sqlite_where=status.in_(BLOCKING_EXECUTION_STATES),
            postgresql_where=status.in_(BLOCKING_EXECUTION_STATES),
        ),
        Index('tool_execution_operation_lane_status_idx', 'operation_id', 'lane_id', 'status'),
        Index('tool_execution_operation_lane_raw_idx', 'operation_id', 'lane_id', 'raw_tool_call_id'),
    )


@dataclass(frozen=True)
class OperationClaim:
    operation: ChatOperation
    owner_token: str | None
    owns_launch: bool
    conflict: bool = False


@dataclass(frozen=True)
class LaunchClaim:
    """The result of this caller's READY -> RUNNING transition."""

    operation: ChatOperation | None
    owns_launch: bool


@dataclass(frozen=True)
class ExecutionClaim:
    execution: ToolExecution | None
    action: str
    content: str | None = None


def _now() -> int:
    return int(time.time())


def _as_list(value: Any) -> list:
    return value if isinstance(value, list) else []


def _pending_lane_ids(operation: ChatOperation, completed_lanes: list[str] | None = None) -> list[str]:
    """Expected model lanes which have not reached their terminal callback."""
    completed = set(completed_lanes if completed_lanes is not None else _as_list(operation.completed_lanes))
    return [lane for lane in _as_list(operation.assistant_message_ids) if lane not in completed]


def _unfinished_operation_state(
    operation: ChatOperation,
    executions: list[ToolExecution],
    completed_lanes: list[str] | None = None,
) -> str | None:
    """Return the durable aggregate state of both model lanes and image work."""
    states = {execution.status for execution in executions}
    if EXECUTION_UNKNOWN in states:
        return OPERATION_UNKNOWN
    # A queued/running model lane may not have reached its first tool call.
    # Its absence from tool_execution is not evidence that it finished.
    if _pending_lane_ids(operation, completed_lanes):
        return OPERATION_RUNNING
    if EXECUTION_DELIVERY_PENDING in states:
        return OPERATION_DELIVERY_PENDING
    if EXECUTION_SENDING in states:
        return OPERATION_RUNNING
    return None


def _operation_result(
    operation: ChatOperation,
    executions: list[ToolExecution],
    *,
    incomplete: bool,
    completed_lanes: list[str] | None = None,
) -> dict:
    lanes = _as_list(operation.assistant_message_ids)
    pending_lanes = _pending_lane_ids(operation, completed_lanes)
    result = {
        'chat_id': operation.chat_id,
        'assistant_message_id': lanes[0] if len(lanes) == 1 else None,
        'assistant_message_ids': lanes,
        'files': [item for execution in executions for item in _as_list(execution.result_files)],
    }
    if incomplete or pending_lanes:
        result['code'] = 'operation_incomplete'
        result['pending_lanes'] = pending_lanes
        result['pending_executions'] = [
            {
                'lane_id': execution.lane_id,
                'tool_name': execution.tool_name,
                'status': execution.status,
            }
            for execution in executions
            if execution.status in BLOCKING_EXECUTION_STATES
        ]
    return result


def request_fingerprint(
    *,
    model_id: str | None,
    chat_scope: str | None,
    parent_id: str | None,
    user_message: dict | None,
    assistant_message_ids: list[dict],
    form_data: dict,
) -> str:
    """Hash only execution-relevant client intent, excluding connection/session state."""

    def message_identity(message: dict | None) -> dict:
        message = message or {}
        return {
            'id': message.get('id'),
            'parent_id': message.get('parentId'),
            'content': message.get('content'),
            'files': message.get('files') or [],
        }

    payload = {
        'chat_scope': chat_scope,
        'model': model_id,
        'parent_id': parent_id,
        'user_message': message_identity(user_message),
        'assistant_messages': assistant_message_ids,
        'messages': form_data.get('messages') or [],
        'params': form_data.get('params') or {},
        'files': form_data.get('files') or [],
        'features': form_data.get('features') or {},
        'variables': form_data.get('variables') or {},
        'chat_variables': form_data.get('chat_variables') or {},
        'model_item': form_data.get('model_item') or {},
        'tool_ids': form_data.get('tool_ids') or [],
        'skill_ids': form_data.get('skill_ids') or [],
        'filter_ids': form_data.get('filter_ids') or [],
        'tool_servers': form_data.get('tool_servers') or [],
        'terminal_id': form_data.get('terminal_id'),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def operation_response(operation: ChatOperation) -> dict:
    response = {
        'status': True,
        'task_ids': _as_list(operation.task_ids),
        'chat_id': operation.chat_id,
        'operation_id': operation.id,
        # READY is intentionally visible: the operation owns the request key
        # but no task has been persisted yet, so reporting RUNNING would make
        # a replay look as though it had a task to attach to.
        'operation_status': operation.status,
    }
    if operation.result is not None:
        response['result'] = operation.result
    return response


class OperationsTable:
    async def acquire(
        self,
        *,
        user_id: str,
        client_key: str,
        fingerprint: str,
        chat_id: str,
        user_message_id: str | None,
        assistant_message_ids: list[str],
        lease_seconds: int = 120,
    ) -> OperationClaim:
        now = _now()
        owner = str(uuid.uuid4())
        operation = ChatOperation(
            id=str(uuid.uuid4()),
            user_id=user_id,
            client_key=client_key,
            request_fingerprint=fingerprint,
            chat_id=chat_id,
            user_message_id=user_message_id,
            assistant_message_ids=assistant_message_ids,
            completed_lanes=[],
            task_ids=[],
            status=OPERATION_READY,
            lease_owner=owner,
            lease_expires_at=now + lease_seconds,
            version=1,
            created_at=now,
            updated_at=now,
        )
        try:
            async with get_async_db_context() as db:
                db.add(operation)
                await db.commit()
                await db.refresh(operation)
            return OperationClaim(operation=operation, owner_token=owner, owns_launch=True)
        except IntegrityError:
            pass

        async with get_async_db_context() as db:
            await db.rollback()
            existing = await db.scalar(
                select(ChatOperation).where(
                    ChatOperation.user_id == user_id,
                    ChatOperation.client_key == client_key,
                )
            )
            if existing is None:
                return OperationClaim(operation=operation, owner_token=None, owns_launch=False)
            if existing.request_fingerprint != fingerprint:
                return OperationClaim(operation=existing, owner_token=None, owns_launch=False, conflict=True)

            if existing.status == OPERATION_READY and (existing.lease_expires_at or 0) <= now:
                claimed = await db.execute(
                    update(ChatOperation)
                    .where(
                        ChatOperation.id == existing.id,
                        ChatOperation.status == OPERATION_READY,
                        ChatOperation.version == existing.version,
                        ChatOperation.lease_expires_at <= now,
                    )
                    .values(
                        lease_owner=owner,
                        lease_expires_at=now + lease_seconds,
                        version=existing.version + 1,
                        updated_at=now,
                    )
                )
                await db.commit()
                if claimed.rowcount == 1:
                    return OperationClaim(
                        operation=await db.get(ChatOperation, existing.id),
                        owner_token=owner,
                        owns_launch=True,
                    )

            if existing.status == OPERATION_RUNNING and (existing.lease_expires_at or 0) <= now:
                await self._mark_expired_running_unknown(db, existing, now)
                await db.commit()
                existing = await db.get(ChatOperation, existing.id)

            return OperationClaim(operation=existing, owner_token=None, owns_launch=False)

    async def _mark_expired_running_unknown(self, db, operation: ChatOperation, now: int) -> None:
        claimed = await db.execute(
            update(ChatOperation)
            .where(
                ChatOperation.id == operation.id,
                ChatOperation.status == OPERATION_RUNNING,
                ChatOperation.version == operation.version,
                ChatOperation.lease_expires_at <= now,
            )
            .values(
                status=OPERATION_UNKNOWN,
                lease_owner=None,
                lease_expires_at=None,
                version=operation.version + 1,
                updated_at=now,
                result={
                    'code': 'operation_lease_expired',
                    'message': (
                        'Execution ownership expired; paid work may already have been sent and requires confirmation.'
                    ),
                },
            )
        )
        if claimed.rowcount:
            await db.execute(
                update(ToolExecution)
                .where(ToolExecution.operation_id == operation.id, ToolExecution.status == EXECUTION_SENDING)
                .values(
                    status=EXECUTION_UNKNOWN,
                    lease_owner=None,
                    lease_expires_at=None,
                    updated_at=now,
                )
            )

    async def mark_running(self, operation_id: str, owner_token: str, task_ids: list[str]) -> LaunchClaim:
        """Atomically take task-start ownership, without trusting another worker's state."""
        now = _now()
        async with get_async_db_context() as db:
            started = await db.execute(
                update(ChatOperation)
                .where(
                    ChatOperation.id == operation_id,
                    ChatOperation.status == OPERATION_READY,
                    ChatOperation.lease_owner == owner_token,
                    ChatOperation.lease_expires_at > now,
                )
                .values(
                    status=OPERATION_RUNNING,
                    task_ids=task_ids,
                    lease_expires_at=now + 300,
                    version=ChatOperation.version + 1,
                    updated_at=now,
                )
            )
            await db.commit()
            return LaunchClaim(await db.get(ChatOperation, operation_id), bool(started.rowcount))

    async def release_ready(self, operation_id: str, owner_token: str) -> None:
        """A pre-dispatch failure is known not to have sent paid work, so release it immediately."""
        async with get_async_db_context() as db:
            await db.execute(
                update(ChatOperation)
                .where(
                    ChatOperation.id == operation_id,
                    ChatOperation.status == OPERATION_READY,
                    ChatOperation.lease_owner == owner_token,
                )
                .values(lease_expires_at=0, updated_at=_now(), version=ChatOperation.version + 1)
            )
            await db.commit()

    async def get(self, operation_id: str) -> ChatOperation | None:
        async with get_async_db_context() as db:
            return await db.get(ChatOperation, operation_id)

    async def get_execution(self, execution_id: str) -> ToolExecution | None:
        async with get_async_db_context() as db:
            return await db.get(ToolExecution, execution_id)

    async def _claim_existing_command(
        self,
        db,
        *,
        operation_id: str,
        lane_id: str,
        raw_tool_call_id: str | None,
        command_key: str,
        tool_name: str,
        parameters: dict,
    ) -> ExecutionClaim | None:
        if not raw_tool_call_id:
            return None
        existing = await db.scalar(
            select(ToolExecution).where(
                ToolExecution.operation_id == operation_id,
                ToolExecution.lane_id == lane_id,
                ToolExecution.command_key == command_key,
            )
        )
        if existing is None:
            return None
        if existing.tool_name != tool_name or (existing.parameters or {}) != parameters:
            return ExecutionClaim(
                None,
                'block',
                'The persisted tool command conflicts with this tool_call_id.',
            )
        if existing.status == EXECUTION_COMPLETED:
            return ExecutionClaim(existing, 'reuse')
        return ExecutionClaim(existing, 'block', self._pending_message(existing.status))

    async def claim_image_execution(
        self,
        *,
        operation_id: str,
        owner_token: str,
        lane_id: str,
        raw_tool_call_id: str | None,
        tool_name: str,
        parameters: dict,
        source_file_refs: list[dict],
        locally_invalid: str | None = None,
    ) -> ExecutionClaim:
        now = _now()
        command_key = raw_tool_call_id or str(uuid.uuid4())
        async with get_async_db_context() as db:
            operation = await db.get(ChatOperation, operation_id)
            if operation is None or operation.lease_owner != owner_token or operation.status != OPERATION_RUNNING:
                return ExecutionClaim(None, 'block', 'Operation is no longer owned by this worker.')
            if (operation.lease_expires_at or 0) <= now:
                # A worker may wake up after its lease has elapsed without a
                # client replay arriving first. Do not let that stale worker
                # send a new paid request; make the uncertainty durable.
                await self._mark_expired_running_unknown(db, operation, now)
                await db.commit()
                return ExecutionClaim(
                    None,
                    'block',
                    'Operation ownership expired; confirmation is required before another image call.',
                )

            existing_claim = await self._claim_existing_command(
                db,
                operation_id=operation_id,
                lane_id=lane_id,
                raw_tool_call_id=raw_tool_call_id,
                command_key=command_key,
                tool_name=tool_name,
                parameters=parameters,
            )
            if existing_claim is not None:
                return existing_claim

            # A missing source image is a local validation failure of an
            # *editing* request, not permission to silently fulfill a
            # different request by generating a replacement image.  The
            # invalid edit is durable precisely so a later model turn cannot
            # evade that intent by changing its tool name or call id.  A
            # subsequent valid edit remains allowed, and a new user operation
            # has its own operation id so independent text-to-image requests
            # are unaffected.
            if tool_name == 'generate_image':
                invalid_edit = await db.scalar(
                    select(ToolExecution).where(
                        ToolExecution.operation_id == operation_id,
                        ToolExecution.lane_id == lane_id,
                        ToolExecution.tool_name == 'edit_image',
                        ToolExecution.status == EXECUTION_NOT_SENT,
                    )
                )
                if invalid_edit is not None:
                    return ExecutionClaim(
                        invalid_edit,
                        'block',
                        (
                            'An image source is required to continue editing; '
                            'submit a new request to generate a new image.'
                        ),
                    )

            blocking = await db.scalar(
                select(ToolExecution).where(
                    ToolExecution.operation_id == operation_id,
                    ToolExecution.lane_id == lane_id,
                    ToolExecution.status.in_(BLOCKING_EXECUTION_STATES),
                )
            )
            if blocking is not None:
                return ExecutionClaim(blocking, 'block', self._pending_message(blocking.status))

            state = EXECUTION_NOT_SENT if locally_invalid else EXECUTION_SENDING
            execution = ToolExecution(
                id=str(uuid.uuid4()),
                operation_id=operation_id,
                user_id=operation.user_id,
                lane_id=lane_id,
                command_key=command_key,
                raw_tool_call_id=raw_tool_call_id,
                tool_name=tool_name,
                parameters=parameters,
                source_file_refs=source_file_refs,
                status=state,
                result={'error': locally_invalid} if locally_invalid else None,
                result_files=[],
                sent_at=now if state == EXECUTION_SENDING else None,
                lease_owner=owner_token if state == EXECUTION_SENDING else None,
                lease_expires_at=operation.lease_expires_at if state == EXECUTION_SENDING else None,
                version=1,
                created_at=now,
                updated_at=now,
            )
            db.add(execution)
            try:
                await db.commit()
            except IntegrityError:
                # The partial unique index is the atomic lane claim.  A
                # concurrent new tool_call_id must observe the winner and
                # stop before it can call a paid upstream.
                await db.rollback()
                blocking = await db.scalar(
                    select(ToolExecution).where(
                        ToolExecution.operation_id == operation_id,
                        ToolExecution.lane_id == lane_id,
                        ToolExecution.status.in_(BLOCKING_EXECUTION_STATES),
                    )
                )
                if blocking is not None:
                    return ExecutionClaim(blocking, 'block', self._pending_message(blocking.status))
                return ExecutionClaim(None, 'block', 'Another image execution is being recorded.')
            await db.refresh(execution)
            return ExecutionClaim(execution, 'local_error' if locally_invalid else 'execute', locally_invalid)

    async def mark_execution_completed(
        self, execution_id: str, owner_token: str, result: Any, result_files: list[dict]
    ) -> ToolExecution | None:
        return await self._set_execution(
            execution_id,
            owner_token,
            EXECUTION_COMPLETED,
            result=result,
            result_files=result_files,
        )

    async def mark_execution_unknown(self, execution_id: str, owner_token: str, error: str) -> ToolExecution | None:
        return await self._set_execution_and_operation_state(
            execution_id,
            owner_token,
            execution_state=EXECUTION_UNKNOWN,
            operation_state=OPERATION_UNKNOWN,
            result={'error': error, 'code': 'result_unknown'},
        )

    async def mark_delivery_pending(
        self,
        execution_id: str | None,
        owner_token: str | None,
        result_files: list[dict],
        error: str,
    ) -> ToolExecution | None:
        if not execution_id or not owner_token:
            return None
        now = _now()
        result = {'error': error, 'code': 'delivery_pending'}
        async with get_async_db_context() as db:
            execution = await db.get(ToolExecution, execution_id)
            if execution is None or execution.lease_owner != owner_token:
                return None
            operation = await db.get(ChatOperation, execution.operation_id)
            if operation is None or operation.lease_owner != owner_token or operation.status != OPERATION_RUNNING:
                return None

            # The calling model lane still has to enter its terminal callback
            # after delivery fails, and sibling model lanes may not yet have
            # called any tool at all.  Keep the shared lease until lane
            # completion has made those facts durable; otherwise a pending
            # image from A would revoke B's legal right to continue.
            operation_state = OPERATION_RUNNING
            execution_updated = await db.execute(
                update(ToolExecution)
                .where(
                    ToolExecution.id == execution_id,
                    ToolExecution.lease_owner == owner_token,
                    ToolExecution.version == execution.version,
                )
                .values(
                    status=EXECUTION_DELIVERY_PENDING,
                    result=result,
                    result_files=result_files,
                    lease_owner=None,
                    lease_expires_at=None,
                    version=execution.version + 1,
                    updated_at=now,
                )
            )
            operation_updated = await db.execute(
                update(ChatOperation)
                .where(
                    ChatOperation.id == operation.id,
                    ChatOperation.lease_owner == owner_token,
                    ChatOperation.status == OPERATION_RUNNING,
                    ChatOperation.version == operation.version,
                )
                .values(
                    status=operation_state,
                    result=result,
                    lease_owner=operation.lease_owner,
                    lease_expires_at=operation.lease_expires_at,
                    version=operation.version + 1,
                    updated_at=now,
                )
            )
            if not execution_updated.rowcount or not operation_updated.rowcount:
                await db.rollback()
                return None
            await db.commit()
            return await db.get(ToolExecution, execution_id)

    async def _set_execution_and_operation_state(
        self,
        execution_id: str,
        owner_token: str,
        *,
        execution_state: str,
        operation_state: str,
        result: dict,
        result_files: list[dict] | None = None,
    ) -> ToolExecution | None:
        """Persist a blocking execution outcome and its operation atomically.

        The operation is the entry-point replay record, while the execution is
        the delivery recovery record.  They cannot be committed separately:
        otherwise a crash could leave a saved image invisible behind an
        ``UNKNOWN`` operation and make recovery unreachable.
        """
        now = _now()
        async with get_async_db_context() as db:
            execution = await db.get(ToolExecution, execution_id)
            if execution is None or execution.lease_owner != owner_token:
                return None
            operation = await db.get(ChatOperation, execution.operation_id)
            if operation is None or operation.lease_owner != owner_token or operation.status != OPERATION_RUNNING:
                return None

            execution_updated = await db.execute(
                update(ToolExecution)
                .where(
                    ToolExecution.id == execution_id,
                    ToolExecution.lease_owner == owner_token,
                    ToolExecution.version == execution.version,
                )
                .values(
                    status=execution_state,
                    result=result,
                    result_files=result_files if result_files is not None else execution.result_files,
                    lease_owner=None,
                    lease_expires_at=None,
                    version=execution.version + 1,
                    updated_at=now,
                )
            )
            operation_updated = await db.execute(
                update(ChatOperation)
                .where(
                    ChatOperation.id == operation.id,
                    ChatOperation.lease_owner == owner_token,
                    ChatOperation.status == OPERATION_RUNNING,
                    ChatOperation.version == operation.version,
                )
                .values(
                    status=operation_state,
                    result=result,
                    lease_owner=None,
                    lease_expires_at=None,
                    version=operation.version + 1,
                    updated_at=now,
                )
            )
            if not execution_updated.rowcount or not operation_updated.rowcount:
                await db.rollback()
                return None
            await db.commit()
            return await db.get(ToolExecution, execution_id)

    async def _set_execution(
        self,
        execution_id: str,
        owner_token: str,
        state: str,
        *,
        result: Any,
        result_files: list[dict] | None = None,
    ) -> ToolExecution | None:
        now = _now()
        async with get_async_db_context() as db:
            execution = await db.get(ToolExecution, execution_id)
            if execution is None or execution.lease_owner != owner_token:
                return None
            updated = await db.execute(
                update(ToolExecution)
                .where(
                    ToolExecution.id == execution_id,
                    ToolExecution.lease_owner == owner_token,
                    ToolExecution.version == execution.version,
                )
                .values(
                    status=state,
                    result=result,
                    result_files=result_files if result_files is not None else execution.result_files,
                    lease_owner=None,
                    lease_expires_at=None,
                    version=execution.version + 1,
                    updated_at=now,
                )
            )
            await db.commit()
            return await db.get(ToolExecution, execution_id) if updated.rowcount else None

    async def _set_operation_state(
        self, operation_id: str, owner_token: str, state: str, result: dict
    ) -> None:
        async with get_async_db_context() as db:
            await db.execute(
                update(ChatOperation)
                .where(
                    ChatOperation.id == operation_id,
                    ChatOperation.lease_owner == owner_token,
                    ChatOperation.status == OPERATION_RUNNING,
                )
                .values(status=state, result=result, updated_at=_now(), version=ChatOperation.version + 1)
            )
            await db.commit()

    async def _stopped_lane_state(
        self,
        db,
        operation: ChatOperation,
        executions: list[ToolExecution],
        completed_lanes: list[str],
        now: int,
        result_unknown: bool = False,
    ) -> tuple[str, dict]:
        if result_unknown:
            # The model stream was interrupted after the request had already
            # been dispatched.  There may be no ToolExecution yet (a native
            # tool call is only claimed after the stream terminates), but a
            # replay must not turn that uncertain request into another paid
            # attempt.
            await db.execute(
                update(ToolExecution)
                .where(
                    ToolExecution.operation_id == operation.id,
                    ToolExecution.status == EXECUTION_SENDING,
                )
                .values(
                    status=EXECUTION_UNKNOWN,
                    lease_owner=None,
                    lease_expires_at=None,
                    version=ToolExecution.version + 1,
                    updated_at=now,
                )
            )
            unknown_executions = (
                await db.scalars(select(ToolExecution).where(ToolExecution.operation_id == operation.id))
            ).all()
            result = _operation_result(
                operation,
                unknown_executions,
                incomplete=True,
                completed_lanes=completed_lanes,
            )
            result.update(
                {
                    'code': 'operation_result_unknown',
                    'message': (
                        'The upstream model stream ended before its result could be confirmed; '
                        'confirmation is required before retrying.'
                    ),
                }
            )
            return OPERATION_UNKNOWN, result

        if not any(execution.status == EXECUTION_SENDING for execution in executions):
            return (
                OPERATION_FAILED,
                {
                    'code': 'operation_failed',
                    'message': 'The model lane failed before completion.',
                },
            )

        # A canceled/failed model task cannot prove that an already-dispatched
        # image request did not reach its provider.  Persist uncertainty
        # rather than presenting an ordinary model failure as final.
        await db.execute(
            update(ToolExecution)
            .where(
                ToolExecution.operation_id == operation.id,
                ToolExecution.status == EXECUTION_SENDING,
            )
            .values(
                status=EXECUTION_UNKNOWN,
                lease_owner=None,
                lease_expires_at=None,
                version=ToolExecution.version + 1,
                updated_at=now,
            )
        )
        unknown_executions = (
            await db.scalars(select(ToolExecution).where(ToolExecution.operation_id == operation.id))
        ).all()
        result = _operation_result(
            operation,
            unknown_executions,
            incomplete=True,
            completed_lanes=completed_lanes,
        )
        result.update(
            {
                'code': 'operation_result_unknown',
                'message': (
                    'A model lane stopped while an image request may already have been sent; '
                    'confirmation is required.'
                ),
            }
        )
        return OPERATION_UNKNOWN, result

    async def finish_lane(
        self,
        operation_id: str,
        owner_token: str,
        lane_id: str,
        succeeded: bool,
        result_unknown: bool = False,
    ) -> ChatOperation | None:
        """Record a completed model lane without letting a stale worker win.

        Separate model tasks finish independently.  A JSON list is not a
        mutable SQLAlchemy value, so mutate a copy and use the operation
        version as compare-and-swap protection.  That gives a losing lane a
        chance to reload the winner's list instead of silently dropping it.
        """
        for _ in range(4):
            async with get_async_db_context() as db:
                operation = await db.get(ChatOperation, operation_id)
                if operation is None or operation.lease_owner != owner_token:
                    return operation

                now = _now()
                if operation.status == OPERATION_RUNNING and (operation.lease_expires_at or 0) <= now:
                    # The owner no longer has authority to make a late success
                    # visible.  It may have sent paid work, so fail closed.
                    await self._mark_expired_running_unknown(db, operation, now)
                    await db.commit()
                    return await db.get(ChatOperation, operation_id)

                completed_lanes = list(_as_list(operation.completed_lanes))
                if lane_id not in completed_lanes:
                    completed_lanes.append(lane_id)

                next_status = operation.status
                next_result = operation.result
                next_owner = operation.lease_owner
                next_expiry = operation.lease_expires_at
                executions = (
                    await db.scalars(select(ToolExecution).where(ToolExecution.operation_id == operation_id))
                ).all()
                all_lanes_finished = not _pending_lane_ids(operation, completed_lanes)
                next_result = _operation_result(
                    operation, executions, incomplete=True, completed_lanes=completed_lanes
                )

                if not succeeded and operation.status == OPERATION_RUNNING:
                    next_status, next_result = await self._stopped_lane_state(
                        db,
                        operation,
                        executions,
                        completed_lanes,
                        now,
                        result_unknown=result_unknown,
                    )
                    next_owner = None
                    next_expiry = None
                elif operation.status == OPERATION_RUNNING and all_lanes_finished:
                    unfinished_state = _unfinished_operation_state(operation, executions, completed_lanes)
                    if unfinished_state is None:
                        next_status = OPERATION_COMPLETED
                        next_result = _operation_result(
                            operation, executions, incomplete=False, completed_lanes=completed_lanes
                        )
                        next_owner = None
                        next_expiry = None
                    else:
                        next_status = unfinished_state
                        next_result = _operation_result(
                            operation, executions, incomplete=True, completed_lanes=completed_lanes
                        )
                        if unfinished_state != OPERATION_RUNNING:
                            next_owner = None
                            next_expiry = None

                claimed = await db.execute(
                    update(ChatOperation)
                    .where(
                        ChatOperation.id == operation_id,
                        ChatOperation.lease_owner == owner_token,
                        ChatOperation.version == operation.version,
                    )
                    .values(
                        completed_lanes=completed_lanes,
                        status=next_status,
                        result=next_result,
                        lease_owner=next_owner,
                        lease_expires_at=next_expiry,
                        version=operation.version + 1,
                        updated_at=now,
                    )
                )
                await db.commit()
                if claimed.rowcount:
                    return await db.get(ChatOperation, operation_id)

        # A concurrent winner made progress every time.  Return its durable
        # state rather than attempting a fifth write with stale ownership.
        return await self.get(operation_id)

    async def recover_delivery(self, operation_id: str) -> ChatOperation | None:
        """Attach durable image files without running a model or image provider again."""
        from open_webui.models.chats import Chats
        from open_webui.models.files import Files

        async with get_async_db_context() as db:
            operation = await db.get(ChatOperation, operation_id)
            if operation is None:
                return None
            pending = (
                await db.scalars(
                    select(ToolExecution).where(
                        ToolExecution.operation_id == operation_id,
                        ToolExecution.status == EXECUTION_DELIVERY_PENDING,
                    )
                )
            ).all()

        recovered_delivery = False
        for execution in pending:
            files = _as_list(execution.result_files)
            if not files:
                continue
            owned = [await Files.get_file_by_id_and_user_id(item.get('id', ''), execution.user_id) for item in files]
            if len([file for file in owned if file]) != len(files):
                continue
            attached = await Chats.add_message_files_by_id_and_message_id(operation.chat_id, execution.lane_id, files)
            if attached is None:
                continue
            async with get_async_db_context() as db:
                current = await db.get(ToolExecution, execution.id)
                if current and current.status == EXECUTION_DELIVERY_PENDING:
                    recovered = await db.execute(
                        update(ToolExecution)
                        .where(
                            ToolExecution.id == current.id,
                            ToolExecution.status == EXECUTION_DELIVERY_PENDING,
                            ToolExecution.version == current.version,
                        )
                        .values(
                            status=EXECUTION_COMPLETED,
                            result={'status': 'success', 'images': files},
                            updated_at=_now(),
                            version=current.version + 1,
                        )
                    )
                    await db.commit()
                    recovered_delivery = bool(recovered.rowcount)

        # ``UNKNOWN`` normally means an upstream request may have been sent
        # without a result and must stay blocked.  It becomes completable only
        # when this call actually recovered a durable delivery record.  A
        # delivery-pending operation can additionally finish a prior recovery
        # that crashed after the execution row committed.
        if recovered_delivery:
            return await self._complete_after_delivery(operation_id, allow_unknown=True, allow_running=True)
        if operation.status == OPERATION_DELIVERY_PENDING:
            return await self._complete_after_delivery(operation_id)
        return await self.get(operation_id)

    async def _complete_after_delivery(
        self, operation_id: str, *, allow_unknown: bool = False, allow_running: bool = False
    ) -> ChatOperation | None:
        allowed_states = [OPERATION_DELIVERY_PENDING]
        if allow_unknown:
            allowed_states.append(OPERATION_UNKNOWN)
        if allow_running:
            allowed_states.append(OPERATION_RUNNING)
        allowed_states = tuple(allowed_states)

        # ``finish_lane`` and recovery can both update the operation summary.
        # A lost compare-and-swap must reload and aggregate once more, so the
        # response does not retain a stale pending-execution summary even
        # though its status was safe.
        for _ in range(3):
            async with get_async_db_context() as db:
                operation = await db.get(ChatOperation, operation_id)
                if operation is None or operation.status not in allowed_states:
                    return operation
                now = _now()
                if operation.status == OPERATION_RUNNING and (operation.lease_expires_at or 0) <= now:
                    await self._mark_expired_running_unknown(db, operation, now)
                    await db.commit()
                    return await db.get(ChatOperation, operation_id)
                executions = (
                    await db.scalars(select(ToolExecution).where(ToolExecution.operation_id == operation_id))
                ).all()
                unfinished_state = _unfinished_operation_state(operation, executions)
                next_status = unfinished_state or OPERATION_COMPLETED
                result = _operation_result(operation, executions, incomplete=unfinished_state is not None)
                # Delivery recovery never re-runs model inference.  If any other
                # execution remains nonterminal, retain its durable state rather
                # than presenting the recovered file as the entire operation.
                next_owner = operation.lease_owner if next_status == OPERATION_RUNNING else None
                next_expiry = operation.lease_expires_at if next_status == OPERATION_RUNNING else None
                if next_status == OPERATION_RUNNING and (not next_owner or not next_expiry):
                    next_status = OPERATION_UNKNOWN
                    next_owner = None
                    next_expiry = None
                updated = await db.execute(
                    update(ChatOperation)
                    .where(
                        ChatOperation.id == operation_id,
                        ChatOperation.status.in_(allowed_states),
                        ChatOperation.version == operation.version,
                    )
                    .values(
                        status=next_status,
                        result=result,
                        lease_owner=next_owner,
                        lease_expires_at=next_expiry,
                        updated_at=_now(),
                        version=operation.version + 1,
                    )
                )
                await db.commit()
                if updated.rowcount:
                    return await db.get(ChatOperation, operation_id, populate_existing=True)

        return await self.get(operation_id)

    @staticmethod
    def _pending_message(state: str) -> str:
        return (
            'A prior paid image execution is awaiting delivery recovery.'
            if state == EXECUTION_DELIVERY_PENDING
            else 'A prior paid image execution may have been sent; confirmation is required before another image call.'
        )


Operations = OperationsTable()
