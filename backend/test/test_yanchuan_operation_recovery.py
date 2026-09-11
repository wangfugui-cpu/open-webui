"""Recovery, ownership, and request-key checks for durable YanChuan operations."""

from __future__ import annotations

import asyncio

import pytest
from open_webui.models.chats import Chats
from open_webui.models.files import File, FileForm, Files
from open_webui.models.operations import (
    EXECUTION_DELIVERY_PENDING,
    EXECUTION_SENDING,
    EXECUTION_UNKNOWN,
    OPERATION_COMPLETED,
    OPERATION_FAILED,
    OPERATION_RUNNING,
    OPERATION_UNKNOWN,
    ChatOperation,
    Operations,
    ToolExecution,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from test_yanchuan_execution_reliability import _create_empty_chat, _edit_call, _EntryHarness, _payload, main


@pytest.mark.asyncio
async def test_replay_recovers_saved_file_delivery_without_a_second_image_call(
    isolated_runtime_db, tmp_path, monkeypatch
):
    harness = _EntryHarness(
        isolated_runtime_db,
        tmp_path,
        monkeypatch,
        fail_message_association=True,
        router_backed_image_provider=True,
    )
    chat_id = 'delivery-recovery-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)
    body = _payload(
        harness.model,
        chat_id=chat_id,
        user_message_id='delivery-recovery-user',
        assistant_message_id='delivery-recovery-assistant',
    )
    headers = {'Idempotency-Key': 'delivery-recovery-operation'}

    first_response = await harness.submit(body, headers=headers)
    await harness.wait_for_tasks()

    async with AsyncSession(isolated_runtime_db) as session:
        operation = await session.get(ChatOperation, first_response['operation_id'])
        execution = await session.scalar(
            select(ToolExecution).where(ToolExecution.operation_id == first_response['operation_id'])
        )
        assert operation is not None
        assert operation.status == 'DELIVERY_PENDING'
        # The model stopped after the image was durably saved; delivery is
        # pending, but the lane itself has reached its terminal callback.
        assert operation.completed_lanes == ['delivery-recovery-assistant']
        assert execution is not None
        assert execution.status == EXECUTION_DELIVERY_PENDING
        file_id = execution.result_files[0]['id']
        saved_file = await session.get(File, file_id)
        assert saved_file is not None
        assert saved_file.user_id == harness.user.id

    assert await Files.get_file_by_id_and_user_id(file_id, harness.user.id) is not None
    chat = await Chats.get_chat_by_id(chat_id)
    assert chat is not None
    assert 'delivery-recovery-assistant' in chat.chat['history']['messages']

    # This is the old two-commit crash shape: the file execution says it can
    # be delivered but the entry operation looks unknown.  New writes commit
    # both rows atomically; this assertion keeps the replay-side recovery
    # backstop for any already-partial local record.
    async with AsyncSession(isolated_runtime_db) as session:
        operation = await session.get(ChatOperation, first_response['operation_id'])
        assert operation is not None
        operation.status = OPERATION_UNKNOWN
        operation.result = {'code': 'operation_lease_expired'}
        await session.commit()

    replay_response = await harness.submit(body, headers=headers)

    observed = await harness.observe(chat_id)
    assert replay_response['operation_id'] == first_response['operation_id']
    assert replay_response['operation_status'] == OPERATION_COMPLETED
    assert replay_response['result']['files'] == [
        {
            'type': 'image',
            'id': file_id,
            'url': f'/api/v1/files/{file_id}/content',
            'content_type': 'image/png',
        }
    ]
    assert len(harness.created_tasks) == 1
    assert observed.image_upstream_calls == 1
    assert observed.saved_files == 1
    assert observed.assistant_files == 1


@pytest.mark.asyncio
async def test_each_model_lane_is_durably_recorded_before_operation_completion(isolated_runtime_db):
    claim = await Operations.acquire(
        user_id='lane-user',
        client_key='lane-operation',
        fingerprint='lane-fingerprint',
        chat_id='lane-chat',
        user_message_id='lane-user-message',
        assistant_message_ids=['lane-a', 'lane-b'],
    )
    assert claim.owns_launch
    running = await Operations.mark_running(claim.operation.id, claim.owner_token, ['lane-task-a', 'lane-task-b'])
    assert running.owns_launch

    first = await Operations.finish_lane(claim.operation.id, claim.owner_token, 'lane-a', succeeded=True)
    assert first is not None
    assert first.status != OPERATION_COMPLETED
    second = await Operations.finish_lane(claim.operation.id, claim.owner_token, 'lane-b', succeeded=True)

    assert second is not None
    assert second.status == OPERATION_COMPLETED
    assert second.completed_lanes == ['lane-a', 'lane-b']


@pytest.mark.asyncio
async def test_operation_keeps_the_credential_reference_that_started_it(isolated_runtime_db):
    claim = await Operations.acquire(
        user_id='credential-user',
        client_key='credential-operation',
        fingerprint='credential-fingerprint',
        chat_id='credential-chat',
        user_message_id='credential-user-message',
        assistant_message_ids=['credential-assistant'],
        credential_session_id='browser-key-session-a',
    )
    assert claim.owns_launch

    replay = await Operations.acquire(
        user_id='credential-user',
        client_key='credential-operation',
        fingerprint='credential-fingerprint',
        chat_id='credential-chat',
        user_message_id='credential-user-message',
        assistant_message_ids=['credential-assistant'],
        credential_session_id='browser-key-session-b',
    )

    assert not replay.owns_launch
    assert replay.operation.credential_session_id == 'browser-key-session-a'


@pytest.mark.asyncio
async def test_interrupted_upstream_stream_without_a_tool_row_is_durable_unknown_not_retryable(isolated_runtime_db):
    """A native tool call may be in an unfinished stream, before it can be claimed."""
    claim = await Operations.acquire(
        user_id='stream-unknown-user',
        client_key='stream-unknown-operation',
        fingerprint='stream-unknown-fingerprint',
        chat_id='stream-unknown-chat',
        user_message_id='stream-unknown-user-message',
        assistant_message_ids=['stream-unknown-assistant'],
    )
    assert claim.owns_launch
    assert (
        await Operations.mark_running(claim.operation.id, claim.owner_token, ['stream-unknown-task'])
    ).owns_launch

    operation = await Operations.finish_lane(
        claim.operation.id,
        claim.owner_token,
        'stream-unknown-assistant',
        succeeded=False,
        result_unknown=True,
    )

    assert operation is not None
    assert operation.status == OPERATION_UNKNOWN
    assert operation.result['code'] == 'operation_result_unknown'
    assert operation.completed_lanes == ['stream-unknown-assistant']
    async with AsyncSession(isolated_runtime_db) as session:
        executions = (
            await session.scalars(select(ToolExecution).where(ToolExecution.operation_id == claim.operation.id))
        ).all()
    assert executions == []


@pytest.mark.asyncio
async def test_expired_ready_owner_cannot_start_a_second_chat_or_image_task(
    isolated_runtime_db, tmp_path, monkeypatch
):
    """The loser must return durable state before creating any background task."""

    harness = _EntryHarness(isolated_runtime_db, tmp_path, monkeypatch)
    chat_id = 'stale-ready-owner-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)
    body = _payload(
        harness.model,
        chat_id=chat_id,
        user_message_id='stale-ready-owner-user-message',
        assistant_message_id='stale-ready-owner-assistant-message',
    )
    headers = {'Idempotency-Key': 'stale-ready-owner-operation'}
    old_worker_paused = asyncio.Event()
    release_old_worker = asyncio.Event()
    release_model = asyncio.Event()
    real_mark_running = Operations.mark_running
    real_model_stream = main.chat_completion_handler
    mark_calls = 0

    async def pause_first_ready_to_running(*args, **kwargs):
        nonlocal mark_calls
        mark_calls += 1
        if mark_calls == 1:
            old_worker_paused.set()
            await release_old_worker.wait()
        return await real_mark_running(*args, **kwargs)

    async def pause_model(*args, **kwargs):
        await release_model.wait()
        return await real_model_stream(*args, **kwargs)

    monkeypatch.setattr(Operations, 'mark_running', pause_first_ready_to_running)
    monkeypatch.setattr(main, 'chat_completion_handler', pause_model)
    first_request = asyncio.create_task(harness.submit(body, headers=headers))
    try:
        await asyncio.wait_for(old_worker_paused.wait(), 5)
        async with AsyncSession(isolated_runtime_db) as session:
            operation = await session.scalar(
                select(ChatOperation).where(ChatOperation.client_key == 'stale-ready-owner-operation')
            )
            assert operation is not None
            operation.lease_expires_at = 0
            await session.commit()

        second_response = await harness.submit(body, headers=headers)
        release_old_worker.set()
        first_response = await asyncio.wait_for(first_request, 5)
        release_model.set()
        await harness.wait_for_tasks()
    finally:
        release_old_worker.set()
        release_model.set()
        await asyncio.gather(first_request, return_exceptions=True)
        await harness.wait_for_tasks()

    observed = await harness.observe(chat_id)
    assert observed.background_tasks == 1, observed
    # One normal tool loop performs an initial and a continuation model call;
    # the stale worker must not add another pair before image protection runs.
    assert observed.chat_upstream_calls == 2, observed
    assert observed.image_upstream_calls == 1, observed
    assert first_response['operation_id'] == second_response['operation_id']
    assert first_response['operation_status'] in {OPERATION_RUNNING, OPERATION_COMPLETED}
    assert second_response['operation_status'] in {OPERATION_RUNNING, OPERATION_COMPLETED}


@pytest.mark.asyncio
async def test_concurrent_new_tool_call_ids_claim_only_one_paid_execution(isolated_runtime_db):
    claim = await Operations.acquire(
        user_id='contested-lane-user',
        client_key='contested-lane-operation',
        fingerprint='contested-lane-fingerprint',
        chat_id='contested-lane-chat',
        user_message_id='contested-lane-user-message',
        assistant_message_ids=['contested-lane-assistant'],
    )
    assert claim.owns_launch
    assert (await Operations.mark_running(claim.operation.id, claim.owner_token, ['contested-lane-task'])).owns_launch

    async def claim_call(raw_tool_call_id: str):
        return await Operations.claim_image_execution(
            operation_id=claim.operation.id,
            owner_token=claim.owner_token,
            lane_id='contested-lane-assistant',
            raw_tool_call_id=raw_tool_call_id,
            tool_name='generate_image',
            parameters={'prompt': 'one paid image only'},
            source_file_refs=[],
        )

    first, second = await asyncio.gather(claim_call('concurrent-call-a'), claim_call('concurrent-call-b'))
    assert sorted((first.action, second.action)) == ['block', 'execute']

    async with AsyncSession(isolated_runtime_db) as session:
        executions = (
            await session.scalars(select(ToolExecution).where(ToolExecution.operation_id == claim.operation.id))
        ).all()
        assert len(executions) == 1
        assert executions[0].raw_tool_call_id in {'concurrent-call-a', 'concurrent-call-b'}


@pytest.mark.asyncio
async def test_new_chat_initialization_retry_reuses_reserved_chat_id_before_task_launch(
    isolated_runtime_db, tmp_path, monkeypatch
):
    harness = _EntryHarness(isolated_runtime_db, tmp_path, monkeypatch)
    body = _payload(
        harness.model,
        chat_id='',
        user_message_id='initialization-retry-user',
        assistant_message_id='initialization-retry-assistant',
    )
    headers = {'Idempotency-Key': 'initialization-retry-operation'}
    real_mark_running = Operations.mark_running

    async def fail_after_chat_initialization(*_args, **_kwargs):
        raise RuntimeError('simulated crash after chat insert and before task launch')

    monkeypatch.setattr(Operations, 'mark_running', fail_after_chat_initialization)
    with pytest.raises(RuntimeError, match='after chat insert'):
        await harness.submit(body, headers=headers)
    async with AsyncSession(isolated_runtime_db) as session:
        operation = await session.scalar(
            select(ChatOperation).where(ChatOperation.client_key == 'initialization-retry-operation')
        )
        assert operation is not None
        reserved_chat_id = operation.chat_id
        operation.lease_expires_at = 0
        await session.commit()
    monkeypatch.setattr(Operations, 'mark_running', real_mark_running)

    replay_response = await harness.submit(body, headers=headers)
    await harness.wait_for_tasks()
    observed = await harness.observe(replay_response['chat_id'])

    assert replay_response['chat_id'] == reserved_chat_id
    assert observed.chats == 1, observed
    assert observed.messages == 2, observed
    assert observed.background_tasks == 1, observed
    assert observed.image_upstream_calls == 1, observed


@pytest.mark.asyncio
async def test_expired_owner_marks_sent_execution_unknown_and_cannot_overwrite(
    isolated_runtime_db, tmp_path, monkeypatch
):
    claim = await Operations.acquire(
        user_id='lease-user',
        client_key='lease-operation',
        fingerprint='stable-request-fingerprint',
        chat_id='lease-chat',
        user_message_id='lease-user-message',
        assistant_message_ids=['lease-assistant-message'],
    )
    assert claim.owns_launch
    assert (await Operations.mark_running(claim.operation.id, claim.owner_token, ['lease-task'])).owns_launch
    execution_claim = await Operations.claim_image_execution(
        operation_id=claim.operation.id,
        owner_token=claim.owner_token,
        lane_id='lease-assistant-message',
        raw_tool_call_id='lease-model-call',
        tool_name='edit_image',
        parameters={'prompt': 'edit', 'image_urls': ['/api/v1/files/source/content']},
        source_file_refs=[{'url': '/api/v1/files/source/content'}],
    )
    assert execution_claim.action == 'execute'

    async with AsyncSession(isolated_runtime_db) as session:
        operation = await session.get(ChatOperation, claim.operation.id)
        assert operation is not None
        operation.lease_expires_at = 0
        await session.commit()

    late_claim = await Operations.claim_image_execution(
        operation_id=claim.operation.id,
        owner_token=claim.owner_token,
        lane_id='lease-assistant-message',
        raw_tool_call_id='late-worker-call',
        tool_name='generate_image',
        parameters={'prompt': 'must not be sent'},
        source_file_refs=[],
    )
    assert late_claim.action == 'block'

    replay = await Operations.acquire(
        user_id='lease-user',
        client_key='lease-operation',
        fingerprint='stable-request-fingerprint',
        chat_id='ignored-on-replay',
        user_message_id='lease-user-message',
        assistant_message_ids=['lease-assistant-message'],
    )
    assert not replay.owns_launch
    assert replay.operation.status == OPERATION_UNKNOWN
    assert await Operations.mark_execution_completed(
        execution_claim.execution.id,
        claim.owner_token,
        {'status': 'success'},
        [],
    ) is None

    async with AsyncSession(isolated_runtime_db) as session:
        execution = await session.get(ToolExecution, execution_claim.execution.id)
        assert execution is not None
        assert execution.status == EXECUTION_UNKNOWN
        assert execution.lease_owner is None


@pytest.mark.asyncio
async def test_unknown_image_execution_replay_returns_confirmation_state_without_new_task(
    isolated_runtime_db, tmp_path, monkeypatch
):
    harness = _EntryHarness(
        isolated_runtime_db,
        tmp_path,
        monkeypatch,
        image_mode='timeout_unknown',
        router_backed_image_provider=True,
        continuation_steps=[
            {'kind': 'tools', 'tool_calls': [_edit_call('retry-with-new-call-id')]},
            {'kind': 'text', 'text': 'must not cause another image request'},
        ],
    )
    chat_id = 'unknown-replay-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)
    body = _payload(
        harness.model,
        chat_id=chat_id,
        user_message_id='unknown-replay-user',
        assistant_message_id='unknown-replay-assistant',
    )
    headers = {'Idempotency-Key': 'unknown-replay-operation'}

    first_response = await harness.submit(body, headers=headers)
    await harness.wait_for_tasks()
    replay_response = await harness.submit(body, headers=headers)

    async with AsyncSession(isolated_runtime_db) as session:
        operation = await session.get(ChatOperation, first_response['operation_id'])
        execution = await session.scalar(
            select(ToolExecution).where(ToolExecution.operation_id == first_response['operation_id'])
        )
        assert operation is not None
        assert operation.status == OPERATION_UNKNOWN
        assert execution is not None
        assert execution.status == EXECUTION_UNKNOWN

    observed = await harness.observe(chat_id)
    assert replay_response['operation_id'] == first_response['operation_id']
    assert replay_response['operation_status'] == OPERATION_UNKNOWN
    assert len(harness.created_tasks) == 1
    assert observed.chat_upstream_calls == 3
    assert observed.image_upstream_calls == 1
    assert observed.saved_files == 0


async def _delivery_fixture(
    isolated_runtime_db, tmp_path, monkeypatch, *, lanes: list[str], execution_lanes: list[str] | None = None
):
    """Create durable execution rows without running a model or image provider."""

    harness = _EntryHarness(isolated_runtime_db, tmp_path, monkeypatch)
    chat_id = f"delivery-aggregate-{'-'.join(lanes)}"
    await _create_empty_chat(chat_id, harness.user, harness.model)
    for lane in lanes:
        await Chats.upsert_message_to_chat_by_id_and_message_id(
            chat_id,
            lane,
            {'id': lane, 'role': 'assistant', 'content': '', 'files': []},
        )

    claim = await Operations.acquire(
        user_id=harness.user.id,
        client_key=f'delivery-aggregate-operation-{"-".join(lanes)}',
        fingerprint='delivery-aggregate-fingerprint',
        chat_id=chat_id,
        user_message_id='delivery-aggregate-user-message',
        assistant_message_ids=lanes,
    )
    assert claim.owns_launch
    assert (await Operations.mark_running(claim.operation.id, claim.owner_token, lanes)).owns_launch

    execution_lanes = lanes if execution_lanes is None else execution_lanes
    executions = {}
    files = {}
    for lane in execution_lanes:
        execution = await Operations.claim_image_execution(
            operation_id=claim.operation.id,
            owner_token=claim.owner_token,
            lane_id=lane,
            raw_tool_call_id=f'delivery-aggregate-{lane}-call',
            tool_name='generate_image',
            parameters={'prompt': lane},
            source_file_refs=[],
        )
        assert execution.action == 'execute'
        executions[lane] = execution.execution

        path = tmp_path / f'delivery-aggregate-{lane}.png'
        path.write_bytes(f'delivery-{lane}'.encode())
        stored = await Files.insert_new_file(
            harness.user.id,
            FileForm(
                id=f'delivery-aggregate-{lane}-file',
                filename=path.name,
                path=str(path),
                data={},
                meta={'content_type': 'image/png'},
            ),
        )
        files[lane] = [
            {
                'id': stored.id,
                'type': 'image',
                'url': f'/api/v1/files/{stored.id}/content',
                'content_type': 'image/png',
            }
        ]

    return harness, claim, chat_id, executions, files


async def _assistant_file_ids(chat_id: str, lane_id: str) -> list[str]:
    chat = await Chats.get_chat_by_id(chat_id)
    assert chat is not None
    message = chat.chat['history']['messages'][lane_id]
    return [file['id'] for file in message.get('files') or []]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('other_state', 'expected_operation_state', 'expected_other_execution_state'),
    [
        ('sending', OPERATION_RUNNING, EXECUTION_SENDING),
        ('unknown', OPERATION_UNKNOWN, EXECUTION_UNKNOWN),
        ('completed', OPERATION_COMPLETED, 'COMPLETED'),
        ('delivery_pending', OPERATION_COMPLETED, 'COMPLETED'),
    ],
)
async def test_delivery_recovery_aggregates_every_lane_before_declaring_completion(
    isolated_runtime_db,
    tmp_path,
    monkeypatch,
    other_state,
    expected_operation_state,
    expected_other_execution_state,
):
    harness, claim, chat_id, executions, files = await _delivery_fixture(
        isolated_runtime_db, tmp_path, monkeypatch, lanes=['lane-a', 'lane-b']
    )
    assert await Operations.mark_delivery_pending(
        executions['lane-a'].id,
        claim.owner_token,
        files['lane-a'],
        'lane-a association failed',
    )

    if other_state == 'unknown':
        assert await Operations.finish_lane(claim.operation.id, claim.owner_token, 'lane-a', succeeded=True)
        assert await Operations.mark_execution_unknown(
            executions['lane-b'].id, claim.owner_token, 'lane-b result is unknown'
        )
    elif other_state == 'completed':
        assert await Chats.add_message_files_by_id_and_message_id(chat_id, 'lane-b', files['lane-b'])
        assert await Operations.mark_execution_completed(
            executions['lane-b'].id,
            claim.owner_token,
            {'status': 'success', 'images': files['lane-b']},
            files['lane-b'],
        )
        assert await Operations.finish_lane(claim.operation.id, claim.owner_token, 'lane-a', succeeded=True)
        assert await Operations.finish_lane(claim.operation.id, claim.owner_token, 'lane-b', succeeded=True)
    elif other_state == 'delivery_pending':
        assert await Operations.finish_lane(claim.operation.id, claim.owner_token, 'lane-a', succeeded=True)
        assert await Operations.mark_delivery_pending(
            executions['lane-b'].id,
            claim.owner_token,
            files['lane-b'],
            'lane-b association failed',
        )
        assert await Operations.finish_lane(claim.operation.id, claim.owner_token, 'lane-b', succeeded=True)
    else:
        assert await Operations.finish_lane(claim.operation.id, claim.owner_token, 'lane-a', succeeded=True)

    recovered = await Operations.recover_delivery(claim.operation.id)
    assert recovered is not None
    assert recovered.status == expected_operation_state
    lane_a = await Operations.get_execution(executions['lane-a'].id)
    lane_b = await Operations.get_execution(executions['lane-b'].id)
    assert lane_a is not None and lane_a.status == 'COMPLETED'
    assert lane_b is not None and lane_b.status == expected_other_execution_state
    assert await _assistant_file_ids(chat_id, 'lane-a') == [files['lane-a'][0]['id']]

    if expected_operation_state != OPERATION_COMPLETED:
        assert recovered.result['code'] == 'operation_incomplete'
        assert recovered.result['pending_executions'] == [
            {
                'lane_id': 'lane-b',
                'tool_name': 'generate_image',
                'status': expected_other_execution_state,
            }
        ]
    else:
        assert set(file['id'] for file in recovered.result['files']) == {
            files['lane-a'][0]['id'],
            files['lane-b'][0]['id'],
        }

    assert harness.created_tasks == []
    assert harness.chat_upstream_calls == []
    assert harness.image_upstream_calls == []


@pytest.mark.asyncio
async def test_recovered_delivery_preserves_unstarted_sibling_lane_execution_right(
    isolated_runtime_db, tmp_path, monkeypatch
):
    """A sibling still in model inference has no ToolExecution row yet, but is not done."""

    harness, claim, chat_id, executions, files = await _delivery_fixture(
        isolated_runtime_db,
        tmp_path,
        monkeypatch,
        lanes=['lane-a', 'lane-b'],
        execution_lanes=['lane-a'],
    )
    assert await Operations.mark_delivery_pending(
        executions['lane-a'].id,
        claim.owner_token,
        files['lane-a'],
        'lane-a association failed',
    )

    recovered = await Operations.recover_delivery(claim.operation.id)
    assert recovered is not None
    assert recovered.status == OPERATION_RUNNING
    assert recovered.completed_lanes == []
    assert recovered.lease_owner == claim.owner_token
    assert recovered.result['pending_lanes'] == ['lane-a', 'lane-b']
    assert await _assistant_file_ids(chat_id, 'lane-a') == [files['lane-a'][0]['id']]

    # B may reach its first paid tool only after A's saved result is recovered.
    lane_b = await Operations.claim_image_execution(
        operation_id=claim.operation.id,
        owner_token=claim.owner_token,
        lane_id='lane-b',
        raw_tool_call_id='lane-b-after-a-recovery',
        tool_name='generate_image',
        parameters={'prompt': 'B still owns its model lane'},
        source_file_refs=[],
    )
    assert lane_b.action == 'execute'
    assert await Operations.mark_execution_completed(
        lane_b.execution.id,
        claim.owner_token,
        {'status': 'success'},
        [],
    )
    assert await Operations.finish_lane(claim.operation.id, claim.owner_token, 'lane-a', succeeded=True)
    completed = await Operations.finish_lane(claim.operation.id, claim.owner_token, 'lane-b', succeeded=True)

    assert completed is not None and completed.status == OPERATION_COMPLETED
    assert completed.completed_lanes == ['lane-a', 'lane-b']
    assert completed.result['files'] == files['lane-a']
    assert harness.created_tasks == []
    assert harness.chat_upstream_calls == []
    assert harness.image_upstream_calls == []


@pytest.mark.asyncio
async def test_recovered_delivery_waits_for_unstarted_pure_text_sibling_lane(
    isolated_runtime_db, tmp_path, monkeypatch
):
    harness, claim, chat_id, executions, files = await _delivery_fixture(
        isolated_runtime_db,
        tmp_path,
        monkeypatch,
        lanes=['lane-a', 'lane-b'],
        execution_lanes=['lane-a'],
    )
    assert await Operations.mark_delivery_pending(
        executions['lane-a'].id,
        claim.owner_token,
        files['lane-a'],
        'lane-a association failed',
    )
    # A's model turn ended normally; B is a pure-text turn with no execution row.
    assert await Operations.finish_lane(claim.operation.id, claim.owner_token, 'lane-a', succeeded=True)

    recovered = await Operations.recover_delivery(claim.operation.id)
    assert recovered is not None and recovered.status == OPERATION_RUNNING
    assert recovered.completed_lanes == ['lane-a']
    assert recovered.result['pending_lanes'] == ['lane-b']
    assert await _assistant_file_ids(chat_id, 'lane-a') == [files['lane-a'][0]['id']]

    completed = await Operations.finish_lane(claim.operation.id, claim.owner_token, 'lane-b', succeeded=True)
    assert completed is not None and completed.status == OPERATION_COMPLETED
    assert completed.completed_lanes == ['lane-a', 'lane-b']
    assert completed.result['files'] == files['lane-a']
    assert harness.created_tasks == []
    assert harness.chat_upstream_calls == []
    assert harness.image_upstream_calls == []


@pytest.mark.asyncio
async def test_failed_model_lane_is_persisted_as_failed_not_completed(
    isolated_runtime_db, tmp_path, monkeypatch
):
    harness, claim, _chat_id, _executions, _files = await _delivery_fixture(
        isolated_runtime_db,
        tmp_path,
        monkeypatch,
        lanes=['lane-a'],
        execution_lanes=[],
    )

    failed = await Operations.finish_lane(claim.operation.id, claim.owner_token, 'lane-a', succeeded=False)
    assert failed is not None and failed.status == OPERATION_FAILED
    assert failed.completed_lanes == ['lane-a']
    assert failed.result['code'] == 'operation_failed'
    assert await Operations.recover_delivery(claim.operation.id) is not None
    assert (await Operations.get(claim.operation.id)).status == OPERATION_FAILED
    assert harness.created_tasks == []
    assert harness.chat_upstream_calls == []
    assert harness.image_upstream_calls == []


@pytest.mark.asyncio
async def test_stopped_model_lane_with_sent_image_persists_unknown_without_retry(
    isolated_runtime_db, tmp_path, monkeypatch
):
    harness, claim, _chat_id, executions, _files = await _delivery_fixture(
        isolated_runtime_db,
        tmp_path,
        monkeypatch,
        lanes=['lane-a'],
    )

    unknown = await Operations.finish_lane(claim.operation.id, claim.owner_token, 'lane-a', succeeded=False)
    execution = await Operations.get_execution(executions['lane-a'].id)
    assert unknown is not None and unknown.status == OPERATION_UNKNOWN
    assert unknown.completed_lanes == ['lane-a']
    assert unknown.result['code'] == 'operation_result_unknown'
    assert execution is not None and execution.status == EXECUTION_UNKNOWN
    assert execution.lease_owner is None
    assert harness.created_tasks == []
    assert harness.chat_upstream_calls == []
    assert harness.image_upstream_calls == []


@pytest.mark.asyncio
async def test_single_lane_delivery_recovery_completes_without_replaying_model_or_image(
    isolated_runtime_db, tmp_path, monkeypatch
):
    harness, claim, chat_id, executions, files = await _delivery_fixture(
        isolated_runtime_db, tmp_path, monkeypatch, lanes=['lane-a']
    )
    assert await Operations.mark_delivery_pending(
        executions['lane-a'].id,
        claim.owner_token,
        files['lane-a'],
        'single lane association failed',
    )
    assert await Operations.finish_lane(claim.operation.id, claim.owner_token, 'lane-a', succeeded=True)

    recovered = await Operations.recover_delivery(claim.operation.id)
    assert recovered is not None and recovered.status == OPERATION_COMPLETED
    assert await _assistant_file_ids(chat_id, 'lane-a') == [files['lane-a'][0]['id']]
    assert recovered.result['files'] == files['lane-a']
    assert harness.created_tasks == []
    assert harness.chat_upstream_calls == []
    assert harness.image_upstream_calls == []


@pytest.mark.asyncio
async def test_lane_completion_racing_delivery_recovery_keeps_pending_sibling_and_one_attachment(
    isolated_runtime_db, tmp_path, monkeypatch
):
    harness, claim, chat_id, executions, files = await _delivery_fixture(
        isolated_runtime_db,
        tmp_path,
        monkeypatch,
        lanes=['lane-a', 'lane-b'],
        execution_lanes=['lane-a'],
    )
    assert await Operations.mark_delivery_pending(
        executions['lane-a'].id,
        claim.owner_token,
        files['lane-a'],
        'lane-a concurrent association recovery',
    )

    await asyncio.gather(
        Operations.finish_lane(claim.operation.id, claim.owner_token, 'lane-a', succeeded=True),
        Operations.recover_delivery(claim.operation.id),
        Operations.recover_delivery(claim.operation.id),
    )
    pending = await Operations.get(claim.operation.id)
    assert pending is not None and pending.status == OPERATION_RUNNING
    assert pending.completed_lanes == ['lane-a']
    assert pending.result['pending_lanes'] == ['lane-b']
    assert pending.result['pending_executions'] == []
    assert await _assistant_file_ids(chat_id, 'lane-a') == [files['lane-a'][0]['id']]

    completed = await Operations.finish_lane(claim.operation.id, claim.owner_token, 'lane-b', succeeded=True)
    assert completed is not None and completed.status == OPERATION_COMPLETED
    assert harness.created_tasks == []
    assert harness.chat_upstream_calls == []
    assert harness.image_upstream_calls == []


@pytest.mark.asyncio
async def test_concurrent_duplicate_delivery_recovery_does_not_duplicate_attachment_or_terminal_state(
    isolated_runtime_db, tmp_path, monkeypatch
):
    harness, claim, chat_id, executions, files = await _delivery_fixture(
        isolated_runtime_db, tmp_path, monkeypatch, lanes=['lane-a']
    )
    assert await Operations.mark_delivery_pending(
        executions['lane-a'].id,
        claim.owner_token,
        files['lane-a'],
        'concurrent association recovery',
    )
    assert await Operations.finish_lane(claim.operation.id, claim.owner_token, 'lane-a', succeeded=True)

    first, second = await asyncio.gather(
        Operations.recover_delivery(claim.operation.id), Operations.recover_delivery(claim.operation.id)
    )
    final = await Operations.get(claim.operation.id)
    execution = await Operations.get_execution(executions['lane-a'].id)
    assert first is not None and second is not None and final is not None
    assert final.status == OPERATION_COMPLETED
    assert execution is not None and execution.status == 'COMPLETED'
    assert await _assistant_file_ids(chat_id, 'lane-a') == [files['lane-a'][0]['id']]
    assert final.result['files'] == files['lane-a']
    assert harness.created_tasks == []
    assert harness.chat_upstream_calls == []
    assert harness.image_upstream_calls == []


@pytest.mark.asyncio
async def test_browser_managed_request_without_operation_key_is_rejected_before_chat_write(
    isolated_runtime_db, tmp_path, monkeypatch
):
    harness = _EntryHarness(
        isolated_runtime_db,
        tmp_path,
        monkeypatch,
        initial_tool_calls=[_edit_call()],
    )
    chat_id = 'missing-operation-key-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)
    body = _payload(
        harness.model,
        chat_id=chat_id,
        user_message_id='missing-operation-key-user',
        assistant_message_id='missing-operation-key-assistant',
    )
    # A reconnecting browser may not have a socket id yet. That must not
    # downgrade a UI-shaped image request into the legacy unprotected path.
    body.pop('session_id')
    status_code, response = await harness.submit(
        body,
        expected_status=None,
        include_status=True,
    )

    observed = await harness.observe(chat_id)
    assert status_code == 400
    assert response['detail']['code'] == 'idempotency_key_required'
    assert observed.messages == 0
    assert observed.background_tasks == 0
    assert observed.image_upstream_calls == 0
