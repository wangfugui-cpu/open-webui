"""Normal-behavior and request-key controls for entry-level reliability checks."""

from __future__ import annotations

import copy

import pytest
from open_webui.models.chats import Chats
from open_webui.models.files import File
from open_webui.models.operations import EXECUTION_NOT_SENT, OPERATION_COMPLETED, ChatOperation, ToolExecution
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from test_yanchuan_execution_reliability import (
    _create_empty_chat,
    _edit_call,
    _EntryHarness,
    _generate_call,
    _invalid_edit_call,
    _model,
    _payload,
    _user,
)


@pytest.mark.asyncio
async def test_entry_rejects_same_operation_key_with_different_request_content(
    isolated_runtime_db, tmp_path, monkeypatch
):
    harness = _EntryHarness(isolated_runtime_db, tmp_path, monkeypatch)
    chat_id = 'operation-key-conflict-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)
    headers = {'Idempotency-Key': 'operation-content-conflict'}
    original = _payload(
        harness.model,
        chat_id=chat_id,
        user_message_id='operation-conflict-user-message',
        assistant_message_id='operation-conflict-assistant-message',
    )
    conflicting = _payload(
        harness.model,
        chat_id=chat_id,
        user_message_id='operation-conflict-user-message',
        assistant_message_id='operation-conflict-assistant-message',
        content='请改成冷色调，而不是原来的暖色调',
    )

    await harness.submit(original, headers=headers)
    await harness.wait_for_tasks()
    before_conflict = await Chats.get_chat_by_id(chat_id)
    assert before_conflict is not None
    before_history = copy.deepcopy(before_conflict.chat['history'])
    status_code, response = await harness.submit(
        conflicting,
        headers=headers,
        expected_status=None,
        include_status=True,
    )
    await harness.wait_for_tasks(start=1)

    observed = await harness.observe(chat_id)
    after_conflict = await Chats.get_chat_by_id(chat_id)
    assert status_code == 409, response
    assert response['detail']['code'] == 'idempotency_key_payload_mismatch'
    assert after_conflict is not None
    assert after_conflict.chat['history'] == before_history
    assert observed.background_tasks == 1, observed
    assert observed.image_upstream_calls == 1, observed
    assert observed.saved_files == 1, observed


@pytest.mark.asyncio
async def test_normal_follow_up_and_explicit_regenerate_remain_distinct_operations(
    isolated_runtime_db, tmp_path, monkeypatch
):
    harness = _EntryHarness(isolated_runtime_db, tmp_path, monkeypatch)
    chat_id = 'normal-follow-up-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)

    first = _payload(
        harness.model,
        chat_id=chat_id,
        user_message_id='normal-first-user-message',
        assistant_message_id='normal-first-assistant-message',
    )
    follow_up = _payload(
        harness.model,
        chat_id=chat_id,
        user_message_id='normal-follow-up-user-message',
        assistant_message_id='normal-follow-up-assistant-message',
        content='继续把阴影调淡一点',
    )
    follow_up['parent_id'] = 'normal-first-assistant-message'
    regenerate = _payload(
        harness.model,
        chat_id=chat_id,
        user_message_id='normal-first-user-message',
        assistant_message_id='normal-regenerate-assistant-message',
    )
    regenerate['parent_id'] = 'normal-first-user-message'

    for body, operation_id in (
        (first, 'operation-normal-first'),
        (follow_up, 'operation-normal-follow-up'),
        (regenerate, 'operation-normal-regenerate'),
    ):
        start = len(harness.created_tasks)
        await harness.submit(body, headers={'Idempotency-Key': operation_id})
        await harness.wait_for_tasks(start=start)

    observed = await harness.observe(chat_id)
    assert observed.chats == 1, observed
    assert observed.messages == 5, observed
    assert observed.background_tasks == 3, observed
    assert observed.chat_upstream_calls == 6, observed
    assert observed.image_upstream_calls == 3, observed
    assert observed.saved_files == 3, observed
    assert observed.assistant_files == 3, observed
    assert {response['chat_id'] for response in observed.client_responses} == {chat_id}, observed


@pytest.mark.asyncio
async def test_normal_response_with_two_distinct_image_operations_is_not_collapsed(
    isolated_runtime_db, tmp_path, monkeypatch
):
    harness = _EntryHarness(
        isolated_runtime_db,
        tmp_path,
        monkeypatch,
        initial_tool_calls=[_edit_call('legal-edit-call'), _generate_call('legal-generate-call')],
    )
    chat_id = 'normal-two-images-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)

    await harness.submit(
        _payload(
            harness.model,
            chat_id=chat_id,
            user_message_id='normal-two-images-user-message',
            assistant_message_id='normal-two-images-assistant-message',
        ),
        headers={'Idempotency-Key': 'operation-normal-two-images'},
    )
    await harness.wait_for_tasks()

    observed = await harness.observe(chat_id)
    assert observed.chats == 1, observed
    assert observed.messages == 2, observed
    assert observed.background_tasks == 1, observed
    assert observed.chat_upstream_calls == 2, observed
    assert observed.image_upstream_calls == 2, observed
    assert observed.saved_files == 2, observed
    assert observed.assistant_files == 2, observed


@pytest.mark.asyncio
async def test_normal_multi_model_response_keeps_its_two_execution_lanes(
    isolated_runtime_db, tmp_path, monkeypatch
):
    harness = _EntryHarness(isolated_runtime_db, tmp_path, monkeypatch)
    second_model = _model('reliability-model-second')
    harness.models[second_model['id']] = second_model
    chat_id = 'normal-multi-model-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)
    body = _payload(
        harness.model,
        chat_id=chat_id,
        user_message_id='normal-multi-model-user-message',
        assistant_message_id='unused-single-model-id',
    )
    body.pop('id')
    body['message_ids'] = [
        {'model_id': harness.model['id'], 'message_id': 'normal-multi-model-first-assistant'},
        {'model_id': second_model['id'], 'message_id': 'normal-multi-model-second-assistant'},
    ]

    first_response = await harness.submit(body, headers={'Idempotency-Key': 'operation-normal-multi-model'})
    await harness.wait_for_tasks()

    observed = await harness.observe(chat_id)
    assert observed.chats == 1, observed
    assert observed.messages == 3, observed
    assert observed.background_tasks == 2, observed
    assert observed.chat_upstream_calls == 4, observed
    assert observed.image_upstream_calls == 2, observed
    assert observed.saved_files == 2, observed

    async with AsyncSession(isolated_runtime_db) as session:
        operation = await session.get(ChatOperation, first_response['operation_id'])
        assert operation is not None
        assert operation.status == OPERATION_COMPLETED
        assert set(operation.completed_lanes) == {
            'normal-multi-model-first-assistant',
            'normal-multi-model-second-assistant',
        }
        assert operation.result['assistant_message_ids'] == [
            'normal-multi-model-first-assistant',
            'normal-multi-model-second-assistant',
        ]
        assert len(operation.result['files']) == 2


@pytest.mark.asyncio
async def test_normal_same_raw_tool_call_id_in_distinct_operations_is_not_shared(
    isolated_runtime_db, tmp_path, monkeypatch
):
    """Raw model call IDs are diagnostic only and must not merge distinct user operations."""

    harness = _EntryHarness(
        isolated_runtime_db,
        tmp_path,
        monkeypatch,
        initial_tool_calls=[_edit_call('reused-raw-model-call-id')],
    )
    first_chat_id = 'normal-raw-call-first-chat'
    second_chat_id = 'normal-raw-call-second-chat'

    for chat_id, operation_id, user_message_id, assistant_message_id in (
        (
            first_chat_id,
            'operation-raw-call-first',
            'normal-raw-call-first-user',
            'normal-raw-call-first-assistant',
        ),
        (
            second_chat_id,
            'operation-raw-call-second',
            'normal-raw-call-second-user',
            'normal-raw-call-second-assistant',
        ),
    ):
        await _create_empty_chat(chat_id, harness.user, harness.model)
        start = len(harness.created_tasks)
        await harness.submit(
            _payload(
                harness.model,
                chat_id=chat_id,
                user_message_id=user_message_id,
                assistant_message_id=assistant_message_id,
            ),
            headers={'Idempotency-Key': operation_id},
        )
        await harness.wait_for_tasks(start=start)

    first_observed = await harness.observe(first_chat_id)
    second_observed = await harness.observe(second_chat_id)
    assert first_observed.chats == 2, first_observed
    assert first_observed.background_tasks == 2, first_observed
    assert first_observed.image_upstream_calls == 2, first_observed
    assert first_observed.saved_files == 2, first_observed
    assert first_observed.assistant_files == 1, first_observed
    assert second_observed.assistant_files == 1, second_observed


@pytest.mark.asyncio
async def test_normal_identical_content_from_different_users_is_never_shared(
    isolated_runtime_db, tmp_path, monkeypatch
):
    harness = _EntryHarness(isolated_runtime_db, tmp_path, monkeypatch)
    first_chat_id = 'normal-user-one-chat'
    await _create_empty_chat(first_chat_id, harness.user, harness.model)
    shared_ids = {
        'user_message_id': 'same-content-user-message',
        'assistant_message_id': 'same-content-assistant-message',
    }

    await harness.submit(
        _payload(harness.model, chat_id=first_chat_id, **shared_ids),
        headers={'Idempotency-Key': 'same-operation-key-for-two-users'},
    )
    await harness.wait_for_tasks()

    harness.user = _user('reliability-other-user')
    second_chat_id = 'normal-user-two-chat'
    await _create_empty_chat(second_chat_id, harness.user, harness.model)
    await harness.submit(
        _payload(harness.model, chat_id=second_chat_id, **shared_ids),
        headers={'Idempotency-Key': 'same-operation-key-for-two-users'},
    )
    await harness.wait_for_tasks(start=1)

    observed = await harness.observe(first_chat_id)
    async with AsyncSession(isolated_runtime_db) as session:
        file_owners = set((await session.scalars(select(File.user_id))).all())
    assert observed.chats == 2, observed
    assert observed.messages == 4, observed
    assert observed.background_tasks == 2, observed
    assert observed.chat_upstream_calls == 4, observed
    assert observed.image_upstream_calls == 2, observed
    assert observed.saved_files == 2, observed
    assert observed.assistant_files == 1, observed
    assert file_owners == {'reliability-user', 'reliability-other-user'}, observed


@pytest.mark.asyncio
async def test_local_validation_failure_then_valid_reference_continues_with_edit_only(
    isolated_runtime_db, tmp_path, monkeypatch
):
    """A locally rejected edit can continue only after the model supplies a valid image reference."""

    harness = _EntryHarness(
        isolated_runtime_db,
        tmp_path,
        monkeypatch,
        initial_tool_calls=[_invalid_edit_call()],
        continuation_steps=[
            {'kind': 'tools', 'tool_calls': [_edit_call('safe-after-not-sent-call')]},
            {'kind': 'text', 'text': 'local validation final client message'},
        ],
    )
    chat_id = 'local-validation-not-sent-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)

    await harness.submit(
        _payload(
            harness.model,
            chat_id=chat_id,
            user_message_id='local-validation-user-message',
            assistant_message_id='local-validation-assistant-message',
        ),
        headers={'Idempotency-Key': 'operation-local-validation-not-sent'},
    )
    await harness.wait_for_tasks()

    observed = await harness.observe(chat_id)
    assert observed.chats == 1, observed
    assert observed.messages == 2, observed
    assert observed.background_tasks == 1, observed
    assert observed.chat_upstream_calls == 3, observed
    assert observed.image_upstream_calls == 1, observed
    assert observed.image_upstream_calls == len(
        [call for call in harness.image_upstream_calls if call['kind'] == 'edit']
    ), observed
    assert observed.saved_files == 1, observed
    assert observed.assistant_files == 1, observed


@pytest.mark.asyncio
async def test_missing_edit_source_cannot_be_replaced_by_generation_but_a_new_operation_can_generate(
    isolated_runtime_db, tmp_path, monkeypatch
):
    """Keep a failed photo edit on its edit intent; a later user operation is independent."""

    harness = _EntryHarness(
        isolated_runtime_db,
        tmp_path,
        monkeypatch,
        initial_tool_calls=[_invalid_edit_call()],
        continuation_steps=[
            {'kind': 'tools', 'tool_calls': [_generate_call('replacement-after-missing-source')]},
            {'kind': 'text', 'text': 'Please provide the original photo to continue editing.'},
        ],
    )
    chat_id = 'missing-source-does-not-become-generation-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)
    first = await harness.submit(
        _payload(
            harness.model,
            chat_id=chat_id,
            user_message_id='missing-source-user-message',
            assistant_message_id='missing-source-assistant-message',
        ),
        headers={'Idempotency-Key': 'missing-source-edit-operation'},
    )
    await harness.wait_for_tasks()

    async with AsyncSession(isolated_runtime_db) as session:
        executions = (
            await session.scalars(select(ToolExecution).where(ToolExecution.operation_id == first['operation_id']))
        ).all()
        assert [(execution.tool_name, execution.status) for execution in executions] == [
            ('edit_image', EXECUTION_NOT_SENT)
        ]

    # A true new user request has a new stable key/message identity and may
    # legitimately request a brand-new image.
    harness.initial_tool_calls = [_generate_call('new-user-operation-generation')]
    harness.continuation_steps = [{'kind': 'text', 'text': 'A new image was generated.'}]
    task_start = len(harness.created_tasks)
    await harness.submit(
        _payload(
            harness.model,
            chat_id=chat_id,
            user_message_id='new-image-user-message',
            assistant_message_id='new-image-assistant-message',
            content='请生成一张全新的山景插画',
        ),
        headers={'Idempotency-Key': 'new-image-user-operation'},
    )
    await harness.wait_for_tasks(start=task_start)

    observed = await harness.observe(chat_id)
    generated = [call for call in harness.image_upstream_calls if call['kind'] == 'generate']
    assert len(generated) == 1, observed
    assert observed.image_upstream_calls == 1, observed
    assert observed.saved_files == 1, observed
    assert observed.assistant_files == 1, observed
