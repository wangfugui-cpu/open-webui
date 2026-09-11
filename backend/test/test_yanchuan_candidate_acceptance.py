"""Offline candidate acceptance checks for the YanChuan Open WebUI fork.

These tests deliberately use a simulated image provider.  They verify the
server-side request and persistence mechanics, not whether a real model will
choose the correct tool or produce a high-quality image.
"""

import asyncio
import copy
import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from open_webui import tasks as task_registry
from open_webui.internal import db as db_internal
from open_webui.internal.db import Base
from open_webui.models.files import File
from open_webui.models.users import UserModel
from open_webui.routers import files as files_router
from open_webui.routers import images as images_router
from open_webui.routers import openai as openai_router
from open_webui.routers import auths as auths_router
from open_webui.routers.chats import overlay_response_streams
from open_webui.tools import builtin
from open_webui.utils import middleware as middleware_utils
from open_webui.utils import models as model_utils
from open_webui.utils import tools as tools_utils


async def _async_value(value):
    return value


async def _no_op_event(*_args, **_kwargs):
    return None


def _user(user_id: str, *, oauth: dict | None = None) -> UserModel:
    return UserModel(
        id=user_id,
        email=f'{user_id}@example.test',
        name=user_id,
        role='user',
        last_active_at=0,
        updated_at=0,
        created_at=0,
        oauth=oauth or {},
    )


@pytest.mark.asyncio
async def test_sub2api_key_login_never_uses_an_administrator_default_role(monkeypatch):
    """A gateway key authenticates an end user, not a local administrator."""
    captured = {}

    async def insert_new_auth(**kwargs):
        captured.update(kwargs)
        return _user('sub2api-user', oauth=kwargs['oauth'])

    async def no_op(*_args, **_kwargs):
        return None

    async def configured_default(_key, default=None):
        return 'admin' if _key == 'ui.default_user_role' else default

    monkeypatch.setattr(auths_router.Auths, 'insert_new_auth', insert_new_auth)
    monkeypatch.setattr(auths_router.Config, 'get', configured_default)
    monkeypatch.setattr(auths_router, 'apply_default_group_assignment', no_op)
    monkeypatch.setattr(auths_router, 'publish_event', no_op)

    await auths_router.create_sub2api_key_user(
        SimpleNamespace(),
        email='sub2api-user-99@users.invalid',
        name='Test User',
        subject='99',
        instance_id='test-instance',
        observed_key={'id': '7', 'name': 'Test key'},
        db=None,
    )

    assert captured['role'] == 'user'


@pytest.mark.asyncio
async def test_sub2api_key_user_can_only_see_models_from_its_own_gateway_connection(monkeypatch):
    """A fresh Open WebUI DB must not hide a key owner's discovered models."""
    user = _user('sub2api-user', oauth={'sub2api': {'subject': '2'}})
    own_gateway_model = {'id': 'gpt-5.6-terra', 'urlIdx': 0}
    other_unregistered_model = {'id': 'another-provider-model', 'urlIdx': 1}

    async def runtime_config():
        return True, ['http://sub2api:8080/v1', 'https://other.example/v1'], ['', ''], {}

    monkeypatch.setattr(model_utils.openai, 'get_openai_runtime_config', runtime_config)
    monkeypatch.setattr(
        model_utils.openai,
        'is_sub2api_key_login_connection',
        lambda url: url == 'http://sub2api:8080/v1',
    )

    visible = await model_utils.get_filtered_models([own_gateway_model, other_unregistered_model], user)

    assert visible == [own_gateway_model]
    await model_utils.check_model_access(user, own_gateway_model)
    with pytest.raises(Exception, match='Model not found'):
        await model_utils.check_model_access(user, other_unregistered_model)


def _terra_model() -> dict:
    return {
        'id': 'gpt-5.6-terra',
        'info': {
            'meta': {
                'capabilities': {
                    'builtin_tools': True,
                    'image_generation': True,
                    'file_context': False,
                }
            }
        },
    }


def _request(model: dict) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(internal=False, direct=False),
        app=SimpleNamespace(state=SimpleNamespace(MODELS={model['id']: model})),
    )


def _configure_native_payload(monkeypatch) -> None:
    """Keep process_chat_payload real while disabling unrelated integrations."""

    async def get_config(key, default=None):
        return {
            'images.edit.enable': True,
            'user.permissions': {},
            'task.model.default': '',
            'task.model.external': '',
        }.get(key, default)

    async def get_many(*_keys):
        return {
            'web.search.enable': False,
            'image_generation.enable': True,
            'images.edit.enable': True,
            'code_interpreter.enable': False,
            'notes.enable': False,
            'channels.enable': False,
            'automations.enable': False,
            'calendar.enable': False,
            'ui.enable_user_webhooks': False,
            'subagents.enable': False,
            'subagents.background_enabled': False,
        }

    async def permitted(*_args, **_kwargs):
        return True

    async def unchanged_url_images(form_data, **_kwargs):
        return form_data

    async def unchanged_pipeline(_request, form_data, *_args, **_kwargs):
        return form_data

    async def no_rag(form_data, _extra_params, _user):
        return form_data, {'sources': []}

    async def no_compaction(_request, _user, messages, _metadata, _model_id, _models, _system_prompt):
        return messages, None, None

    async def no_system_oauth(*_args, **_kwargs):
        return None

    async def event_factory(_metadata):
        return _no_op_event

    monkeypatch.setattr(middleware_utils.Config, 'get', get_config)
    monkeypatch.setattr(middleware_utils.Config, 'get_many', get_many)
    monkeypatch.setattr(middleware_utils, 'has_permission', permitted)
    monkeypatch.setattr(tools_utils, 'has_permission', permitted)
    monkeypatch.setattr(middleware_utils, 'get_event_emitter', event_factory)
    monkeypatch.setattr(middleware_utils, 'get_event_call', event_factory)
    monkeypatch.setattr(middleware_utils, 'get_system_oauth_token', no_system_oauth)
    monkeypatch.setattr(middleware_utils, 'get_task_model_id', lambda model_id, *_args: model_id)
    monkeypatch.setattr(middleware_utils, 'get_reasoning_format', lambda _model: None)
    monkeypatch.setattr(middleware_utils, 'convert_url_images_to_base64', unchanged_url_images)
    monkeypatch.setattr(middleware_utils, 'compact_messages_for_request', no_compaction)
    monkeypatch.setattr(middleware_utils, 'process_pipeline_inlet_filter', unchanged_pipeline)
    monkeypatch.setattr(middleware_utils, 'chat_completion_files_handler', no_rag)
    monkeypatch.setattr(middleware_utils, 'resolve_system_prompt', no_system_oauth)
    monkeypatch.setattr(middleware_utils, 'add_file_context', lambda messages, *_args: _async_value(messages))
    monkeypatch.setattr(middleware_utils, 'ENABLE_PLUGINS', False)


class _FakeUpstreamResponse:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    def raise_for_status(self):
        return None

    async def json(self, **_kwargs):
        return {'data': [{'b64_json': 'simulated-upstream-image'}]}


class _FakeUpstreamSession:
    def __init__(self, calls):
        self.calls = calls

    def post(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeUpstreamResponse()


class _TimeoutUpstreamResponse:
    async def __aenter__(self):
        raise asyncio.TimeoutError('simulated network timeout after request dispatch')

    async def __aexit__(self, *_args):
        return False


class _TimeoutUpstreamSession:
    def __init__(self, calls):
        self.calls = calls

    def post(self, **kwargs):
        self.calls.append(kwargs)
        return _TimeoutUpstreamResponse()


def _native_tool_output(call_id: str, image_url: str) -> list[dict]:
    """The persisted Responses-style output replayed by the next chat turn."""
    return [
        {
            'type': 'function_call',
            'call_id': call_id,
            'name': 'edit_image',
            'arguments': json.dumps({'prompt': '美化这张照片'}),
            'status': 'completed',
        },
        {
            'type': 'function_call_output',
            'call_id': call_id,
            'output': [
                {'type': 'input_text', 'text': 'The edit is available in the chat.'},
                {'type': 'input_image', 'image_url': image_url},
            ],
        },
    ]


@pytest.mark.asyncio
async def test_candidate_native_image_chain_reaches_simulated_provider_persists_and_replays(monkeypatch, tmp_path):
    """UI-shaped payload -> replay -> native tool -> provider -> saved output -> next replay."""
    _configure_native_payload(monkeypatch)
    model = _terra_model()
    request = _request(model)
    user = _user('family-a', oauth={'sub2api': {'subject': 'family-a'}})
    image_path = tmp_path / 'original.jpg'
    image_path.write_bytes(b'original-image-bytes')

    source_reads = []
    uploaded = []
    upstream_calls = []
    routed_key_users = []
    stored_files = []
    emitted = []

    original_message = {
        'id': 'user-original',
        'role': 'user',
        'content': '美化这张照片，保留人物和背景',
        'files': [
            {
                'id': 'original-file',
                'type': 'file',
                'content_type': 'image/jpeg',
                'url': '/api/v1/files/original-file/content',
            }
        ],
    }
    assistant_message = {
        'id': 'assistant-edit',
        'role': 'assistant',
        'content': '',
        'files': [],
        'output': [],
        'model': model['id'],
    }
    follow_up = {
        'id': 'user-follow-up',
        'role': 'user',
        'content': '再暖一点，其他保持不变',
        'files': [],
    }
    stored_messages = [original_message, assistant_message, follow_up]

    async def load_from_db(_chat_id, message_id):
        if message_id == original_message['id']:
            return [copy.deepcopy(original_message)]
        return copy.deepcopy(stored_messages)

    async def fake_get_file_content(file_id, current_user):
        source_reads.append((file_id, current_user.id))
        return FileResponse(image_path)

    async def fake_get_session():
        return _FakeUpstreamSession(upstream_calls)

    async def fake_get_effective_key(_url, _configured_key, current_user, **_kwargs):
        routed_key_users.append(current_user.id)
        return f'test-key-for-{current_user.id}'

    async def fake_get_image_data(value, *_args, **_kwargs):
        assert value == 'simulated-upstream-image'
        return b'edited-image-bytes', 'image/png'

    async def fake_upload_image(_request, image_data, content_type, metadata, current_user, db=None):
        uploaded.append((image_data, content_type, metadata, current_user.id))
        return SimpleNamespace(id='edited-v1'), {
            'id': 'edited-v1',
            'url': '/api/v1/files/edited-v1/content',
            'content_type': 'image/png',
        }

    async def persist_files(chat_id, message_id, files):
        stored_files.append((chat_id, message_id, copy.deepcopy(files)))
        assistant_message['files'] = copy.deepcopy(files)
        return files

    async def emit(event):
        emitted.append(event)

    async def no_chat(*_args, **_kwargs):
        return None

    monkeypatch.setattr(middleware_utils, 'is_saved_chat_id', lambda _chat_id: True)
    monkeypatch.setattr(middleware_utils, 'load_messages_from_db', load_from_db)
    monkeypatch.setattr(middleware_utils.Chats, 'get_chat_folder_id', lambda *_args: _async_value(None))
    monkeypatch.setattr(middleware_utils.Chats, 'get_chat_by_id', no_chat)
    monkeypatch.setattr(middleware_utils, 'get_event_emitter', lambda *_args: _async_value(emit))

    # The actual image router is used; only its network/storage boundaries are simulated.
    monkeypatch.setattr(builtin, 'image_edits', images_router.image_edits)
    monkeypatch.setattr(builtin, 'is_saved_chat_id', lambda _chat_id: True)
    monkeypatch.setattr(builtin.Chats, 'add_message_files_by_id_and_message_id', persist_files)
    monkeypatch.setattr(images_router, 'get_file_content_by_id', fake_get_file_content)
    monkeypatch.setattr(images_router, 'get_session', fake_get_session)
    monkeypatch.setattr(images_router, 'get_effective_openai_api_key', fake_get_effective_key)
    monkeypatch.setattr(images_router, 'get_image_data', fake_get_image_data)
    monkeypatch.setattr(images_router, 'upload_image', fake_upload_image)
    monkeypatch.setattr(
        images_router,
        'get_image_config',
        lambda: _async_value(
            SimpleNamespace(
                IMAGE_EDIT_SIZE='',
                IMAGE_EDIT_MODEL='gpt-image-2',
                IMAGE_EDIT_ENGINE='openai',
                IMAGES_EDIT_OPENAI_API_BASE_URL='http://sub2api:8080/v1',
                IMAGES_EDIT_OPENAI_API_KEY='configured-test-key',
                IMAGES_EDIT_OPENAI_API_VERSION='',
            )
        ),
    )

    form_data, metadata, _events = await middleware_utils.process_chat_payload(
        request,
        {
            'model': model['id'],
            'messages': [],
            # Mirrors the front end: RAG files remain top-level, current image is on user_message.files.
            'files': [{'id': 'guide', 'type': 'file', 'content_type': 'application/pdf'}],
            'features': {'image_generation': True},
        },
        user,
        {
            'chat_id': 'saved-chat',
            'user_message_id': original_message['id'],
            'user_message': original_message,
            'message_id': assistant_message['id'],
            'session_id': 'candidate-browser-session',
            'params': {'function_calling': 'native'},
        },
        model,
    )

    assert metadata['current_user_image_refs'] == [
        {
            'id': 'original-file',
            'url': '/api/v1/files/original-file/content',
            'content_type': 'image/jpeg',
        }
    ]
    assert {'generate_image', 'edit_image'} <= metadata['tools'].keys()

    tool_result = await middleware_utils.execute_tool_call_for_output(
        request,
        form_data,
        user,
        metadata,
        None,
        emit,
        {
            'id': 'tool-edit-1',
            'function': {
                'name': 'edit_image',
                'arguments': json.dumps(
                    {
                        'prompt': '美化这张照片，保留人物和背景',
                        'image_urls': ['/api/v1/files/original-file/content'],
                    }
                ),
            },
        },
    )

    assert source_reads == [('original-file', 'family-a')]
    assert routed_key_users == ['family-a']
    assert upstream_calls[0]['url'] == 'http://sub2api:8080/v1/images/edits'
    assert upstream_calls[0]['headers']['Authorization'] == 'Bearer test-key-for-family-a'
    assert uploaded[0][1] == 'image/png'
    assert uploaded[0][2] == {
        'model': 'gpt-image-2',
        'prompt': '美化这张照片，保留人物和背景',
    }
    assert uploaded[0][3] == 'family-a'
    assert stored_files == [
        (
            'saved-chat',
            'assistant-edit',
            [
                {
                    'type': 'image',
                    'id': 'edited-v1',
                    'url': '/api/v1/files/edited-v1/content',
                    'content_type': 'image/png',
                }
            ],
        )
    ]
    assert any(event['type'] == 'chat:message:files' for event in emitted)
    assert json.loads(tool_result['content'])['images'][0]['id'] == 'edited-v1'

    assistant_message['output'] = _native_tool_output('tool-edit-1', '/api/v1/files/edited-v1/content')
    next_form_data, next_metadata, _events = await middleware_utils.process_chat_payload(
        request,
        {
            'model': model['id'],
            'messages': [],
            'files': [],
            'features': {'image_generation': True},
        },
        user,
        {
            'chat_id': 'saved-chat',
            'user_message_id': follow_up['id'],
            'user_message': follow_up,
            'message_id': 'assistant-follow-up',
            'session_id': 'candidate-browser-session',
            'params': {'function_calling': 'native'},
        },
        model,
    )

    replay_urls = [
        part.get('image_url', {}).get('url')
        for message in next_form_data['messages']
        if isinstance(message.get('content'), list)
        for part in message['content']
        if isinstance(part, dict) and part.get('type') == 'image_url'
    ]
    assert '/api/v1/files/edited-v1/content' in replay_urls
    assert next_metadata['current_user_image_refs'] == []
    assert {'generate_image', 'edit_image'} <= next_metadata['tools'].keys()


@pytest.mark.asyncio
async def test_candidate_edit_timeout_is_visible_and_not_retried_or_replaced(monkeypatch):
    calls = []

    async def timeout_once(**_kwargs):
        calls.append('edit')
        raise asyncio.TimeoutError('simulated provider timeout')

    async def must_not_generate(**_kwargs):
        raise AssertionError('an edit timeout must not be replaced by text-to-image')

    monkeypatch.setattr(builtin.Config, 'get', lambda *_args, **_kwargs: _async_value(True))
    monkeypatch.setattr(builtin, 'image_edits', timeout_once)
    monkeypatch.setattr(builtin, 'image_generations', must_not_generate)

    result = json.loads(
        await builtin.edit_image(
            prompt='美化这张照片',
            image_urls=['/api/v1/files/original/content'],
            __request__=object(),
        )
    )

    assert calls == ['edit']
    assert 'error' in result
    assert 'images' not in result


@pytest.mark.asyncio
async def test_candidate_router_timeout_dispatches_once_and_keeps_outcome_unknown(monkeypatch, tmp_path):
    """The actual image router does not retry a timeout, but cannot know whether the upstream completed."""
    source = tmp_path / 'timeout-source.jpg'
    source.write_bytes(b'timeout source')
    upstream_calls = []

    async def fake_get_file_content(_file_id, _user):
        return FileResponse(source)

    async def fake_get_session():
        return _TimeoutUpstreamSession(upstream_calls)

    async def fake_effective_key(*_args, **_kwargs):
        return 'test-key'

    monkeypatch.setattr(images_router, 'get_file_content_by_id', fake_get_file_content)
    monkeypatch.setattr(images_router, 'get_session', fake_get_session)
    monkeypatch.setattr(images_router, 'get_effective_openai_api_key', fake_effective_key)
    monkeypatch.setattr(
        images_router,
        'get_image_config',
        lambda: _async_value(
            SimpleNamespace(
                IMAGE_EDIT_SIZE='',
                IMAGE_EDIT_MODEL='gpt-image-2',
                IMAGE_EDIT_ENGINE='openai',
                IMAGES_EDIT_OPENAI_API_BASE_URL='http://simulated-upstream/v1',
                IMAGES_EDIT_OPENAI_API_KEY='configured-test-key',
                IMAGES_EDIT_OPENAI_API_VERSION='',
            )
        ),
    )

    with pytest.raises(HTTPException) as timeout:
        await images_router.image_edits(
            request=object(),
            form_data=images_router.EditImageForm(
                prompt='美化测试图片',
                image='/api/v1/files/timeout-source/content',
            ),
            user=_user('family-a'),
        )

    assert timeout.value.status_code == 400
    assert len(upstream_calls) == 1


@pytest.mark.asyncio
async def test_candidate_result_save_failure_is_visible_after_the_provider_returns(monkeypatch):
    provider_calls = []
    emitted = []

    async def provider_success(**_kwargs):
        provider_calls.append('edit')
        return [{'id': 'provider-image', 'url': '/api/v1/files/provider-image/content'}]

    async def fail_persist(*_args, **_kwargs):
        raise RuntimeError('simulated result persistence failure')

    async def emit(event):
        emitted.append(event)

    monkeypatch.setattr(builtin.Config, 'get', lambda *_args, **_kwargs: _async_value(True))
    monkeypatch.setattr(builtin, 'image_edits', provider_success)
    monkeypatch.setattr(builtin, 'is_saved_chat_id', lambda _chat_id: True)
    monkeypatch.setattr(builtin.Chats, 'add_message_files_by_id_and_message_id', fail_persist)

    result = json.loads(
        await builtin.edit_image(
            prompt='美化这张照片',
            image_urls=['/api/v1/files/original/content'],
            __request__=object(),
            __event_emitter__=emit,
            __chat_id__='saved-chat',
            __message_id__='assistant-message',
        )
    )

    assert provider_calls == ['edit']
    assert 'error' in result
    assert emitted == []


@pytest.mark.asyncio
async def test_candidate_reconnect_reads_existing_stream_without_invoking_upstream_again():
    chat_id = 'candidate-reconnect-chat'
    task_id = 'candidate-reconnect-task'
    started = asyncio.Event()
    release = asyncio.Event()
    upstream_calls = []

    async def simulated_upstream():
        upstream_calls.append('started')
        started.set()
        await release.wait()

    _, task = await task_registry.create_task(None, simulated_upstream(), id=chat_id, task_id=task_id)
    try:
        await started.wait()
        await task_registry.save_response_stream(
            None,
            task_id,
            chat_id,
            'assistant-message',
            'partial answer',
            [{'type': 'message', 'content': [{'type': 'output_text', 'text': 'partial answer'}]}],
        )

        streams = await task_registry.get_response_streams_by_chat_id(None, chat_id)
        reloaded = overlay_response_streams(
            {
                'chat': {
                    'history': {
                        'messages': {
                            'assistant-message': {
                                'role': 'assistant',
                                'content': '',
                                'done': True,
                            }
                        }
                    }
                }
            },
            streams,
        )

        assert upstream_calls == ['started']
        assert reloaded['chat']['history']['messages']['assistant-message']['content'] == 'partial answer'
        assert reloaded['chat']['history']['messages']['assistant-message']['done'] is False
    finally:
        release.set()
        await task
        await asyncio.sleep(0)
        task_registry.tasks.pop(task_id, None)
        task_registry.item_tasks.pop(chat_id, None)
        task_registry.response_streams.pop(task_id, None)


@pytest.mark.asyncio
async def test_task_registry_remains_an_ephemeral_runner_not_operation_idempotency():
    """Task registration is intentionally not the durable operation idempotency boundary."""
    chat_id = 'candidate-duplicate-chat'
    task_ids = ('candidate-duplicate-a', 'candidate-duplicate-b')
    upstream_calls = []

    async def simulated_chargeable_upstream():
        upstream_calls.append('called')

    created = []
    try:
        for task_id in task_ids:
            _, task = await task_registry.create_task(
                None,
                simulated_chargeable_upstream(),
                id=chat_id,
                task_id=task_id,
            )
            created.append(task)
        await asyncio.gather(*created)
        assert upstream_calls == ['called', 'called']
    finally:
        await asyncio.sleep(0)
        for task_id in task_ids:
            task_registry.tasks.pop(task_id, None)
            task_registry.response_streams.pop(task_id, None)
        task_registry.item_tasks.pop(chat_id, None)


@pytest.mark.asyncio
async def test_stopping_a_chat_cancels_every_registered_task_even_as_cleanup_mutates_the_registry():
    """Cancellation must iterate a snapshot, not a list its callbacks alter."""
    chat_id = 'candidate-stop-all-chat'
    started = [asyncio.Event(), asyncio.Event()]
    release = asyncio.Event()

    async def waiting_task(index: int):
        started[index].set()
        await release.wait()

    created = []
    try:
        for index in range(2):
            _, task = await task_registry.create_task(
                None,
                waiting_task(index),
                id=chat_id,
                task_id=f'candidate-stop-all-{index}',
            )
            created.append(task)
        await asyncio.gather(*(event.wait() for event in started))

        result = await task_registry.stop_item_tasks(None, chat_id)
        assert result['status'] is True
        await asyncio.gather(*created, return_exceptions=True)
        assert all(task.cancelled() for task in created)
        assert await task_registry.list_task_ids_by_item_id(None, chat_id) == []
    finally:
        release.set()
        await asyncio.gather(*created, return_exceptions=True)
        for index in range(2):
            task_registry.tasks.pop(f'candidate-stop-all-{index}', None)
            task_registry.response_streams.pop(f'candidate-stop-all-{index}', None)
        task_registry.item_tasks.pop(chat_id, None)


@pytest.mark.asyncio
async def test_candidate_file_acl_uses_real_sqlite_rows_for_two_users(monkeypatch, tmp_path):
    """Run the actual Files/has_access_to_file path against an isolated SQLite database."""
    database = tmp_path / 'candidate-acl.db'
    engine = create_async_engine(f'sqlite+aiosqlite:///{database}')
    alice_path = tmp_path / 'alice.jpg'
    bob_path = tmp_path / 'bob.jpg'
    alice_path.write_bytes(b'alice image')
    bob_path.write_bytes(b'bob image')

    # get_async_db_context only reuses an explicitly provided session when this flag is true.
    # The flag is scoped by monkeypatch, while the database itself is entirely temporary.
    monkeypatch.setattr(db_internal, 'DATABASE_ENABLE_SESSION_SHARING', True)
    monkeypatch.setattr(files_router.Storage, 'get_file', lambda path: path)

    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    async with AsyncSession(engine, expire_on_commit=False) as session:
        session.add_all(
            [
                File(
                    id='alice-file',
                    user_id='family-a',
                    filename='alice.jpg',
                    path=str(alice_path),
                    data={},
                    meta={'name': 'alice.jpg', 'content_type': 'image/jpeg'},
                    created_at=1,
                    updated_at=1,
                ),
                File(
                    id='bob-file',
                    user_id='family-b',
                    filename='bob.jpg',
                    path=str(bob_path),
                    data={},
                    meta={'name': 'bob.jpg', 'content_type': 'image/jpeg'},
                    created_at=1,
                    updated_at=1,
                ),
            ]
        )
        await session.commit()

        content_endpoint = next(
            route.endpoint
            for route in files_router.router.routes
            if route.path == '/{id}/content' and 'GET' in route.methods
        )
        alice_response = await content_endpoint('alice-file', user=_user('family-a'), db=session)
        bob_response = await content_endpoint('bob-file', user=_user('family-b'), db=session)

        assert isinstance(alice_response, FileResponse)
        assert isinstance(bob_response, FileResponse)
        with pytest.raises(HTTPException) as denial:
            await content_endpoint('alice-file', user=_user('family-b'), db=session)
        assert denial.value.status_code == 404

    await engine.dispose()


@pytest.mark.asyncio
async def test_candidate_personal_keys_are_distinct_exactly_matched_and_never_fall_back(monkeypatch):
    """Use the production routing function, with only its encrypted-session store simulated."""
    observed_sessions = []

    async def session_for_user(session_id, user_id):
        observed_sessions.append((session_id, user_id))
        if user_id == 'family-missing':
            return None
        return SimpleNamespace(
            provider=openai_router.SUB2API_KEY_SESSION_PROVIDER,
            token={'access_token': f'test-key-for-{user_id}', 'subject': user_id},
        )

    monkeypatch.setattr(openai_router, 'ENABLE_SUB2API_KEY_LOGIN', True)
    monkeypatch.setattr(openai_router, 'SUB2API_KEY_LOGIN_BASE_URL', 'http://sub2api:8080')
    monkeypatch.setattr(openai_router.OAuthSessions, 'get_session_by_id_and_user_id', session_for_user)

    alice = _user('family-a', oauth={'sub2api': {'subject': 'family-a'}})
    bob = _user('family-b', oauth={'sub2api': {'subject': 'family-b'}})
    missing = _user('family-missing', oauth={'sub2api': {'subject': 'family-missing'}})

    assert await openai_router.get_effective_openai_api_key(
        'http://sub2api:8080/v1', 'configured-service-key', alice,
        request=SimpleNamespace(cookies={'sub2api_key_session_id': 'alice-a'}, state=SimpleNamespace()),
    ) == 'test-key-for-family-a'
    assert await openai_router.get_effective_openai_api_key(
        'http://sub2api:8080/v1', 'configured-service-key', bob,
        request=SimpleNamespace(cookies={'sub2api_key_session_id': 'bob-a'}, state=SimpleNamespace()),
    ) == 'test-key-for-family-b'
    # Runtime calls carry a concrete child endpoint rather than the provider
    # base URL. They must retain the same member key without matching another
    # provider or a lookalike path.
    assert await openai_router.get_effective_openai_api_key(
        'http://sub2api:8080/v1/chat/completions', 'configured-service-key', alice,
        request=SimpleNamespace(cookies={'sub2api_key_session_id': 'alice-a'}, state=SimpleNamespace()),
    ) == 'test-key-for-family-a'
    assert openai_router.is_sub2api_key_login_connection('http://sub2api:8080/v1/models') is True
    assert openai_router.is_sub2api_key_login_connection('http://sub2api:8080/v1-not-a-match/models') is False

    # A different provider URL remains on its own configured credential and does not query the member session.
    assert await openai_router.get_effective_openai_api_key(
        'http://other-provider:8080/v1', 'other-provider-key', alice
    ) == 'other-provider-key'
    assert observed_sessions == [('alice-a', 'family-a'), ('bob-a', 'family-b'), ('alice-a', 'family-a')]

    with pytest.raises(HTTPException) as expired:
        await openai_router.get_effective_openai_api_key(
            'http://sub2api:8080/v1', 'configured-service-key', missing,
            request=SimpleNamespace(cookies={'sub2api_key_session_id': 'missing-a'}, state=SimpleNamespace()),
        )
    assert expired.value.status_code == 401
    assert observed_sessions == [
        ('alice-a', 'family-a'), ('bob-a', 'family-b'), ('alice-a', 'family-a'), ('missing-a', 'family-missing')
    ]
