"""Entry-level reliability acceptance checks for the YanChuan candidate.

The test harness deliberately preserves the public chat endpoint, request
parsing, chat/message/file persistence, task registration, and the native
streaming tool loop. It replaces authentication, configuration, socket
delivery, and the chat-model stream. The ordinary success path simulates the
image provider while writing a real ``File`` row; the three failure cases use
``routers.images.image_edits`` and replace only its network/storage fault
boundaries. This is a chat-entry-to-native-tool-loop test, not a provider test
with real credentials.
"""

from __future__ import annotations

import asyncio
import base64
import importlib
import json
from dataclasses import dataclass
from types import SimpleNamespace

import httpx
import pytest
from fastapi.responses import FileResponse, StreamingResponse
from open_webui.models.chat_messages import ChatMessage
from open_webui.models.chats import Chat, ChatForm, Chats
from open_webui.models.files import File, FileForm, Files
from open_webui.models.users import UserModel
from open_webui.routers import images as images_router
from open_webui.tools import builtin
from open_webui.utils import middleware as middleware_utils
from open_webui.utils import tools as tools_utils
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

main = importlib.import_module('open_webui.main')


def _user(user_id: str = 'reliability-user') -> UserModel:
    return UserModel(
        id=user_id,
        email=f'{user_id}@example.test',
        name=user_id,
        role='user',
        last_active_at=0,
        updated_at=0,
        created_at=0,
    )


def _model(model_id: str = 'reliability-model') -> dict:
    return {
        'id': model_id,
        'info': {
            'meta': {
                'capabilities': {
                    'builtin_tools': True,
                    'image_generation': True,
                    'file_context': False,
                    'citations': False,
                }
            }
        },
    }


async def _no_op(*_args, **_kwargs):
    return None


def _stream(events: list[dict]) -> StreamingResponse:
    async def body():
        for event in events:
            yield f'data: {json.dumps(event)}\n\n'.encode()
        yield b'data: [DONE]\n\n'

    return StreamingResponse(body(), media_type='text/event-stream')


def _edit_call(call_id: str = 'model-edit-call') -> dict:
    return {
        'name': 'edit_image',
        'call_id': call_id,
        'arguments': {
            'prompt': '把照片调暖一点',
            'image_urls': ['/api/v1/files/source/content'],
        },
    }


def _generate_call(call_id: str = 'model-generate-call') -> dict:
    return {
        'name': 'generate_image',
        'call_id': call_id,
        'arguments': {'prompt': '生成一张新的暖色风景插画'},
    }


def _invalid_edit_call(call_id: str = 'invalid-edit-call') -> dict:
    return {
        'name': 'edit_image',
        'call_id': call_id,
        'arguments': {'prompt': '把照片调暖一点', 'image_urls': []},
    }


def _tool_calls_stream(tool_calls: list[dict]) -> StreamingResponse:
    return _stream(
        [
            {
                'choices': [
                    {
                        'delta': {
                            'tool_calls': [
                                {
                                    'index': index,
                                    'id': tool_call['call_id'],
                                    'function': {
                                        'name': tool_call['name'],
                                        'arguments': json.dumps(tool_call['arguments']),
                                    },
                                }
                                for index, tool_call in enumerate(tool_calls)
                            ]
                        }
                    }
                ]
            }
        ]
    )


def _text_stream(text: str) -> StreamingResponse:
    return _stream([{'choices': [{'delta': {'content': text}}]}])


async def _create_empty_chat(chat_id: str, user: UserModel, model: dict) -> None:
    await Chats.insert_new_chat(
        chat_id,
        user.id,
        ChatForm(
            chat={
                'id': chat_id,
                'title': 'Reliability fixture',
                'models': [model['id']],
                'history': {'currentId': None, 'messages': {}},
                'messages': [],
                'files': [],
                'tags': [],
                'timestamp': 0,
            },
            variables={},
            folder_id=None,
        ),
    )


def _payload(
    model: dict,
    *,
    chat_id: str,
    user_message_id: str,
    assistant_message_id: str,
    content: str = '把这张照片调暖一点',
) -> dict:
    body = {
        'model': model['id'],
        'parent_id': 'previous-assistant',
        'user_message': {
            'id': user_message_id,
            'role': 'user',
            'content': content,
            'files': [],
        },
        'id': assistant_message_id,
        'session_id': 'reliability-browser-session',
        'features': {'image_generation': True},
        'messages': [],
        'files': [],
    }
    if chat_id:
        body['chat_id'] = chat_id
    else:
        body['parent_id'] = None
    return body


async def _counts(engine: object) -> tuple[int, int, int]:
    async with AsyncSession(engine) as session:
        chats = await session.scalar(select(func.count()).select_from(Chat))
        messages = await session.scalar(select(func.count()).select_from(ChatMessage))
        files = await session.scalar(select(func.count()).select_from(File))
    return int(chats or 0), int(messages or 0), int(files or 0)


@dataclass(frozen=True)
class _Observation:
    chats: int
    messages: int
    background_tasks: int
    chat_upstream_calls: int
    image_upstream_calls: int
    saved_files: int
    assistant_files: int
    client_responses: tuple[dict, ...]
    completion_events: tuple[dict, ...]


class _EntryHarness:
    def __init__(
        self,
        engine,
        tmp_path,
        monkeypatch,
        *,
        initial_tool_calls: list[dict] | None = None,
        continuation_steps: list[dict] | None = None,
        image_mode: str = 'success',
        fail_message_association: bool = False,
        router_backed_image_provider: bool = False,
        socket_event_emitter: bool = True,
    ):
        self.engine = engine
        self.tmp_path = tmp_path
        self.monkeypatch = monkeypatch
        self.model = _model()
        self.models = {self.model['id']: self.model}
        self.user = _user()
        self.initial_tool_calls = initial_tool_calls or [_edit_call()]
        self.continuation_steps = list(continuation_steps or [{'kind': 'text', 'text': '图片已处理'}])
        self.image_mode = image_mode
        self.fail_message_association = fail_message_association
        self.router_backed_image_provider = router_backed_image_provider
        self.socket_event_emitter = socket_event_emitter
        self.created_tasks: list[tuple[str, str | None, asyncio.Task]] = []
        self.chat_upstream_calls: list[dict] = []
        self.image_upstream_calls: list[dict] = []
        self.events: list[dict] = []
        self.responses: list[dict] = []
        self._remaining_message_association_failures = 1 if fail_message_association else 0
        self._configure()

    def _configure(self) -> None:  # noqa: C901 - a linear test-double wiring boundary
        async def get_config(key, default=None):
            return {
                'models.default_params': {},
                'chat.tool_permissions.enable': False,
                'images.edit.enable': True,
                'image_generation.enable': True,
                'user.permissions': {},
                'task.model.default': '',
                'task.model.external': '',
                'code_interpreter.enable': False,
                'tool_server.connections': [],
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

        async def no_compaction(_request, _user, messages, _metadata, _model_id, _models, _system_prompt):
            return messages, None, None

        async def no_rag(form_data, _extra_params, _user):
            return form_data, {'sources': []}

        async def pass_through_form_data(form_data, **_kwargs):
            return form_data

        async def pass_through_pipeline(_request, form_data, *_args, **_kwargs):
            return form_data

        async def emit(event):
            self.events.append(event)

        async def event_factory(_metadata, **_kwargs):
            return emit

        async def no_model_record(*_args, **_kwargs):
            return None

        async def no_access_check(*_args, **_kwargs):
            return None

        async def no_event_emitter(*_args, **_kwargs):
            return None

        async def initial_model_stream(_request, form_data, _user):
            self.chat_upstream_calls.append({'phase': 'initial', 'messages': form_data.get('messages', [])})
            return _tool_calls_stream(self.initial_tool_calls)

        async def continuation_model_stream(_request, form_data, _user, **_kwargs):
            self.chat_upstream_calls.append({'phase': 'continuation', 'messages': form_data.get('messages', [])})
            step = self.continuation_steps.pop(0) if self.continuation_steps else {'kind': 'text', 'text': '图片已处理'}
            if step['kind'] == 'tools':
                return _tool_calls_stream(step['tool_calls'])
            return _text_stream(step.get('text', '图片已处理'))

        async def store_image(*, form_data, user, kind: str):
            file_id = f'provider-{kind}-{len(self.image_upstream_calls)}'
            path = self.tmp_path / f'{file_id}.png'
            path.write_bytes(b'simulated-image-bytes')
            stored = await Files.insert_new_file(
                user.id,
                FileForm(
                    id=file_id,
                    filename=path.name,
                    path=str(path),
                    data={},
                    meta={'name': path.name, 'content_type': 'image/png'},
                ),
            )
            return [
                {
                    'id': stored.id,
                    'url': f'/api/v1/files/{stored.id}/content',
                    'content_type': 'image/png',
                }
            ]

        async def simulated_image_provider(*, form_data, user, **_kwargs):
            call = {'kind': 'edit', 'prompt': form_data.prompt, 'user_id': user.id, 'phase': 'dispatched'}
            self.image_upstream_calls.append(call)
            if self.image_mode == 'timeout_unknown':
                call['phase'] = 'sent_result_unknown'
                raise TimeoutError('simulated timeout after image request dispatch')
            if self.image_mode == 'file_save_failure':
                call['phase'] = 'upstream_success_file_save_failed'
                raise OSError('simulated image file save failure after upstream success')
            call['phase'] = 'stored'
            return await store_image(form_data=form_data, user=user, kind='edit')

        async def simulated_generation_provider(*, form_data, user, **_kwargs):
            self.image_upstream_calls.append(
                {'kind': 'generate', 'prompt': form_data.prompt, 'user_id': user.id, 'phase': 'stored'}
            )
            return await store_image(form_data=form_data, user=user, kind='generate')

        real_message_association = builtin.Chats.add_message_files_by_id_and_message_id

        async def fail_message_association(*args, **kwargs):
            if self._remaining_message_association_failures:
                self._remaining_message_association_failures -= 1
                raise RuntimeError('simulated chat-message attachment failure after file storage')
            return await real_message_association(*args, **kwargs)

        class RouterResponse:
            async def __aenter__(inner_self):
                if self.image_mode == 'timeout_unknown':
                    raise TimeoutError('simulated timeout after image request dispatch')
                return inner_self

            async def __aexit__(inner_self, *_args):
                return False

            def raise_for_status(inner_self):
                return None

            async def json(inner_self, **_kwargs):
                return {'data': [{'b64_json': 'router-simulated-image'}]}

        class RouterSession:
            def post(inner_self, **kwargs):
                self.image_upstream_calls.append({'kind': 'edit', 'phase': 'dispatched', 'url': kwargs['url']})
                return RouterResponse()

        async def router_get_session():
            return RouterSession()

        async def router_source_file(*_args, **_kwargs):
            source = self.tmp_path / 'router-source.jpg'
            source.write_bytes(b'router source image')
            return FileResponse(source)

        async def router_image_data(value, *_args, **_kwargs):
            assert value == 'router-simulated-image'
            return b'router generated image', 'image/png'

        async def router_upload(_request, image_data, content_type, _metadata, user, db=None):
            if self.image_mode == 'file_save_failure':
                self.image_upstream_calls[-1]['phase'] = 'upstream_success_file_save_failed'
                raise OSError('simulated image file save failure after upstream success')
            self.image_upstream_calls[-1]['phase'] = 'stored'
            image = (await store_image(form_data=None, user=user, kind='router-edit'))[0]
            return SimpleNamespace(id=image['id']), image

        async def router_key(*_args, **_kwargs):
            return 'offline-router-key'

        async def router_config():
            return SimpleNamespace(
                IMAGE_EDIT_SIZE='',
                IMAGE_EDIT_MODEL='offline-edit-model',
                IMAGE_EDIT_ENGINE='openai',
                IMAGES_EDIT_OPENAI_API_BASE_URL='http://offline-image-provider/v1',
                IMAGES_EDIT_OPENAI_API_KEY='offline-router-key',
                IMAGES_EDIT_OPENAI_API_VERSION='',
            )

        real_create_task = main.create_task

        async def recording_create_task(redis, coroutine, id=None, task_id=None):
            created_id, task = await real_create_task(redis, coroutine, id=id, task_id=task_id)
            self.created_tasks.append((created_id, id, task))
            return created_id, task

        self.monkeypatch.setattr(main.Config, 'get', get_config)
        self.monkeypatch.setattr(main.Config, 'get_many', get_many)
        self.monkeypatch.setattr(main.Models, 'get_model_by_id', no_model_record)
        self.monkeypatch.setattr(main, 'check_model_access', no_access_check)
        self.monkeypatch.setattr(main, 'publish_event', _no_op)
        self.monkeypatch.setattr(main, 'emit_chat_list_event', _no_op)
        self.monkeypatch.setattr(main, 'get_event_emitter', no_event_emitter)
        self.monkeypatch.setattr(main, 'chat_completion_handler', initial_model_stream)
        self.monkeypatch.setattr(main, 'create_task', recording_create_task)

        self.monkeypatch.setattr(middleware_utils, 'has_permission', permitted)
        self.monkeypatch.setattr(tools_utils, 'has_permission', permitted)
        self.monkeypatch.setattr(
            middleware_utils,
            'get_event_emitter',
            event_factory if self.socket_event_emitter else no_event_emitter,
        )
        self.monkeypatch.setattr(middleware_utils, 'get_event_call', no_event_emitter)
        self.monkeypatch.setattr(middleware_utils, 'get_system_oauth_token', _no_op)
        self.monkeypatch.setattr(middleware_utils, 'get_task_model_id', lambda model_id, *_args: model_id)
        self.monkeypatch.setattr(
            middleware_utils,
            'convert_url_images_to_base64',
            pass_through_form_data,
        )
        self.monkeypatch.setattr(middleware_utils, 'compact_messages_for_request', no_compaction)
        self.monkeypatch.setattr(
            middleware_utils,
            'process_pipeline_inlet_filter',
            pass_through_pipeline,
        )
        self.monkeypatch.setattr(middleware_utils, 'chat_completion_files_handler', no_rag)
        self.monkeypatch.setattr(middleware_utils, 'resolve_system_prompt', _no_op)
        self.monkeypatch.setattr(middleware_utils, 'add_file_context', lambda messages, *_args: _async_value(messages))
        self.monkeypatch.setattr(middleware_utils, 'generate_chat_completion', continuation_model_stream)
        self.monkeypatch.setattr(middleware_utils, 'publish_chat_finished_event', _no_op)
        self.monkeypatch.setattr(middleware_utils, 'background_tasks_handler', _no_op)
        self.monkeypatch.setattr(middleware_utils, 'outlet_filter_handler', _no_op)
        self.monkeypatch.setattr(middleware_utils, 'ENABLE_PLUGINS', False)
        self.monkeypatch.setattr(
            builtin,
            'image_edits',
            images_router.image_edits if self.router_backed_image_provider else simulated_image_provider,
        )
        self.monkeypatch.setattr(builtin, 'image_generations', simulated_generation_provider)
        if self.router_backed_image_provider:
            self.monkeypatch.setattr(images_router, 'get_session', router_get_session)
            self.monkeypatch.setattr(images_router, 'get_file_content_by_id', router_source_file)
            self.monkeypatch.setattr(images_router, 'get_image_data', router_image_data)
            self.monkeypatch.setattr(images_router, 'upload_image', router_upload)
            self.monkeypatch.setattr(images_router, 'get_effective_openai_api_key', router_key)
            self.monkeypatch.setattr(images_router, 'get_image_config', router_config)
        if self.fail_message_association:
            self.monkeypatch.setattr(builtin.Chats, 'add_message_files_by_id_and_message_id', fail_message_association)

        self.monkeypatch.setattr(main.app.state, 'MODELS', self.models, raising=False)
        self.monkeypatch.setattr(main.app.state, 'redis', None, raising=False)
        self.monkeypatch.setitem(main.app.dependency_overrides, main.get_verified_user, lambda: self.user)

    async def submit(
        self,
        body: dict,
        *,
        headers: dict[str, str] | None = None,
        capture_response: bool = True,
        expected_status: int | None = 200,
        include_status: bool = False,
    ) -> dict | tuple[int, dict]:
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url='http://testserver') as client:
            response = await client.post('/api/chat/completions', json=body, headers=headers)
        if expected_status is not None:
            assert response.status_code == expected_status, response.text
        payload = response.json()
        if capture_response:
            self.responses.append(payload)
        if include_status:
            return response.status_code, payload
        return payload

    async def wait_for_tasks(self, start: int = 0) -> None:
        tasks = [task for _, _, task in self.created_tasks[start:]]
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            assert not [result for result in results if isinstance(result, BaseException)], results

    async def observe(self, chat_id: str) -> _Observation:
        chats, messages, files = await _counts(self.engine)
        chat = await Chats.get_chat_by_id(chat_id)
        assistant_files = 0
        if chat:
            history = (chat.chat or {}).get('history', {}).get('messages', {})
            assistant_files = sum(
                len(message.get('files') or [])
                for message in history.values()
                if message.get('role') == 'assistant'
            )
        return _Observation(
            chats=chats,
            messages=messages,
            background_tasks=len(self.created_tasks),
            chat_upstream_calls=len(self.chat_upstream_calls),
            image_upstream_calls=len(self.image_upstream_calls),
            saved_files=files,
            assistant_files=assistant_files,
            client_responses=tuple(self.responses),
            completion_events=tuple(
                event['data'] for event in self.events if event.get('type') == 'chat:completion'
            ),
        )


async def _async_value(value):
    return value


def _assert_completed_replay_response(
    original_response: dict,
    replay_response: dict,
    *,
    chat_id: str,
    assistant_message_id: str,
    expected_files: int,
) -> None:
    """Contract for the later durable-operation response, not the current legacy task response."""

    assert replay_response['chat_id'] == chat_id
    assert replay_response['operation_id'] == original_response['operation_id']
    assert replay_response['operation_status'] == 'COMPLETED'
    assert replay_response['result']['chat_id'] == chat_id
    assert replay_response['result']['assistant_message_id'] == assistant_message_id
    assert len(replay_response['result']['files']) == expected_files


@pytest.mark.asyncio
async def test_post_response_helpers_do_not_keep_the_primary_chat_task_or_operation_running(
    isolated_runtime_db, tmp_path, monkeypatch
):
    """A hung title/tag helper is independent of the already-durable reply."""
    from open_webui.models.operations import ChatOperation, OPERATION_COMPLETED

    harness = _EntryHarness(isolated_runtime_db, tmp_path, monkeypatch)
    harness.initial_tool_calls = []
    chat_id = 'background-helper-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)
    helper_started = asyncio.Event()
    release_helper = asyncio.Event()

    async def completed_text_stream(_request, _form_data, _user):
        harness.chat_upstream_calls.append({'phase': 'initial-text'})
        return _text_stream('已完成')

    monkeypatch.setattr(main, 'chat_completion_handler', completed_text_stream)

    async def hanging_helper(_ctx):
        helper_started.set()
        await release_helper.wait()

    monkeypatch.setattr(middleware_utils, 'background_tasks_handler', hanging_helper)
    try:
        response = await harness.submit(
            _payload(
                harness.model,
                chat_id=chat_id,
                user_message_id='background-helper-user',
                assistant_message_id='background-helper-assistant',
                content='只回复一句已完成',
            ),
            headers={'Idempotency-Key': 'background-helper-operation'},
        )
        await asyncio.wait_for(harness.wait_for_tasks(), timeout=1)
        await asyncio.wait_for(helper_started.wait(), timeout=1)

        async with AsyncSession(isolated_runtime_db) as session:
            operation = await session.get(ChatOperation, response['operation_id'])
            assert operation is not None
            assert operation.status == OPERATION_COMPLETED
    finally:
        release_helper.set()
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_interrupted_native_tool_stream_becomes_unknown_without_starting_an_image_call(
    isolated_runtime_db, tmp_path, monkeypatch
):
    """An incomplete stream must not leave a RUNNING operation or execute a partial tool call."""
    from open_webui.models.operations import ChatOperation, OPERATION_UNKNOWN

    harness = _EntryHarness(isolated_runtime_db, tmp_path, monkeypatch)
    chat_id = 'interrupted-native-stream-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)

    async def interrupted_stream(_request, _form_data, _user):
        harness.chat_upstream_calls.append({'phase': 'initial-interrupted'})

        async def body():
            yield (
                'data: '
                + json.dumps(
                    {
                        'choices': [
                            {
                                'delta': {
                                    'tool_calls': [
                                        {
                                            'index': 0,
                                            'id': 'partial-edit-call',
                                            'function': {
                                                'name': 'edit_image',
                                                'arguments': '{"prompt":"only change the cup color"}',
                                            },
                                        }
                                    ]
                                }
                            }
                        ]
                    }
                )
                + '\n\n'
            ).encode()
            raise TimeoutError('simulated upstream stream idle timeout')

        return StreamingResponse(body(), media_type='text/event-stream')

    monkeypatch.setattr(main, 'chat_completion_handler', interrupted_stream)
    body = _payload(
        harness.model,
        chat_id=chat_id,
        user_message_id='interrupted-native-stream-user',
        assistant_message_id='interrupted-native-stream-assistant',
        content='把这张照片的杯子改成红色',
    )
    response = await harness.submit(body, headers={'Idempotency-Key': 'interrupted-native-stream-operation'})
    await harness.wait_for_tasks()

    async with AsyncSession(isolated_runtime_db) as session:
        operation = await session.get(ChatOperation, response['operation_id'])
        assert operation is not None
        assert operation.status == OPERATION_UNKNOWN
        assert operation.result['code'] == 'operation_result_unknown'

    replay = await harness.submit(body, headers={'Idempotency-Key': 'interrupted-native-stream-operation'})
    assert replay['operation_status'] == OPERATION_UNKNOWN
    assert len(harness.created_tasks) == 1
    assert len(harness.chat_upstream_calls) == 1
    assert harness.image_upstream_calls == []


@pytest.mark.asyncio
async def test_terminal_empty_stream_cannot_complete_an_empty_assistant_placeholder(
    isolated_runtime_db, tmp_path, monkeypatch
):
    """A 200 + [DONE] without a response must not become a completed operation."""
    from open_webui.models.operations import ChatOperation, OPERATION_UNKNOWN

    harness = _EntryHarness(isolated_runtime_db, tmp_path, monkeypatch)
    chat_id = 'terminal-empty-stream-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)

    async def terminal_empty_stream(_request, _form_data, _user):
        harness.chat_upstream_calls.append({'phase': 'initial-empty-terminal'})

        async def body():
            yield b'data: [DONE]\n\n'

        return StreamingResponse(body(), media_type='text/event-stream')

    monkeypatch.setattr(main, 'chat_completion_handler', terminal_empty_stream)
    response = await harness.submit(
        _payload(
            harness.model,
            chat_id=chat_id,
            user_message_id='terminal-empty-stream-user',
            assistant_message_id='terminal-empty-stream-assistant',
            content='请编辑这张图片',
        ),
        headers={'Idempotency-Key': 'terminal-empty-stream-operation'},
    )
    await harness.wait_for_tasks()

    async with AsyncSession(isolated_runtime_db) as session:
        operation = await session.get(ChatOperation, response['operation_id'])
        assert operation is not None
        assert operation.status == OPERATION_UNKNOWN
        assert operation.result['code'] == 'operation_result_unknown'

    assistant = await Chats.get_message_by_id_and_message_id(chat_id, 'terminal-empty-stream-assistant')
    assert assistant is not None
    assert assistant.get('error')
    assert assistant.get('content', '') == ''
    assert harness.image_upstream_calls == []


@pytest.mark.asyncio
async def test_stream_without_terminal_event_becomes_unknown_even_after_text_delta(
    isolated_runtime_db, tmp_path, monkeypatch
):
    """A partial visible delta is not a confirmed finished model response."""
    from open_webui.models.operations import ChatOperation, OPERATION_UNKNOWN

    harness = _EntryHarness(isolated_runtime_db, tmp_path, monkeypatch)
    chat_id = 'missing-terminal-stream-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)

    async def missing_terminal_stream(_request, _form_data, _user):
        harness.chat_upstream_calls.append({'phase': 'initial-missing-terminal'})

        async def body():
            yield b'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'

        return StreamingResponse(body(), media_type='text/event-stream')

    monkeypatch.setattr(main, 'chat_completion_handler', missing_terminal_stream)
    response = await harness.submit(
        _payload(
            harness.model,
            chat_id=chat_id,
            user_message_id='missing-terminal-stream-user',
            assistant_message_id='missing-terminal-stream-assistant',
            content='请说明结果',
        ),
        headers={'Idempotency-Key': 'missing-terminal-stream-operation'},
    )
    await harness.wait_for_tasks()

    async with AsyncSession(isolated_runtime_db) as session:
        operation = await session.get(ChatOperation, response['operation_id'])
        assert operation is not None
        assert operation.status == OPERATION_UNKNOWN
        assert operation.result['code'] == 'operation_result_unknown'

    assistant = await Chats.get_message_by_id_and_message_id(chat_id, 'missing-terminal-stream-assistant')
    assert assistant is not None
    assert assistant.get('error')
    assert harness.image_upstream_calls == []


@pytest.mark.asyncio
async def test_malformed_upstream_tool_frame_becomes_unknown_without_executing_it(
    isolated_runtime_db, tmp_path, monkeypatch
):
    """A parser failure cannot silently drop a native tool command and complete."""
    from open_webui.models.operations import ChatOperation, OPERATION_UNKNOWN

    harness = _EntryHarness(isolated_runtime_db, tmp_path, monkeypatch)
    chat_id = 'malformed-tool-frame-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)

    async def malformed_tool_stream(_request, _form_data, _user):
        harness.chat_upstream_calls.append({'phase': 'initial-malformed-tool-frame'})

        async def body():
            yield b'data: {"choices":[{"delta":{"tool_calls":[\n\n'

        return StreamingResponse(body(), media_type='text/event-stream')

    monkeypatch.setattr(main, 'chat_completion_handler', malformed_tool_stream)
    response = await harness.submit(
        _payload(
            harness.model,
            chat_id=chat_id,
            user_message_id='malformed-tool-frame-user',
            assistant_message_id='malformed-tool-frame-assistant',
            content='只改杯子颜色',
        ),
        headers={'Idempotency-Key': 'malformed-tool-frame-operation'},
    )
    await harness.wait_for_tasks()

    async with AsyncSession(isolated_runtime_db) as session:
        operation = await session.get(ChatOperation, response['operation_id'])
        assert operation is not None
        assert operation.status == OPERATION_UNKNOWN
        assert operation.result['code'] == 'operation_result_unknown'

    assistant = await Chats.get_message_by_id_and_message_id(chat_id, 'malformed-tool-frame-assistant')
    assert assistant is not None
    assert assistant.get('error')
    assert harness.image_upstream_calls == []


@pytest.mark.asyncio
async def test_sse_keepalives_do_not_extend_a_stalled_native_tool_operation(
    isolated_runtime_db, tmp_path, monkeypatch
):
    """Connection keepalives are not a result and must not keep a paid path alive."""
    from open_webui.models.operations import ChatOperation, OPERATION_UNKNOWN

    harness = _EntryHarness(isolated_runtime_db, tmp_path, monkeypatch)
    chat_id = 'keepalive-stalled-native-stream-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)
    monkeypatch.setattr(middleware_utils, 'AIOHTTP_CLIENT_STREAM_IDLE_TIMEOUT', 0.05)

    async def keepalive_stalled_stream(_request, _form_data, _user):
        harness.chat_upstream_calls.append({'phase': 'initial-keepalive-stalled'})

        async def body():
            yield b'data: {"choices":[{"delta":{"content":"working"}}]}\n\n'
            while True:
                await asyncio.sleep(0.01)
                yield b': transport keepalive\n\n'

        return StreamingResponse(body(), media_type='text/event-stream')

    monkeypatch.setattr(main, 'chat_completion_handler', keepalive_stalled_stream)
    body = _payload(
        harness.model,
        chat_id=chat_id,
        user_message_id='keepalive-stalled-native-stream-user',
        assistant_message_id='keepalive-stalled-native-stream-assistant',
        content='编辑图片但不要执行不完整的调用',
    )
    response = await harness.submit(body, headers={'Idempotency-Key': 'keepalive-stalled-native-stream-operation'})
    await asyncio.wait_for(harness.wait_for_tasks(), timeout=1)

    async with AsyncSession(isolated_runtime_db) as session:
        operation = await session.get(ChatOperation, response['operation_id'])
        assert operation is not None
        assert operation.status == OPERATION_UNKNOWN
        assert operation.result['code'] == 'operation_result_unknown'
    assert harness.image_upstream_calls == []


@pytest.mark.asyncio
async def test_cancelling_an_unfinished_provider_stream_marks_the_operation_unknown(
    isolated_runtime_db, tmp_path, monkeypatch
):
    """A local stop cannot claim a provider that already received work did nothing."""
    from open_webui.models.operations import ChatOperation, OPERATION_UNKNOWN

    harness = _EntryHarness(isolated_runtime_db, tmp_path, monkeypatch)
    chat_id = 'cancelled-native-stream-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)
    entered = asyncio.Event()
    release = asyncio.Event()

    async def unfinished_stream(_request, _form_data, _user):
        async def body():
            entered.set()
            yield b'data: {"choices":[{"delta":{"content":"working"}}]}\n\n'
            await release.wait()

        return StreamingResponse(body(), media_type='text/event-stream')

    monkeypatch.setattr(main, 'chat_completion_handler', unfinished_stream)
    body = _payload(
        harness.model,
        chat_id=chat_id,
        user_message_id='cancelled-native-stream-user',
        assistant_message_id='cancelled-native-stream-assistant',
        content='停止前不要把结果当作完成',
    )
    response = await harness.submit(body, headers={'Idempotency-Key': 'cancelled-native-stream-operation'})
    task = harness.created_tasks[-1][2]
    try:
        await asyncio.wait_for(entered.wait(), timeout=1)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await asyncio.sleep(0)

        async with AsyncSession(isolated_runtime_db) as session:
            operation = await session.get(ChatOperation, response['operation_id'])
            assert operation is not None
            assert operation.status == OPERATION_UNKNOWN
            assert operation.result['code'] == 'operation_result_unknown'
    finally:
        release.set()


@pytest.mark.asyncio
async def test_entry_serial_replay_reuses_model_image_and_saved_result(isolated_runtime_db, tmp_path, monkeypatch):
    """POST -> persistence -> task -> native tool loop; the first response is deliberately ignored before replay."""

    harness = _EntryHarness(isolated_runtime_db, tmp_path, monkeypatch)
    chat_id = 'existing-replay-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)
    body = _payload(
        harness.model,
        chat_id=chat_id,
        user_message_id='same-user-message',
        assistant_message_id='same-assistant-message',
    )
    headers = {'Idempotency-Key': 'operation-serial-response-loss'}

    # The server accepted this response, but the test intentionally discards it as a client-side response loss.
    first_response = await harness.submit(body, headers=headers, capture_response=False)
    await harness.wait_for_tasks()
    replay_response = await harness.submit(body, headers=headers)
    await harness.wait_for_tasks(start=1)

    observed = await harness.observe(chat_id)
    assert observed.chats == 1, observed
    assert observed.messages == 2, observed
    assert observed.background_tasks == 1, observed
    assert observed.chat_upstream_calls == 2, observed
    assert observed.image_upstream_calls == 1, observed
    assert observed.saved_files == 1, observed
    assert observed.assistant_files == 1, observed
    continuation = next(call for call in harness.chat_upstream_calls if call['phase'] == 'continuation')
    continuation_images = [
        part['image_url']['url']
        for message in continuation['messages']
        if message.get('role') == 'user' and isinstance(message.get('content'), list)
        for part in message['content']
        if part.get('type') == 'image_url' and isinstance(part.get('image_url'), dict)
    ]
    assert any(
        image_url.startswith('data:image/')
        and base64.b64decode(image_url.split(',', 1)[1]) == b'simulated-image-bytes'
        for image_url in continuation_images
    ), continuation
    assert len(observed.client_responses) == 1, observed
    assert {response['chat_id'] for response in observed.client_responses} == {chat_id}, observed
    _assert_completed_replay_response(
        first_response,
        replay_response,
        chat_id=chat_id,
        assistant_message_id='same-assistant-message',
        expected_files=1,
    )


@pytest.mark.asyncio
async def test_entry_without_socket_still_executes_native_tool_loop(
    isolated_runtime_db, tmp_path, monkeypatch
):
    """A persisted REST replay must not skip tools merely because no socket is attached."""

    harness = _EntryHarness(
        isolated_runtime_db,
        tmp_path,
        monkeypatch,
        socket_event_emitter=False,
    )
    chat_id = 'no-socket-native-tool-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)

    response = await harness.submit(
        _payload(
            harness.model,
            chat_id=chat_id,
            user_message_id='no-socket-native-tool-user',
            assistant_message_id='no-socket-native-tool-assistant',
        ),
        headers={'Idempotency-Key': 'operation-no-socket-native-tool'},
    )
    await harness.wait_for_tasks()

    observed = await harness.observe(chat_id)
    assert response['operation_status'] == 'RUNNING'
    assert observed.background_tasks == 1, observed
    assert observed.chat_upstream_calls == 2, observed
    assert observed.image_upstream_calls == 1, observed
    assert observed.saved_files == 1, observed
    assert observed.assistant_files == 1, observed


@pytest.mark.asyncio
async def test_entry_concurrent_replay_claims_one_paid_tool_loop(isolated_runtime_db, tmp_path, monkeypatch):
    harness = _EntryHarness(isolated_runtime_db, tmp_path, monkeypatch)
    chat_id = 'concurrent-replay-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)
    body = _payload(
        harness.model,
        chat_id=chat_id,
        user_message_id='concurrent-user-message',
        assistant_message_id='concurrent-assistant-message',
    )
    headers = {'Idempotency-Key': 'operation-concurrent-replay'}

    first_response, second_response = await asyncio.gather(
        harness.submit(body, headers=headers),
        harness.submit(body, headers=headers),
    )
    await harness.wait_for_tasks()

    observed = await harness.observe(chat_id)
    assert observed.chats == 1, observed
    assert observed.messages == 2, observed
    assert observed.background_tasks == 1, observed
    assert observed.chat_upstream_calls == 2, observed
    assert observed.image_upstream_calls == 1, observed
    assert observed.saved_files == 1, observed
    assert observed.assistant_files == 1, observed
    assert len(observed.client_responses) == 2, observed
    assert first_response['operation_id'] == second_response['operation_id']
    assert {response['operation_status'] for response in (first_response, second_response)} <= {
        'READY',
        'RUNNING',
        'COMPLETED',
    }

    completed_response = await harness.submit(body, headers=headers)
    assert len(harness.created_tasks) == 1
    _assert_completed_replay_response(
        first_response,
        completed_response,
        chat_id=chat_id,
        assistant_message_id='concurrent-assistant-message',
        expected_files=1,
    )


@pytest.mark.asyncio
async def test_entry_new_chat_response_loss_replay_reuses_one_chat_and_image(
    isolated_runtime_db, tmp_path, monkeypatch
):
    harness = _EntryHarness(isolated_runtime_db, tmp_path, monkeypatch)
    body = _payload(
        harness.model,
        chat_id='',
        user_message_id='new-chat-user-message',
        assistant_message_id='new-chat-assistant-message',
    )
    headers = {'Idempotency-Key': 'operation-new-chat-response-loss'}

    # Discard the accepted first HTTP response: a client that never learned chat_id retries its original body.
    first_response = await harness.submit(body, headers=headers, capture_response=False)
    await harness.wait_for_tasks()
    replay_response = await harness.submit(body, headers=headers)
    await harness.wait_for_tasks(start=1)

    replay_chat_id = harness.responses[0]['chat_id']
    observed = await harness.observe(replay_chat_id)
    assert observed.chats == 1, observed
    assert observed.messages == 2, observed
    assert observed.background_tasks == 1, observed
    assert observed.chat_upstream_calls == 2, observed
    assert observed.image_upstream_calls == 1, observed
    assert observed.saved_files == 1, observed
    assert observed.assistant_files == 1, observed
    assert first_response['chat_id'] == replay_chat_id
    assert {response['chat_id'] for response in observed.client_responses} == {replay_chat_id}, observed
    _assert_completed_replay_response(
        first_response,
        replay_response,
        chat_id=replay_chat_id,
        assistant_message_id='new-chat-assistant-message',
        expected_files=1,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('case', 'image_mode', 'fail_message_association', 'retry_call'),
    [
        ('sent-result-unknown-then-edit', 'timeout_unknown', False, _edit_call('retry-edit-call')),
        (
            'upstream-success-file-save-failure-then-generate',
            'file_save_failure',
            False,
            _generate_call('retry-generate-call'),
        ),
        ('stored-file-association-failure-then-generate', 'success', True, _generate_call('retry-generate-call')),
    ],
)
async def test_tool_loop_failure_blocks_a_second_paid_image_call(
    isolated_runtime_db,
    tmp_path,
    monkeypatch,
    case,
    image_mode,
    fail_message_association,
    retry_call,
):
    harness = _EntryHarness(
        isolated_runtime_db,
        tmp_path,
        monkeypatch,
        image_mode=image_mode,
        fail_message_association=fail_message_association,
        router_backed_image_provider=True,
        continuation_steps=[
            {'kind': 'tools', 'tool_calls': [retry_call]},
            {'kind': 'text', 'text': f'{case}-final-client-message'},
        ],
    )
    chat_id = f'tool-failure-{case}'
    await _create_empty_chat(chat_id, harness.user, harness.model)

    await harness.submit(
        _payload(
            harness.model,
            chat_id=chat_id,
            user_message_id=f'{case}-user-message',
            assistant_message_id=f'{case}-assistant-message',
        ),
        headers={'Idempotency-Key': f'operation-{case}'},
    )
    await harness.wait_for_tasks()

    observed = await harness.observe(chat_id)
    assert observed.chats == 1, observed
    assert observed.messages == 2, observed
    assert observed.background_tasks == 1, observed
    assert observed.image_upstream_calls == 1, observed
    assert observed.chat_upstream_calls == 3, observed
    assert len(observed.client_responses) == 1, observed
    assert observed.completion_events, observed


@pytest.mark.asyncio
async def test_tool_loop_reuses_same_tool_call_id_without_a_second_image_call(
    isolated_runtime_db, tmp_path, monkeypatch
):
    repeated_call = _edit_call('stable-recovery-call-id')
    harness = _EntryHarness(
        isolated_runtime_db,
        tmp_path,
        monkeypatch,
        initial_tool_calls=[repeated_call],
        continuation_steps=[
            {'kind': 'tools', 'tool_calls': [repeated_call]},
            {'kind': 'text', 'text': 'recovery replay final client message'},
        ],
    )
    chat_id = 'tool-call-recovery-chat'
    await _create_empty_chat(chat_id, harness.user, harness.model)

    await harness.submit(
        _payload(
            harness.model,
            chat_id=chat_id,
            user_message_id='tool-recovery-user-message',
            assistant_message_id='tool-recovery-assistant-message',
        ),
        headers={'Idempotency-Key': 'operation-tool-call-recovery'},
    )
    await harness.wait_for_tasks()

    observed = await harness.observe(chat_id)
    assert observed.chats == 1, observed
    assert observed.messages == 2, observed
    assert observed.background_tasks == 1, observed
    assert observed.image_upstream_calls == 1, observed
    assert observed.saved_files == 1, observed
    assert observed.assistant_files == 1, observed
    assert observed.completion_events, observed
