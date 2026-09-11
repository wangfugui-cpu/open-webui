import json
from types import SimpleNamespace

import pytest
from fastapi import HTTPException
from open_webui.models.users import UserModel
from open_webui.routers import images as images_router
from open_webui.routers import openai as openai_router
from open_webui.tools import builtin
from open_webui.utils import middleware as middleware_utils
from open_webui.utils import tools as tools_utils
from open_webui.utils.middleware import (
    get_current_user_image_references,
    should_handle_direct_image_generation,
)


async def _async_value(value):
    return value


def _native_test_user() -> UserModel:
    return UserModel(
        id='family-user',
        email='family-user@example.test',
        name='Family User',
        role='user',
        last_active_at=0,
        updated_at=0,
        created_at=0,
    )


async def _no_op_event(*_args, **_kwargs):
    return None


@pytest.mark.asyncio
async def test_legacy_image_http_endpoints_fail_closed_without_explicit_opt_in(monkeypatch) -> None:
    """The native tool path is protected by operations; legacy routes are not."""

    async def get_config(key, default=None):
        if key == 'images.direct_api.enable':
            return False
        return default

    monkeypatch.setattr(images_router.Config, 'get', get_config)
    user = _native_test_user()

    with pytest.raises(HTTPException) as generation_error:
        await images_router.generate_images(
            None,
            images_router.CreateImageForm(prompt='should not reach an image provider'),
            user,
        )
    assert generation_error.value.status_code == 403

    with pytest.raises(HTTPException) as edit_error:
        await images_router.edit_images(
            None,
            images_router.EditImageForm(
                image='data:image/png;base64,AA==',
                prompt='should not reach an image provider',
            ),
            user,
        )
    assert edit_error.value.status_code == 403


def _configure_native_payload_dependencies(
    monkeypatch, *, image_edit_enabled: bool, notes_enabled: bool = False
) -> None:
    async def get_config(key, default=None):
        return {
            'images.edit.enable': image_edit_enabled,
            'user.permissions': {},
            'task.model.default': '',
            'task.model.external': '',
        }.get(key, default)

    async def get_many(*_keys):
        return {
            'web.search.enable': False,
            'image_generation.enable': True,
            'images.edit.enable': image_edit_enabled,
            'code_interpreter.enable': False,
            'notes.enable': notes_enabled,
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

    async def event_emitter(_metadata):
        return _no_op_event

    monkeypatch.setattr(middleware_utils.Config, 'get', get_config)
    monkeypatch.setattr(middleware_utils.Config, 'get_many', get_many)
    monkeypatch.setattr(middleware_utils, 'has_permission', permitted)
    monkeypatch.setattr(tools_utils, 'has_permission', permitted)
    monkeypatch.setattr(middleware_utils, 'get_event_emitter', event_emitter)
    monkeypatch.setattr(middleware_utils, 'get_event_call', event_emitter)
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


def _native_model() -> dict:
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


def _native_request(model: dict) -> SimpleNamespace:
    return SimpleNamespace(
        state=SimpleNamespace(internal=False, direct=False),
        app=SimpleNamespace(state=SimpleNamespace(MODELS={model['id']: model})),
    )


def test_only_legacy_sessions_use_direct_image_generation() -> None:
    """A key-login native session must keep its native image tools."""
    assert should_handle_direct_image_generation({'params': {'function_calling': 'legacy'}}) is True
    assert should_handle_direct_image_generation({'params': {'function_calling': 'native'}}) is False
    assert should_handle_direct_image_generation({'params': {}}) is False
    assert should_handle_direct_image_generation(None) is False


def test_current_user_image_references_use_the_ui_message_not_rag_files() -> None:
    metadata = {
        # Mirrors the UI request: image stays on user_message.files, while the
        # top-level files payload contains only retrieval/document attachments.
        'user_message': {
            'role': 'user',
            'content': '美化这张照片',
            'files': [
                {'id': 'document', 'type': 'file', 'content_type': 'application/pdf'},
                {'id': 'original', 'type': 'file', 'content_type': 'image/jpeg', 'url': '/files/original'},
            ],
        },
        'files': [{'id': 'document', 'type': 'file', 'content_type': 'application/pdf'}],
    }

    assert get_current_user_image_references(metadata, []) == [
        {'id': 'original', 'url': '/files/original', 'content_type': 'image/jpeg'}
    ]
    assert all(not item['content_type'].startswith('image/') for item in metadata['files'])


def test_temporary_message_image_url_fallback_does_not_scan_history() -> None:
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': '旧图'},
                {'type': 'image_url', 'image_url': {'url': '/files/original'}},
            ],
        },
        {'role': 'assistant', 'content': '已收到'},
        {'role': 'user', 'content': '生成一张无关的新插画'},
    ]

    assert get_current_user_image_references({}, messages) == []


def test_authoritative_current_user_message_does_not_fall_back_to_history() -> None:
    """An image-free current request must not inherit an old chat attachment."""
    metadata = {'user_message': {'role': 'user', 'content': '生成一张全新的插画', 'files': []}}
    messages = [
        {
            'role': 'user',
            'content': '此前上传的照片',
            'files': [{'type': 'image', 'url': '/api/v1/files/original/content'}],
        }
    ]

    assert get_current_user_image_references(metadata, messages) == []


def test_current_user_message_can_use_its_matching_temporary_image_url() -> None:
    """Temporary/API request metadata may keep text while message content keeps the image."""
    metadata = {'user_message': {'role': 'user', 'content': '美化这张照片', 'files': []}}
    messages = [
        {
            'role': 'user',
            'content': [
                {'type': 'text', 'text': '美化这张照片'},
                {
                    'type': 'image_url',
                    'image_url': {'url': '/api/v1/files/original/content'},
                },
            ],
        }
    ]

    assert get_current_user_image_references(metadata, messages) == [
        {'url': '/api/v1/files/original/content', 'content_type': 'image/*'}
    ]


@pytest.mark.asyncio
async def test_persistent_page_request_rebuilds_current_image_and_executes_native_edit(monkeypatch) -> None:
    """Exercise UI-shaped request -> DB replay -> native tool wrapper -> execution."""
    _configure_native_payload_dependencies(monkeypatch, image_edit_enabled=False)
    model = _native_model()
    request = _native_request(model)
    user = _native_test_user()

    current_user_message = {
        'id': 'current-user-message',
        'role': 'user',
        'content': '美化这张照片，保留人物和背景',
        'files': [
            {
                'id': 'original',
                'type': 'file',
                'content_type': 'image/jpeg',
                'url': '/api/v1/files/original/content',
            }
        ],
    }

    async def load_from_db(*_args, **_kwargs):
        return [current_user_message.copy()]

    async def no_tool_result_processing(*_args, **_kwargs):
        return _args[2], [], []

    monkeypatch.setattr(middleware_utils, 'is_saved_chat_id', lambda _chat_id: True)
    monkeypatch.setattr(middleware_utils, 'load_messages_from_db', load_from_db)
    monkeypatch.setattr(middleware_utils.Chats, 'get_chat_folder_id', lambda *_args: _async_value(None))
    monkeypatch.setattr(middleware_utils.Chats, 'get_chat_by_id', lambda *_args: _async_value(None))
    monkeypatch.setattr(middleware_utils, 'process_tool_result', no_tool_result_processing)

    form_data, metadata, _events = await middleware_utils.process_chat_payload(
        request,
        {
            'model': model['id'],
            'messages': [],
            # Mirrors Chat.svelte: top-level files are retrieval files, not images.
            'files': [{'id': 'guide', 'type': 'file', 'content_type': 'application/pdf'}],
            'features': {'image_generation': True},
        },
        user,
        {
            'chat_id': 'saved-chat',
            'user_message_id': current_user_message['id'],
            'user_message': current_user_message,
            'message_id': 'assistant-message',
            'session_id': 'browser-session',
            'params': {'function_calling': 'native'},
        },
        model,
    )

    assert metadata['files'] == [{'id': 'guide', 'type': 'file', 'content_type': 'application/pdf'}]
    assert metadata['current_user_image_refs'] == [
        {
            'id': 'original',
            'url': '/api/v1/files/original/content',
            'content_type': 'image/jpeg',
        }
    ]
    assert form_data['messages'][0]['content'][1]['image_url']['url'] == '/api/v1/files/original/content'
    assert {'generate_image', 'edit_image'} <= metadata['tools'].keys()

    result = await middleware_utils.execute_tool_call_for_output(
        request,
        form_data,
        user,
        metadata,
        None,
        _no_op_event,
        {
            'id': 'edit-call',
            'function': {
                'name': 'edit_image',
                'arguments': json.dumps(
                    {
                        'prompt': '美化这张照片，保留人物和背景',
                        'image_urls': ['/api/v1/files/original/content'],
                    }
                ),
            },
        },
    )

    assert json.loads(result['content'])['error']['code'] == 'image_editing_unavailable'


@pytest.mark.asyncio
async def test_native_request_carries_note_tools_and_save_instruction_for_ordinary_user(monkeypatch) -> None:
    _configure_native_payload_dependencies(monkeypatch, image_edit_enabled=False, notes_enabled=True)
    model = _native_model()
    request = _native_request(model)
    user = _native_test_user()
    current_user_message = {
        'id': 'save-note-message',
        'role': 'user',
        'content': '把这份整理保存下来，方便继续修改',
        'files': [],
    }

    form_data, metadata, _events = await middleware_utils.process_chat_payload(
        request,
        {
            'model': model['id'],
            'messages': [current_user_message.copy()],
            'features': {'image_generation': False},
        },
        user,
        {
            'chat_id': '',
            'user_message': current_user_message,
            'message_id': 'save-note-assistant-message',
            'session_id': 'browser-session',
            'params': {'function_calling': 'native'},
        },
        model,
    )

    assert {'write_note', 'replace_note_content', 'export_note'} <= metadata['tools'].keys()
    assert {'write_note', 'replace_note_content', 'export_note'} <= {
        tool['function']['name'] for tool in form_data['tools']
    }
    assert any(
        message.get('role') == 'system' and 'call write_note' in str(message.get('content'))
        for message in form_data['messages']
    )


@pytest.mark.asyncio
async def test_temporary_page_request_allows_a_new_image_unrelated_to_current_attachment(monkeypatch) -> None:
    """A current upload is context, not an unconditional instruction to edit it."""
    _configure_native_payload_dependencies(monkeypatch, image_edit_enabled=False)
    model = _native_model()
    request = _native_request(model)
    user = _native_test_user()
    generated = []

    async def capture_generation(**kwargs):
        generated.append(kwargs)
        return [{'id': 'new-illustration', 'url': '/api/v1/files/new-illustration/content'}]

    async def no_tool_result_processing(*_args, **_kwargs):
        return _args[2], [], []

    monkeypatch.setattr(builtin, 'image_generations', capture_generation)
    monkeypatch.setattr(middleware_utils, 'is_saved_chat_id', lambda _chat_id: False)
    monkeypatch.setattr(middleware_utils, 'process_tool_result', no_tool_result_processing)

    current_user_message = {
        'id': 'temporary-user-message',
        'role': 'user',
        'content': '不要使用附件，生成一幅全新的鲸鱼插画',
        'files': [
            {
                'id': 'reference-photo',
                'type': 'file',
                'content_type': 'image/jpeg',
                'url': '/api/v1/files/reference-photo/content',
            }
        ],
    }

    form_data, metadata, _events = await middleware_utils.process_chat_payload(
        request,
        {
            'model': model['id'],
            # Mirrors the temporary-chat frontend payload after it converts the
            # attachment into OpenAI image_url message content.
            'messages': [
                {
                    'role': 'user',
                    'content': [
                        {'type': 'text', 'text': current_user_message['content']},
                        {
                            'type': 'image_url',
                            'image_url': {'url': '/api/v1/files/reference-photo/content'},
                        },
                    ],
                }
            ],
            'files': [],
            'features': {'image_generation': True},
        },
        user,
        {
            'chat_id': '',
            'user_message': current_user_message,
            'message_id': 'temporary-assistant-message',
            'session_id': 'browser-session',
            'params': {'function_calling': 'native'},
        },
        model,
    )

    assert metadata['current_user_image_refs'][0]['id'] == 'reference-photo'
    assert {'generate_image', 'edit_image'} <= metadata['tools'].keys()

    await middleware_utils.execute_tool_call_for_output(
        request,
        form_data,
        user,
        metadata,
        None,
        _no_op_event,
        {
            'id': 'new-image-call',
            'function': {
                'name': 'generate_image',
                'arguments': json.dumps({'prompt': '一幅全新的鲸鱼插画'}),
            },
        },
    )

    assert generated[0]['form_data'].prompt == '一幅全新的鲸鱼插画'


@pytest.mark.asyncio
async def test_native_image_tools_are_available_without_forcing_a_generation(monkeypatch) -> None:
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

    async def get_config(_key, _default=None):
        return {}

    async def permitted(*_args, **_kwargs):
        return True

    monkeypatch.setattr(tools_utils.Config, 'get_many', get_many)
    monkeypatch.setattr(tools_utils.Config, 'get', get_config)
    monkeypatch.setattr(tools_utils, 'has_permission', permitted)

    request = SimpleNamespace(state=SimpleNamespace(internal=False, direct=False))
    model = {'info': {'meta': {'capabilities': {'image_generation': True}}}}
    extra_params = {'__user__': {'id': 'family-user', 'role': 'user'}, '__metadata__': {}}

    tools = await tools_utils.get_builtin_tools(
        request,
        extra_params,
        features={'image_generation': True},
        model=model,
    )

    assert {'generate_image', 'edit_image'} <= tools.keys()
    # Registering the tools only exposes choices to the model. It does not execute either tool.


@pytest.mark.asyncio
async def test_native_note_tools_include_private_export_for_authorized_users(monkeypatch) -> None:
    async def get_many(*_keys):
        return {
            'web.search.enable': False,
            'image_generation.enable': False,
            'images.edit.enable': False,
            'code_interpreter.enable': False,
            'notes.enable': True,
            'channels.enable': False,
            'automations.enable': False,
            'calendar.enable': False,
            'ui.enable_user_webhooks': False,
            'subagents.enable': False,
            'subagents.background_enabled': False,
        }

    async def get_config(_key, _default=None):
        return {}

    async def permitted(*_args, **_kwargs):
        return True

    monkeypatch.setattr(tools_utils.Config, 'get_many', get_many)
    monkeypatch.setattr(tools_utils.Config, 'get', get_config)
    monkeypatch.setattr(tools_utils, 'has_permission', permitted)

    tools = await tools_utils.get_builtin_tools(
        SimpleNamespace(state=SimpleNamespace(internal=False, direct=False)),
        {'__user__': {'id': 'ordinary-user', 'role': 'user'}, '__metadata__': {}},
        features={},
        model={'info': {'meta': {'capabilities': {}}}},
    )

    assert {'search_notes', 'view_note', 'write_note', 'replace_note_content', 'export_note'} <= tools.keys()
    assert tools['export_note']['spec']['name'] == 'export_note'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('image_generation_enabled', 'image_edit_enabled', 'expected_tools'),
    [
        (False, False, set()),
        (False, True, {'edit_image'}),
        (True, False, {'generate_image', 'edit_image'}),
        (True, True, {'generate_image', 'edit_image'}),
    ],
)
async def test_native_image_tool_registration_respects_independent_provider_switches(
    monkeypatch,
    image_generation_enabled,
    image_edit_enabled,
    expected_tools,
) -> None:
    """Generation and editing switches must not hide the other provider's tool."""

    async def get_many(*_keys):
        return {
            'web.search.enable': False,
            'image_generation.enable': image_generation_enabled,
            'images.edit.enable': image_edit_enabled,
            'code_interpreter.enable': False,
            'notes.enable': False,
            'channels.enable': False,
            'automations.enable': False,
            'calendar.enable': False,
            'ui.enable_user_webhooks': False,
            'subagents.enable': False,
            'subagents.background_enabled': False,
        }

    async def get_config(_key, _default=None):
        return {}

    async def permitted(*_args, **_kwargs):
        return True

    monkeypatch.setattr(tools_utils.Config, 'get_many', get_many)
    monkeypatch.setattr(tools_utils.Config, 'get', get_config)
    monkeypatch.setattr(tools_utils, 'has_permission', permitted)

    tools = await tools_utils.get_builtin_tools(
        SimpleNamespace(state=SimpleNamespace(internal=False, direct=False)),
        {'__user__': {'id': 'family-user', 'role': 'user'}, '__metadata__': {}},
        features={'image_generation': True},
        model={'info': {'meta': {'capabilities': {'image_generation': True}}}},
    )

    assert {name for name in ('generate_image', 'edit_image') if name in tools} == expected_tools


@pytest.mark.asyncio
async def test_native_image_tools_respect_feature_and_model_switches(monkeypatch) -> None:
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

    async def get_config(_key, _default=None):
        return {}

    async def permitted(*_args, **_kwargs):
        return True

    monkeypatch.setattr(tools_utils.Config, 'get_many', get_many)
    monkeypatch.setattr(tools_utils.Config, 'get', get_config)
    monkeypatch.setattr(tools_utils, 'has_permission', permitted)

    request = SimpleNamespace(state=SimpleNamespace(internal=False, direct=False))
    extra_params = {'__user__': {'id': 'family-user', 'role': 'user'}, '__metadata__': {}}
    disabled_model = {
        'info': {
            'meta': {
                'capabilities': {'image_generation': True},
                'builtinTools': {'image_generation': False},
            }
        }
    }

    tools = await tools_utils.get_builtin_tools(
        request,
        extra_params,
        features={'image_generation': True},
        model=disabled_model,
    )
    assert 'generate_image' not in tools
    assert 'edit_image' not in tools


@pytest.mark.asyncio
async def test_edit_tool_uses_only_the_selected_second_source_image(monkeypatch) -> None:
    captured = []

    async def capture_edit(**kwargs):
        captured.append(kwargs)
        return [{'id': 'edited-v1', 'url': '/api/v1/files/edited-v1/content'}]

    monkeypatch.setattr(builtin, 'image_edits', capture_edit)
    monkeypatch.setattr(builtin.Config, 'get', lambda *_args, **_kwargs: _async_value(True))

    result = await builtin.edit_image(
        prompt='只调整第二张的亮度',
        image_urls=['/api/v1/files/second/content'],
        __request__=object(),
    )

    first_edit = json.loads(result)
    assert first_edit['images'] == [{'id': 'edited-v1', 'url': '/api/v1/files/edited-v1/content'}]
    assert captured[0]['form_data'].prompt == '只调整第二张的亮度'
    assert captured[0]['form_data'].image == ['/api/v1/files/second/content']

    await builtin.edit_image(
        prompt='再暖一点',
        image_urls=[first_edit['images'][0]['url']],
        __request__=object(),
    )
    assert captured[1]['form_data'].image == ['/api/v1/files/edited-v1/content']

    await builtin.edit_image(
        prompt='回到原图，只调整亮度',
        image_urls=['/api/v1/files/original/content'],
        __request__=object(),
    )
    assert captured[2]['form_data'].image == ['/api/v1/files/original/content']


@pytest.mark.asyncio
async def test_native_edit_persists_the_result_for_a_follow_up_turn(monkeypatch) -> None:
    stored_files = []
    events = []

    async def capture_edit(**_kwargs):
        return [{'id': 'edited-v1', 'url': '/api/v1/files/edited-v1/content'}]

    async def persist_files(chat_id, message_id, files):
        stored_files.append((chat_id, message_id, files))
        return [{**files[0], 'id': 'stored-edited-v1'}]

    async def emit(event):
        events.append(event)

    monkeypatch.setattr(builtin, 'image_edits', capture_edit)
    monkeypatch.setattr(builtin.Config, 'get', lambda *_args, **_kwargs: _async_value(True))
    monkeypatch.setattr(builtin, 'is_saved_chat_id', lambda _chat_id: True)
    monkeypatch.setattr(builtin.Chats, 'add_message_files_by_id_and_message_id', persist_files)

    result = await builtin.edit_image(
        prompt='再暖一点',
        image_urls=['/api/v1/files/edited-v0/content'],
        __request__=object(),
        __event_emitter__=emit,
        __chat_id__='saved-chat',
        __message_id__='assistant-message',
    )

    assert stored_files == [
        (
            'saved-chat',
            'assistant-message',
            [
                {
                    'type': 'image',
                    'id': 'edited-v1',
                    'url': '/api/v1/files/edited-v1/content',
                }
            ],
        )
    ]
    assert events[0]['data']['files'][0]['id'] == 'stored-edited-v1'
    assert json.loads(result)['images'][0]['url'] == '/api/v1/files/edited-v1/content'


@pytest.mark.asyncio
async def test_native_file_context_keeps_original_and_second_image_references_separate(monkeypatch) -> None:
    stored_messages = {
        'original-user': {
            'role': 'user',
            'content': '美化第一张',
            'files': [{'type': 'image', 'id': 'original', 'url': '/api/v1/files/original/content'}],
        },
        'assistant': {'role': 'assistant', 'content': '好的', 'parentId': 'original-user'},
        'second-user': {
            'role': 'user',
            'content': '只修改第二张',
            'parentId': 'assistant',
            'files': [{'type': 'image', 'id': 'second', 'url': '/api/v1/files/second/content'}],
        },
    }

    async def get_chat(*_args, **_kwargs):
        return SimpleNamespace(chat={'history': {'messages': stored_messages, 'currentId': 'second-user'}})

    monkeypatch.setattr(middleware_utils, 'is_saved_chat_id', lambda _chat_id: True)
    monkeypatch.setattr(middleware_utils.Chats, 'get_chat_by_id_and_user_id', get_chat)

    payload_messages = [
        {'role': 'user', 'content': '美化第一张'},
        {'role': 'assistant', 'content': '工具调用'},
        {'role': 'tool', 'content': '工具结果'},
        {'role': 'user', 'content': '只修改第二张'},
    ]

    result = await middleware_utils.add_file_context(
        payload_messages,
        'saved-chat',
        SimpleNamespace(id='family-user'),
    )

    first_user, second_user = result[0], result[-1]
    assert 'id="original"' in first_user['content']
    assert 'id="second"' not in first_user['content']
    assert 'id="second"' in second_user['content']
    assert 'id="original"' not in second_user['content']


@pytest.mark.asyncio
async def test_native_edit_fails_closed_for_current_source_when_editing_is_disabled(monkeypatch) -> None:
    async def get_config(key, _default=None):
        return {'images.edit.enable': False}.get(key, _default)

    async def generation_must_not_run(**_kwargs):
        raise AssertionError('current-turn source image must not be replaced by text-to-image')

    monkeypatch.setattr(builtin.Config, 'get', get_config)
    monkeypatch.setattr(builtin, 'image_generations', generation_must_not_run)

    result = await builtin.edit_image(
        prompt='美化这张照片，保留人物和背景',
        image_urls=['/api/v1/files/original/content'],
        __request__=object(),
        __current_user_image_refs__=[{'type': 'image', 'url': '/api/v1/files/original/content'}],
    )

    error = json.loads(result)['error']
    assert error['code'] == 'image_editing_unavailable'
    assert 'No replacement image was generated' in error['message']


@pytest.mark.asyncio
async def test_native_tool_wrapper_passes_current_attachment_to_edit(monkeypatch) -> None:
    async def get_many(*_keys):
        return {
            'web.search.enable': False,
            'image_generation.enable': True,
            'images.edit.enable': False,
            'code_interpreter.enable': False,
            'notes.enable': False,
            'channels.enable': False,
            'automations.enable': False,
            'calendar.enable': False,
            'ui.enable_user_webhooks': False,
            'subagents.enable': False,
            'subagents.background_enabled': False,
        }

    async def get_config(key, _default=None):
        if key == 'images.edit.enable':
            return False
        return {}

    async def permitted(*_args, **_kwargs):
        return True

    async def generation_must_not_run(**_kwargs):
        raise AssertionError('edit failure must not fall back to image generation')

    monkeypatch.setattr(tools_utils.Config, 'get_many', get_many)
    monkeypatch.setattr(tools_utils.Config, 'get', get_config)
    monkeypatch.setattr(tools_utils, 'has_permission', permitted)
    monkeypatch.setattr(builtin, 'image_generations', generation_must_not_run)

    request = SimpleNamespace(state=SimpleNamespace(internal=False, direct=False))
    native_tools = await tools_utils.get_builtin_tools(
        request,
        {
            '__user__': {'id': 'family-user', 'role': 'user'},
            '__metadata__': {
                'files': [],
            },
            '__current_user_image_refs__': [{'id': 'original', 'url': '/api/v1/files/original/content'}],
        },
        features={'image_generation': True},
        model={'info': {'meta': {'capabilities': {'image_generation': True}}}},
    )

    assert 'generate_image' in native_tools
    assert 'edit_image' in native_tools

    result = await native_tools['edit_image']['callable'](
        prompt='美化这张照片',
        image_urls=['/api/v1/files/original/content'],
    )
    assert json.loads(result)['error']['code'] == 'image_editing_unavailable'


@pytest.mark.asyncio
async def test_native_generator_allows_new_image_without_a_current_source_image(monkeypatch) -> None:
    captured = []

    async def get_config(key, _default=None):
        return {'images.edit.enable': False}.get(key, _default)

    async def capture_generation(**kwargs):
        captured.append(kwargs)
        return [{'id': 'new-illustration', 'url': '/api/v1/files/new-illustration/content'}]

    monkeypatch.setattr(builtin.Config, 'get', get_config)
    monkeypatch.setattr(builtin, 'image_generations', capture_generation)

    result = await builtin.generate_image(
        prompt='生成一幅全新的插画，不使用此前图片',
        __request__=object(),
    )

    assert json.loads(result)['images'][0]['id'] == 'new-illustration'
    assert captured[0]['form_data'].prompt == '生成一幅全新的插画，不使用此前图片'


@pytest.mark.asyncio
async def test_edit_tool_requires_a_source_and_never_falls_back_to_generation(monkeypatch) -> None:
    async def generation_must_not_run(**_kwargs):
        raise AssertionError('image generation must not be used as an edit fallback')

    monkeypatch.setattr(builtin, 'image_generations', generation_must_not_run)

    result = await builtin.edit_image(
        prompt='美化这张图',
        image_urls=[],
        __request__=object(),
    )

    error = json.loads(result)['error']
    assert error['code'] == 'image_reference_required'
    assert 'do not generate' in error['message']


@pytest.mark.asyncio
async def test_edit_tool_does_not_fall_back_when_the_edit_provider_fails(monkeypatch) -> None:
    async def fail_edit(**_kwargs):
        raise HTTPException(status_code=502, detail='upstream edit failed')

    async def generation_must_not_run(**_kwargs):
        raise AssertionError('image generation must not be used as an edit fallback')

    monkeypatch.setattr(builtin, 'image_edits', fail_edit)
    monkeypatch.setattr(builtin, 'image_generations', generation_must_not_run)
    monkeypatch.setattr(builtin.Config, 'get', lambda *_args, **_kwargs: _async_value(True))

    result = await builtin.edit_image(
        prompt='再暖一点',
        image_urls=['/api/v1/files/edited-v1/content'],
        __request__=object(),
    )

    assert 'upstream edit failed' in json.loads(result)['error']


@pytest.mark.asyncio
async def test_legacy_source_image_does_not_silently_fall_back_to_generation(monkeypatch) -> None:
    events = []

    async def get_config(key, _default=None):
        return {
            'images.edit.enable': False,
            'image_generation.enable': True,
        }.get(key, _default)

    async def emit(event):
        events.append(event)

    async def generation_must_not_run(**_kwargs):
        raise AssertionError('source-image edit must not fall back to image generation')

    monkeypatch.setattr(middleware_utils.Config, 'get', get_config)
    monkeypatch.setattr(middleware_utils, 'image_generations', generation_must_not_run)

    form_data = {
        'model': 'gpt-5.6-terra',
        'messages': [
            {
                'role': 'user',
                'content': '美化这张照片，保留人物和背景',
                'files': [{'type': 'image', 'url': '/api/v1/files/original/content'}],
            }
        ],
    }

    result = await middleware_utils.chat_image_generation_handler(
        request=object(),
        form_data=form_data,
        extra_params={
            '__metadata__': {'chat_id': 'local:test', 'message_id': 'message-1'},
            '__event_emitter__': emit,
        },
        user=None,
    )

    assert events[0]['data']['description'] == 'Image editing is disabled'
    assert 'No replacement image was generated' in result['messages'][0]['content']


@pytest.mark.asyncio
async def test_legacy_history_image_does_not_block_a_current_new_image_request(monkeypatch) -> None:
    captured = []

    async def get_config(key, _default=None):
        return {
            'images.edit.enable': False,
            'image_generation.enable': True,
            'image_generation.prompt.enable': False,
        }.get(key, _default)

    async def capture_generation(**kwargs):
        captured.append(kwargs)
        return [{'id': 'new-illustration', 'url': '/api/v1/files/new-illustration/content'}]

    monkeypatch.setattr(middleware_utils.Config, 'get', get_config)
    monkeypatch.setattr(middleware_utils, 'image_generations', capture_generation)

    form_data = {
        'model': 'gpt-5.6-terra',
        'messages': [
            {
                'role': 'user',
                'content': '这是此前上传的照片',
                'files': [{'type': 'image', 'url': '/api/v1/files/original/content'}],
            },
            {'role': 'assistant', 'content': '已收到'},
            {'role': 'user', 'content': '生成一幅完全无关的新插画'},
        ],
    }

    await middleware_utils.chat_image_generation_handler(
        request=object(),
        form_data=form_data,
        extra_params={
            '__metadata__': {
                'chat_id': 'local:test',
                'message_id': 'message-2',
                'current_user_image_refs': [],
            },
            '__current_user_image_refs__': [],
            '__event_emitter__': _no_op_event,
        },
        user=None,
    )

    assert captured[0]['form_data'].prompt == '生成一幅完全无关的新插画'


@pytest.mark.asyncio
async def test_image_edit_rejects_empty_or_inaccessible_sources_before_upstream(monkeypatch) -> None:
    async def image_config():
        return SimpleNamespace(IMAGE_EDIT_SIZE='', IMAGE_EDIT_MODEL='gpt-image-2')

    async def inaccessible_file(*_args, **_kwargs):
        raise HTTPException(status_code=404, detail='File not found')

    monkeypatch.setattr(images_router, 'get_image_config', image_config)
    monkeypatch.setattr(images_router, 'get_file_content_by_id', inaccessible_file)

    with pytest.raises(HTTPException, match='source image') as empty_error:
        await images_router.image_edits(
            request=object(),
            form_data=images_router.EditImageForm(prompt='美化', image=[]),
            user=None,
        )
    assert empty_error.value.status_code == 400

    with pytest.raises(HTTPException, match='File not found') as inaccessible_error:
        await images_router.image_edits(
            request=object(),
            form_data=images_router.EditImageForm(prompt='美化', image='another-users-file'),
            user=None,
        )
    assert inaccessible_error.value.status_code == 404


@pytest.mark.asyncio
async def test_sub2api_key_loss_does_not_fall_back_to_the_admin_key(monkeypatch) -> None:
    monkeypatch.setattr(openai_router, 'is_sub2api_key_login_connection', lambda _url: True)
    monkeypatch.setattr(openai_router, 'is_sub2api_key_login_user', lambda _user: True)

    async def no_session(*_args, **_kwargs):
        return None

    monkeypatch.setattr(
        openai_router.OAuthSessions,
        'get_session_by_provider_and_user_id',
        no_session,
    )

    with pytest.raises(HTTPException, match='expired') as error:
        await openai_router.get_effective_openai_api_key(
            'http://sub2api:8080/v1',
            'admin-key-must-not-be-used',
            SimpleNamespace(id='family-user'),
        )

    assert error.value.status_code == 401


@pytest.mark.asyncio
async def test_sub2api_image_requests_select_the_current_users_key(monkeypatch) -> None:
    monkeypatch.setattr(openai_router, 'is_sub2api_key_login_connection', lambda _url: True)
    monkeypatch.setattr(openai_router, 'is_sub2api_key_login_user', lambda _user: True)

    async def current_user_session(*_args, **_kwargs):
        return SimpleNamespace(token={'access_token': 'family-user-test-key'})

    monkeypatch.setattr(
        openai_router.OAuthSessions,
        'get_session_by_provider_and_user_id',
        current_user_session,
    )

    api_key = await openai_router.get_effective_openai_api_key(
        'http://sub2api:8080/v1',
        'admin-key-must-not-be-used',
        SimpleNamespace(id='family-user'),
    )

    assert api_key == 'family-user-test-key'
