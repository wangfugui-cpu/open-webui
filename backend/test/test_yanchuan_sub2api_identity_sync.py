"""Identity and per-browser credential regressions for YanChuan key login."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from open_webui.models.users import UserModel
from open_webui.routers import auths as auths_router
from open_webui.routers import openai as openai_router


def _user(user_id: str, *, name: str = 'Old name', oauth: dict | None = None) -> UserModel:
    return UserModel(
        id=user_id,
        email=f'{user_id}@users.invalid',
        name=name,
        role='user',
        oauth=oauth or {},
        last_active_at=0,
        updated_at=0,
        created_at=0,
    )


@pytest.mark.asyncio
async def test_sub2api_name_sync_keeps_stable_identity_and_observed_key_history(monkeypatch):
    user = _user(
        'local-user',
        oauth={'sub2api': {'subject': '42', 'observed_keys': [{'id': '1', 'name': 'Old key'}]}},
    )
    updates = []

    async def update_user(user_id, values, db=None):
        updates.append((user_id, values))
        return _user(user_id, name=values.get('name', user.name), oauth=values.get('oauth', user.oauth))

    monkeypatch.setattr(auths_router.Users, 'update_user_by_id', update_user)

    synced = await auths_router.sync_sub2api_identity(
        user,
        '42',
        'yan-chuan-production',
        '改名后的言川用户',
        {'id': '2', 'name': '第二把密钥'},
        db=None,
    )

    assert synced.id == user.id
    assert updates == [
        (
            'local-user',
            {
                'name': '改名后的言川用户',
                'oauth': {
                    'sub2api': {
                        'instance_id': 'yan-chuan-production',
                        'stable_user_id': '42',
                        'subject': '42',
                        'observed_keys': [
                            {'id': '1', 'name': 'Old key'},
                            {'id': '2', 'name': '第二把密钥'},
                        ],
                    }
                },
            },
        )
    ]


def test_sub2api_identity_namespace_never_uses_display_name_or_email():
    assert auths_router.get_sub2api_user_email('instance-a', '42') == auths_router.get_sub2api_user_email(
        'instance-a', '42'
    )
    assert auths_router.get_sub2api_user_email('instance-a', '42') != auths_router.get_sub2api_user_email(
        'instance-b', '42'
    )


@pytest.mark.asyncio
async def test_two_browser_sessions_for_same_identity_keep_their_own_key(monkeypatch):
    user = _user(
        'local-user',
        oauth={'sub2api': {'instance_id': 'yan-chuan', 'stable_user_id': '42', 'subject': '42'}},
    )
    sessions = {
        'browser-a': SimpleNamespace(
            provider=openai_router.SUB2API_KEY_SESSION_PROVIDER,
            token={'access_token': 'key-a', 'subject': '42', 'instance_id': 'yan-chuan'},
        ),
        'browser-b': SimpleNamespace(
            provider=openai_router.SUB2API_KEY_SESSION_PROVIDER,
            token={'access_token': 'key-b', 'subject': '42', 'instance_id': 'yan-chuan'},
        ),
    }

    async def get_session(session_id, user_id):
        assert user_id == 'local-user'
        return sessions.get(session_id)

    monkeypatch.setattr(openai_router, 'ENABLE_SUB2API_KEY_LOGIN', True)
    monkeypatch.setattr(openai_router, 'SUB2API_KEY_LOGIN_BASE_URL', 'https://api.yan-chuan.com')
    monkeypatch.setattr(openai_router.OAuthSessions, 'get_session_by_id_and_user_id', get_session)

    request_a = SimpleNamespace(cookies={'sub2api_key_session_id': 'browser-a'}, state=SimpleNamespace())
    request_b = SimpleNamespace(cookies={'sub2api_key_session_id': 'browser-b'}, state=SimpleNamespace())
    assert await openai_router.get_effective_openai_api_key(
        'https://api.yan-chuan.com/v1/images/edits', 'admin-key', user, request=request_a
    ) == 'key-a'
    assert await openai_router.get_effective_openai_api_key(
        'https://api.yan-chuan.com/v1/chat/completions', 'admin-key', user, request=request_b
    ) == 'key-b'

    with pytest.raises(HTTPException) as missing_cookie:
        await openai_router.get_effective_openai_api_key(
            'https://api.yan-chuan.com/v1/chat/completions', 'admin-key', user, request=SimpleNamespace(cookies={}, state=SimpleNamespace())
        )
    assert missing_cookie.value.status_code == 401
