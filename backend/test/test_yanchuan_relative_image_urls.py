import pytest

from open_webui.utils import files


@pytest.mark.asyncio
async def test_relative_chat_file_content_url_is_resolved_as_owned_file(monkeypatch):
    captured = {}

    async def load_owned_file(file_id, user=None):
        captured['file_id'] = file_id
        captured['user'] = user
        return 'data:image/png;base64,owned-image'

    monkeypatch.setattr(files, 'get_image_base64_from_file_id', load_owned_file)

    user = object()
    result = await files.get_image_base64_from_url('/api/v1/files/source-file/content', user=user)

    assert result == 'data:image/png;base64,owned-image'
    assert captured == {'file_id': 'source-file', 'user': user}
