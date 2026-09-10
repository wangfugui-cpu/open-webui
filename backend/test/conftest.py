"""Shared isolated database fixture for YanChuan entry-reliability tests."""

from __future__ import annotations

import pytest_asyncio
from open_webui.internal import db as db_internal
from open_webui.internal.db import Base
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


@pytest_asyncio.fixture
async def isolated_runtime_db(tmp_path, monkeypatch):
    """Route runtime ORM work to a new SQLite database for each test."""

    database = tmp_path / 'execution-reliability.db'
    engine = create_async_engine(f'sqlite+aiosqlite:///{database}')
    sessions = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Runtime models resolve AsyncSessionLocal dynamically through
    # get_async_db_context(), so this preserves their real persistence logic.
    monkeypatch.setattr(db_internal, 'AsyncSessionLocal', sessions)
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)

    try:
        yield engine
    finally:
        await engine.dispose()
