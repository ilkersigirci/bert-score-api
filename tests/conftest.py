from typing import AsyncGenerator, Any
from httpx import AsyncClient
import pytest_asyncio
from bert_score_api.app import create_app

@pytest_asyncio.fixture
def anyio_backend():
    return "asyncio"

@pytest_asyncio.fixture
async def async_client() -> AsyncGenerator[AsyncClient, Any]:

    app = create_app()
    # app.dependency_overrides[get_async_session] = override_get_db

    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client
