from typing import AsyncGenerator, Any
from httpx import AsyncClient
import pytest_asyncio
from bert_score_api.app import create_app
from dotenv import load_dotenv
import logging

import pytest
from dotenv import find_dotenv, load_dotenv


logger = logging.getLogger(__name__)

@pytest.fixture(scope='session', autouse=True)
def load_env():
    logger.info("******* Loading .env.test file ******* ")
    env_file = find_dotenv('.env.test')
    load_dotenv(env_file)

@pytest_asyncio.fixture
def anyio_backend():
    return "asyncio"

@pytest_asyncio.fixture()
async def async_client() -> AsyncGenerator[AsyncClient, Any]:

    #FIXME: This doesn't take some env vars into account. Specifically, HF_HOME

    app = create_app()

    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client
