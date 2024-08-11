import os
import asyncio

def test_env():
    TEST_ENV_KEY = os.getenv("TEST_ENV_KEY")
    assert TEST_ENV_KEY == "TEST_ENV_VALUE"

async def test_provided_loop_is_running_loop(event_loop):
    assert event_loop is asyncio.get_running_loop()
