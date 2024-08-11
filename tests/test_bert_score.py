from httpx import AsyncClient
import pytest
import logging

logger = logging.getLogger(__name__)

@pytest.mark.anyio
async def test_bert_score(async_client: AsyncClient):
    url = "/score_calculation/all"

    data = {
        "candidate": ["The quick brown fox jumps over the lazy dog"],
        "reference": ["A fast brown fox leaps over a lazy dog"]
    }

    response = await async_client.post(url, json=data)

    assert response.status_code == 200

    precision = response.json()["precision"]

    logger.info(f"Precision score: {precision}")
    assert response.json()["precision"] > [0.6]
