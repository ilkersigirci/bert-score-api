import logging
import os
from contextlib import asynccontextmanager
from importlib import metadata

import anyio
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse

from bert_score_api.routers import model, score_calculation
from bert_score_api.utils import (
    check_env_vars,
    download_model_by_language,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    limiter = anyio.to_thread.current_default_thread_limiter()
    limiter.total_tokens = 100  # NOTE: Default is 40

    check_env_vars(env_vars=["LANGUAGE"])

    LANGUAGE = os.environ["LANGUAGE"]

    try:
        download_model_by_language(language=LANGUAGE, use_local_dir=False)
    except Exception as e:
        error_message = f"Failed to download model for language {LANGUAGE}: {e!s}"
        logger.error(error_message)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_message,
        ) from e
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Bert Score Api",
        version=metadata.version("bert_score_api"),
        docs_url="/api/docs",
        redoc_url="/api/redoc",
        openapi_url="/api/openapi.json",
        default_response_class=ORJSONResponse,
        lifespan=lifespan,
    )

    app.include_router(model.router)
    app.include_router(score_calculation.router)

    # For local development
    origins = [
        "http://localhost:3000",
    ]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> str:
        return "ok"

    return app
