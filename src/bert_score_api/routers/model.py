import logging

from fastapi import APIRouter, HTTPException, status

from bert_score_api.deps import get_language_model_map
from bert_score_api.utils import download_model_by_language

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/model", tags=["model"])


@router.get("/languages")
async def supported_languages() -> list[str]:
    return list(get_language_model_map().keys())


@router.post("/download")
async def download_model(language: str) -> None:
    try:
        download_model_by_language(language=language, use_local_dir=False)
    except Exception as e:
        error_message = f"Failed to download model for language {language}: {e!s}"
        logger.error(error_message)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_message,
        ) from e

    return {"message": "Model downloaded successfully"}
