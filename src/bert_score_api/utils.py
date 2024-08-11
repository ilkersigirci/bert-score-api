import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)


def check_env_vars(env_vars: list[str] | None = None) -> None:
    """
    Checks if the required environment variables are set.

    Args:
        env_vars: List of environment variables to check. Defaults to None.

    Raises:
        ValueError: If any of the environment variables are not set.
    """
    if env_vars is None:
        env_vars = ["LIBRARY_BASE_PATH", "HF_HOME"]

    for env_var in env_vars:
        if os.getenv(env_var) is None:
            raise ValueError(f"Please set {env_var} env var.")


def is_model_exists(repo_id: str, use_local_dir: bool = False) -> bool:
    model_author, model_name = repo_id.split("/")

    if use_local_dir is True:
        check_env_vars(env_vars=["LIBRARY_BASE_PATH"])
        LIBRARY_BASE_PATH = os.environ["LIBRARY_BASE_PATH"]
        local_dir_name = repo_id.split("/")[1]
        local_dir_path = f"{LIBRARY_BASE_PATH}/deployment/models/{local_dir_name}"
    else:
        check_env_vars(env_vars=["HF_HOME"])
        HF_HOME = os.environ["HF_HOME"]
        local_dir_path = f"{HF_HOME}/hub/models--{model_author}--{model_name}"

    return Path(local_dir_path).exists()


def download_model_from_huggingface(
    repo_id: str = "dbmdz/bert-base-turkish-cased",
    revision: str = "main",
    ignore_patterns: list[str] | None = None,
    use_local_dir: bool = False,
) -> None:
    token = os.getenv("HF_TOKEN", None)

    if ignore_patterns is None:
        ignore_patterns = ["*.pt"]

    if use_local_dir is True:
        check_env_vars(env_vars=["LIBRARY_BASE_PATH"])
        LIBRARY_BASE_PATH = os.environ["LIBRARY_BASE_PATH"]
        local_dir_name = repo_id.split("/")[1]

        local_dir_path = f"{LIBRARY_BASE_PATH}/deployment/models/{local_dir_name}"
    else:
        local_dir_path = None

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        token=token,
        ignore_patterns=ignore_patterns,
        local_dir=local_dir_path,
    )


def download_model_by_language(language: str, use_local_dir: bool = False) -> None:
    if not is_valid_language(language):
        raise ValueError(f"Invalid language: {language}")

    repo_id = get_language_model_map().get(language)

    if is_model_exists(repo_id=repo_id, use_local_dir=False) is True:
        logger.info(f"Model already exists for {language}, skipping download.")
    else:
        logger.info(f"Downloading model for {language}.")

        download_model_from_huggingface(
            repo_id=repo_id,
            ignore_patterns=[
                "*.pt",
                "*.onnx",
                "*.onnx_data",
                "*ckpt*",
                "*msgpack*",
                "*h5",
            ],
            use_local_dir=use_local_dir,
        )
        logger.info(f"Model downloaded for {language}.")


def get_language_model_map() -> dict[str, str]:
    return {
        "tr": "dbmdz/bert-base-turkish-cased",
        "en": "FacebookAI/roberta-large",
    }


def is_valid_language(language: str) -> bool:
    return language in get_language_model_map()


if __name__ == "__main__":
    repo_id = "dbmdz/bert-base-turkish-cased"
    revision = "main"
    ignore_patterns = ["*.pt", "*.onnx", "*.onnx_data", "*ckpt*", "*msgpack*", "*h5"]

    load_dotenv()

    download_model_from_huggingface(
        repo_id=repo_id,
        revision=revision,
        ignore_patterns=ignore_patterns,
        use_local_dir=True,
    )
