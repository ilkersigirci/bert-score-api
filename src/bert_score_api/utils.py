import os

from dotenv import load_dotenv
from huggingface_hub import snapshot_download


def check_env_vars(env_vars: list[str] | None = None) -> None:
    """
    Checks if the required environment variables are set.

    Args:
        env_vars (list[str], optional): List of environment variables to check. Defaults to None.

    Raises:
        ValueError: If any of the environment variables are not set.
    """
    if env_vars is None:
        env_vars = ["LIBRARY_BASE_PATH", "HF_HOME"]

    for env_var in env_vars:
        if os.getenv(env_var) is None:
            raise ValueError(f"Please set {env_var} env var.")


def download_model_hf(
    repo_id: str = "dbmdz/bert-base-turkish-cased",
    revision: str = "main",
    ignore_patterns: list[str] | None = None,
) -> None:
    check_env_vars(env_vars=["LIBRARY_BASE_PATH"])

    LIBRARY_BASE_PATH = os.environ["LIBRARY_BASE_PATH"]

    token = os.getenv("HF_TOKEN", None)

    local_dir_name = repo_id.split("/")[1]

    if ignore_patterns is None:
        ignore_patterns = ["*.pt"]

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=f"{LIBRARY_BASE_PATH}/deployment/models/{local_dir_name}",
        ignore_patterns=ignore_patterns,
        token=token,
    )


if __name__ == "__main__":
    repo_id = "dbmdz/bert-base-turkish-cased"
    revision = "main"
    ignore_patterns = ["*.pt", "*.onnx", "*.onnx_data", "*ckpt*", "*msgpack*", "*h5"]

    load_dotenv()

    download_model_hf(
        repo_id=repo_id, revision=revision, ignore_patterns=ignore_patterns
    )
