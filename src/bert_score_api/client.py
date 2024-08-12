import os

import httpx

from bert_score_api.utils import check_env_vars


class BERTScoreClient:
    def __init__(self):
        check_env_vars(["BERT_SCORE_API_HOST", "BERT_SCORE_API_PORT"])

        BERT_SCORE_API_HOST = os.environ["BERT_SCORE_API_HOST"]
        BERT_SCORE_API_PORT = os.environ["BERT_SCORE_API_PORT"]

        self.score_endpoint = (
            f"{BERT_SCORE_API_HOST}:{BERT_SCORE_API_PORT}/score_calculation/all"
        )

    def score(self, candidates: list[str], references: list[str]) -> tuple[list[float]]:
        """
        Sends an score request to the bert score api service.

        Args:
            candidates: The list of candidate strings.
            references: The list of reference strings.

        Returns:
            The bert score for each candidate.
        """
        data = {
            "candidate": candidates,
            "reference": references,
        }

        response = httpx.post(self.score_endpoint, json=data).json()

        return (response["precision"], response["recall"], response["f1"])
