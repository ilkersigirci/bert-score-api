import os

from bert_score import BERTScorer

from bert_score_api.utils import check_env_vars, get_language_model_map


def get_bert_scorer() -> BERTScorer:
    check_env_vars(["LANGUAGE", "RESCALE_WITH_BASELINE"])

    LANGUAGE = os.environ["LANGUAGE"].lower()
    RESCALE_WITH_BASELINE = bool(os.environ["RESCALE_WITH_BASELINE"])

    model_type = get_language_model_map().get(LANGUAGE, None)

    if model_type is None:
        raise ValueError(f"Language {LANGUAGE} is not supported.")

    return BERTScorer(
        model_type=model_type,
        lang=LANGUAGE,
        rescale_with_baseline=RESCALE_WITH_BASELINE,
    )
