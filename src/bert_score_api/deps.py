from bert_score import BERTScorer


def get_bert_scorer(
    language: str = "tr", rescale_with_baseline: bool = False
) -> BERTScorer:
    model_map = {
        "tr": "dbmdz/bert-base-turkish-cased",
    }

    return BERTScorer(
        model_type=model_map[language],
        lang=language,
        rescale_with_baseline=rescale_with_baseline,
    )
