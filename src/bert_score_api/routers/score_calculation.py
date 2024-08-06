from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException

from bert_score_api.deps import get_bert_scorer
from bert_score_api.schemes import TextPair

router = APIRouter(prefix="/score_calculation", tags=["calculate"])


@router.post("/all")
async def calculate_bert_score(
    text_pair: TextPair, bert_scorer: Annotated[any, Depends(get_bert_scorer)]
):
    if len(text_pair.candidate) != len(text_pair.reference):
        raise HTTPException(
            status_code=400,
            detail="Candidate and reference lists must have the same length.",
        )

    try:
        P, R, F1 = bert_scorer.score(
            text_pair.candidate,
            text_pair.reference,
            verbose=True,
        )
        return {
            "precision": P.mean().item(),
            "recall": R.mean().item(),
            "f1": F1.mean().item(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
