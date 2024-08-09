from pydantic import BaseModel


class TextPair(BaseModel):
    candidate: list[str]
    reference: list[str]


class BertScoreResult(BaseModel):
    precision: list[float]
    recall: list[float]
    f1: list[float]
