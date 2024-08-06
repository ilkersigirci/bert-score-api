from pydantic import BaseModel


class TextPair(BaseModel):
    candidate: list[str]
    reference: list[str]
