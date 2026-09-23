"""The JSON request and response contracts for the HTTP API."""

from pydantic import BaseModel, ConfigDict, Field


class AskRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")

    document_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=4, ge=1, le=20, strict=True)


class DocumentResponse(BaseModel):
    document_id: str
    source_filename: str
    chunk_count: int
    embedding_model: str


class SourceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    source_id: str
    document_id: str
    source_filename: str
    page_number: int
    chunk_id: int
    score: float
    text: str


class AskResponse(BaseModel):
    document_id: str
    question: str
    answer: str
    answerable: bool
    model: str
    cited_sources: list[SourceResponse]
    retrieved_sources: list[SourceResponse]
