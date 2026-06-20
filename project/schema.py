from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

class ResearchData(BaseModel):
    product_id: str
    texts: List[str] = Field(default_factory=list)
    avg_rating: float = Field(ge=1, le=5)
    review_count: int
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @field_validator('avg_rating')
    def check_rating(cls, v):
        if v < 1 or v > 5:
            raise ValueError('avg_rating must be between 1 and 5')
        return v

    @field_validator('confidence')
    def check_confidence(cls, v):
        if v < 0 or v > 1:
            raise ValueError('confidence must be between 0 and 1')
        return v

class ProductStats(BaseModel):
    product_id: str
    avg_rating: float = Field(ge=1, le=5)
    review_count: int

    @field_validator('avg_rating')
    def check_rating(cls, v):
        if v < 1 or v > 5:
            raise ValueError('avg_rating must be between 1 and 5')
        return v

class ProductAspects(BaseModel):
    pros: List[str] = Field(default_factory=list)
    cons: List[str] = Field(default_factory=list)

class ProductReport(BaseModel):
    product_id: str
    avg_rating: float = Field(ge=1, le=5)
    total_reviews_analyzed: int
    pros: List[str]
    cons: List[str]
    summary: str
    confidence: float = Field(ge=0, le=1)

    @field_validator('avg_rating')
    def check_rating(cls, v):
        if v < 1 or v > 5:
            raise ValueError('avg_rating must be between 1 and 5')
        return v

    @field_validator('confidence')
    def check_confidence(cls, v):
        if v < 0 or v > 1:
            raise ValueError('confidence must be between 0 and 1')
        return v

class SubTask(BaseModel):
    id: int
    description: str
    agent: str  # "researcher" или "analyst"
    depends_on: List[int] = Field(default_factory=list)

class Plan(BaseModel):
    reasoning: str
    subtasks: List[SubTask]


class AssertionVerdict(BaseModel):
    assertion: str
    supported: bool
    evidence: List[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    comment: str

class JudgeReport(BaseModel):
    assertions: List[AssertionVerdict]
    overall_hallucinated: bool
    overall_score: int = Field(ge=1, le=5)
    summary: str