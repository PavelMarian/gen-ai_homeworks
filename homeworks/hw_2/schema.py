from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field, field_validator


class Issue(BaseModel):
    category: Literal["performance", "design", "support", "price", "ads", "reliability"]
    text: str
    quote: str


class Review(BaseModel):
    user_name: str
    rating: int
    issues: list[Issue]

    @field_validator("rating")
    @classmethod
    def validate_rating(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f'Оценка должна быть от 1 до 5, получено: {v}')
        if v > 5:
            raise ValueError(f'Оценка должна быть от 1 до 5, получено: {v}')
        return v


class AspectSentiment(BaseModel):
    aspect: Literal["performance", "design", "support", "price", "ads", "reliability"]
    sentiment: Literal["positive", "neutral", "negative"]
    confidence: float
    quote: str


class ReviewSentiment(BaseModel):
    user_name: str
    aspects: list[AspectSentiment]


class DiscoveredAspect(BaseModel):
    name: str
    description: str
    mention_count: int
    example_quote: str


class DiscoveredAspects(BaseModel):
    aspects: list[DiscoveredAspect]


class ChunkSummary(BaseModel):
    key_points: list[str]
    main_sentiment: Literal["positive", "mixed", "negative"]


class DiscussionSummary(BaseModel):
    headline: str
    key_findings: list[str]
    action_items: list[str]


class GroupSummary(BaseModel):
    key_points: list[str]
    main_sentiment: Literal["positive", "mixed", "negative"]


class ActionVerdict(BaseModel):
    action: str
    support: Literal["supported", "weakly_supported", "not_supported"]
    evidence: list[str]
    comment: str


class JudgeReport(BaseModel):
    verdicts: list[ActionVerdict]
    overall_score: float
    summary: str


class MultiDocSummary(BaseModel):
    common_themes: list[str]
    unique_per_bank: dict[str, list[str]]
