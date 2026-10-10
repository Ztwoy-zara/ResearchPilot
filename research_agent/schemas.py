"""Structured outputs used by ResearchPilot agents."""

from pydantic import BaseModel, Field


class EvaluatorOutput(BaseModel):
    """Machine-readable quality evaluation."""

    score: int = Field(
        ge=1,
        le=10,
        description="Overall report quality score from 1 to 10.",
    )

    problems: list[str] = Field(
        default_factory=list,
        description="Problems found in the report.",
    )

    suggestions: list[str] = Field(
        default_factory=list,
        description="Suggestions for improving the report.",
    )

    improvement_actions: list[str] = Field(
        default_factory=list,
        description="Concrete actions Writer should perform during revision.",
    )