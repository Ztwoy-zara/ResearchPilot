"""Shared state for the ResearchPilot workflow."""

from typing import TypedDict


class EvaluationResult(TypedDict):
    """Structured result produced by Evaluator."""

    score: int
    problems: list[str]
    suggestions: list[str]
    improvement_actions: list[str]


class ResearchState(TypedDict, total=False):
    """Shared data passed between all workflow nodes."""

    query: str
    plan: str
    research: str
    draft: str
    evaluation: EvaluationResult
    revision_count: int
    final_report: str