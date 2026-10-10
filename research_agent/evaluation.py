"""Offline evaluation metrics for ResearchPilot."""

from typing import TypedDict

from research_agent.routing import PASS_SCORE
from research_agent.state import ResearchState


class CaseMetrics(TypedDict):
    """Metrics produced for one evaluation case."""

    success: bool
    citation_valid: bool
    keyword_matched: bool
    score: int
    revision_count: int


class SummaryMetrics(TypedDict):
    """Aggregated metrics for multiple cases."""

    total_cases: int
    success_rate: float
    citation_validity_rate: float
    keyword_match_rate: float
    average_score: float
    average_revisions: float


def evaluate_result(
    state: ResearchState,
    expected_terms: list[str],
    expected_domains: list[str],
) -> CaseMetrics:
    """Evaluate one completed workflow state."""

    report = state.get("final_report", "").strip()
    evaluation = state.get("evaluation")
    score = evaluation["score"] if evaluation else 0
    normalized_report = report.casefold()

    keyword_matched = any(
        term.casefold() in normalized_report
        for term in expected_terms
    )
    citation_valid = any(
        domain.casefold() in normalized_report
        for domain in expected_domains
    )
    workflow_passed = bool(report) and score >= PASS_SCORE

    return {
        "success": (
            workflow_passed
            and keyword_matched
            and citation_valid
        ),
        "citation_valid": citation_valid,
        "keyword_matched": keyword_matched,
        "score": score,
        "revision_count": max(
            state.get("revision_count", 0),
            0,
        ),
    }


def summarize_metrics(
    results: list[CaseMetrics],
) -> SummaryMetrics:
    """Aggregate metrics from multiple evaluation cases."""

    if not results:
        message = "At least one evaluation result is required."
        raise ValueError(message)

    total = len(results)

    return {
        "total_cases": total,
        "success_rate": (
            sum(result["success"] for result in results) / total
        ),
        "citation_validity_rate": (
            sum(result["citation_valid"] for result in results)
            / total
        ),
        "keyword_match_rate": (
            sum(result["keyword_matched"] for result in results)
            / total
        ),
        "average_score": (
            sum(result["score"] for result in results) / total
        ),
        "average_revisions": (
            sum(result["revision_count"] for result in results)
            / total
        ),
    }