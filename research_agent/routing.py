"""Conditional routing for the ResearchPilot quality loop."""

from typing import Literal

from research_agent.state import ResearchState


PASS_SCORE = 8
MAX_REVISIONS = 1

EvaluationRoute = Literal["revise", "finish"]


def route_after_evaluation(state: ResearchState) -> EvaluationRoute:
    """Choose whether Writer should revise the report."""

    evaluation = state.get("evaluation")
    if evaluation is None:
        message = "Evaluator result is missing from workflow state."
        raise ValueError(message)

    score = evaluation["score"]
    revision_count = state.get("revision_count", 0)

    if score >= PASS_SCORE:
        return "finish"

    if revision_count >= MAX_REVISIONS:
        return "finish"

    return "revise"