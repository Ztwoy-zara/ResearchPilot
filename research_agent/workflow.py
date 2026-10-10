"""Explicit LangGraph workflow for ResearchPilot."""

from collections.abc import Callable
from typing import Any

from langgraph.graph import END, START, StateGraph

from research_agent.nodes import (
    finalize_report_node,
    initialize_state_node,
    prepare_revision_node,
)
from research_agent.routing import route_after_evaluation
from research_agent.state import ResearchState


WorkflowNode = Callable[[ResearchState], dict[str, Any]]


def build_research_graph(
    planner_node: WorkflowNode,
    researcher_node: WorkflowNode,
    writer_node: WorkflowNode,
    evaluator_node: WorkflowNode,
) -> Any:
    """Build the explicit ResearchPilot workflow."""

    builder = StateGraph(ResearchState)

    builder.add_node("initialize", initialize_state_node)
    builder.add_node("planner", planner_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("writer", writer_node)
    builder.add_node("evaluator", evaluator_node)
    builder.add_node("prepare_revision", prepare_revision_node)
    builder.add_node("finalize", finalize_report_node)

    builder.add_edge(START, "initialize")
    builder.add_edge("initialize", "planner")
    builder.add_edge("planner", "researcher")
    builder.add_edge("researcher", "writer")
    builder.add_edge("writer", "evaluator")

    builder.add_conditional_edges(
        "evaluator",
        route_after_evaluation,
        {
            "revise": "prepare_revision",
            "finish": "finalize",
        },
    )

    builder.add_edge("prepare_revision", "writer")
    builder.add_edge("finalize", END)

    return builder.compile()