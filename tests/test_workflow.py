"""Tests for the explicit ResearchPilot StateGraph."""

import unittest

from research_agent.state import ResearchState
from research_agent.workflow import build_research_graph


def planner_node(state: ResearchState) -> dict[str, str]:
    """Return a deterministic research plan."""

    return {
        "plan": f"Research: {state['query']}",
    }


def researcher_node(state: ResearchState) -> dict[str, str]:
    """Return deterministic research evidence."""

    return {
        "research": f"Evidence for: {state['plan']}",
    }


def writer_node(state: ResearchState) -> dict[str, str]:
    """Create an initial or revised draft."""

    if state.get("revision_count", 0) == 0:
        return {
            "draft": "Initial report",
        }

    return {
        "draft": "Revised report",
    }


def passing_evaluator_node(
    state: ResearchState,
) -> dict[str, object]:
    """Pass the first Writer draft."""

    return {
        "evaluation": {
            "score": 8,
            "problems": [],
            "suggestions": [],
            "improvement_actions": [],
        },
    }


def revision_evaluator_node(
    state: ResearchState,
) -> dict[str, object]:
    """Fail once and pass after Writer revision."""

    score = 6 if state.get("revision_count", 0) == 0 else 9

    return {
        "evaluation": {
            "score": score,
            "problems": [] if score >= 8 else ["Draft is incomplete"],
            "suggestions": [] if score >= 8 else ["Revise the draft"],
            "improvement_actions": (
                [] if score >= 8 else ["Add missing information"]
            ),
        },
    }


class WorkflowTests(unittest.TestCase):
    """Verify the explicit StateGraph and quality loop."""

    def test_passing_report_finishes_without_revision(self) -> None:
        graph = build_research_graph(
            planner_node=planner_node,
            researcher_node=researcher_node,
            writer_node=writer_node,
            evaluator_node=passing_evaluator_node,
        )

        result = graph.invoke(
            {
                "query": "What is LangGraph?",
                "revision_count": 0,
            },
        )

        self.assertEqual(result["revision_count"], 0)
        self.assertEqual(result["final_report"], "Initial report")
        self.assertEqual(result["evaluation"]["score"], 8)

    def test_low_score_triggers_one_writer_revision(self) -> None:
        graph = build_research_graph(
            planner_node=planner_node,
            researcher_node=researcher_node,
            writer_node=writer_node,
            evaluator_node=revision_evaluator_node,
        )

        result = graph.invoke(
            {
                "query": "What is LangGraph?",
                "revision_count": 0,
            },
        )

        self.assertEqual(result["revision_count"], 1)
        self.assertEqual(result["draft"], "Revised report")
        self.assertEqual(result["final_report"], "Revised report")
        self.assertEqual(result["evaluation"]["score"], 9)


if __name__ == "__main__":
    unittest.main()