"""Tests for the ResearchPilot evaluation route."""

import unittest

from research_agent.nodes import (
    build_evaluator_node,
    evaluation_to_state_update,
    prepare_revision_node,
    initialize_state_node,
)

from research_agent.schemas import EvaluatorOutput

from research_agent.routing import route_after_evaluation
from research_agent.state import ResearchState


def create_state(score: int, revision_count: int) -> ResearchState:
    """Create the minimum valid state needed by the router."""

    return {
        "evaluation": {
            "score": score,
            "problems": [],
            "suggestions": [],
            "improvement_actions": [],
        },
        "revision_count": revision_count,
    }

class FakeEvaluatorModel:
    """Model double that avoids external API calls."""

    def __init__(self) -> None:
        self.schema = None
        self.messages = []

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        self.method = kwargs.get("method")
        return self

    def invoke(self, messages):
        self.messages = messages

        return EvaluatorOutput(
            score=7,
            problems=["缺少来源"],
            suggestions=["补充官方文档"],
            improvement_actions=["增加一条官方引用"],
        )

class RoutingTests(unittest.TestCase):
    """Verify the bounded Writer-Evaluator loop."""

    def test_passing_score_finishes(self) -> None:
        state = create_state(score=8, revision_count=0)

        result = route_after_evaluation(state)

        self.assertEqual(result, "finish")

    def test_low_score_requests_revision(self) -> None:
        state = create_state(score=7, revision_count=0)

        result = route_after_evaluation(state)

        self.assertEqual(result, "revise")

    def test_low_score_stops_after_revision_limit(self) -> None:
        state = create_state(score=7, revision_count=1)

        result = route_after_evaluation(state)

        self.assertEqual(result, "finish")

    def test_missing_evaluation_raises_error(self) -> None:
        state: ResearchState = {"revision_count": 0}

        with self.assertRaises(ValueError):
            route_after_evaluation(state)

    def test_prepare_revision_increments_counter(self) -> None:
        state: ResearchState = {"revision_count": 0}

        update = prepare_revision_node(state)

        self.assertEqual(update, {"revision_count": 1})
        self.assertEqual(state["revision_count"], 0)

    def test_evaluation_output_is_saved_to_state(self) -> None:
        output = EvaluatorOutput(
            score=7,
            problems=["内容不完整"],
            suggestions=["补充官方来源"],
            improvement_actions=["增加一条官方文档引用"],
        )

        update = evaluation_to_state_update(output)

        self.assertEqual(update["evaluation"]["score"], 7)
        self.assertEqual(
            update["evaluation"]["problems"],
            ["内容不完整"],
        )
        self.assertEqual(
            update["evaluation"]["improvement_actions"],
            ["增加一条官方文档引用"],
        )

    def test_evaluator_node_updates_state(self) -> None:
        model = FakeEvaluatorModel()
        evaluator_node = build_evaluator_node(model)

        state: ResearchState = {
            "query": "LangGraph 如何管理状态？",
            "research": "官方资料说明 LangGraph 支持 checkpoint。",
            "draft": "LangGraph 使用 checkpoint 保存状态。",
        }

        update = evaluator_node(state)

        self.assertIs(model.schema, EvaluatorOutput)
        self.assertEqual(model.method, "function_calling")
        self.assertEqual(update["evaluation"]["score"], 7)
        self.assertEqual(
            update["evaluation"]["improvement_actions"],
            ["增加一条官方引用"],
        )
        self.assertIn("LangGraph 如何管理状态", str(model.messages))

    def test_evaluator_node_rejects_missing_draft(self) -> None:
        model = FakeEvaluatorModel()
        evaluator_node = build_evaluator_node(model)
        state: ResearchState = {
            "query": "LangGraph 如何管理状态？",
        }

        with self.assertRaises(ValueError):
            evaluator_node(state)

    def test_initialize_normalizes_negative_revision_count(self) -> None:
        state: ResearchState = {
            "revision_count": -1,
        }

        update = initialize_state_node(state)

        self.assertEqual(update, {"revision_count": 0})

if __name__ == "__main__":
    unittest.main()