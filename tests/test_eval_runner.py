"""Tests for the ResearchPilot evaluation runner."""

import unittest
from types import SimpleNamespace

from research_agent.eval_runner import (
    TokenUsageCallback,
    build_summary,
    estimate_cost_usd,
    reevaluate_records,
    run_cases,
)


class FakeGraph:
    """Fake graph that avoids model and search API calls."""

    def invoke(self, state, config=None):
        return {
            **state,
            "final_report": (
                "LangGraph支持状态管理。"
                "https://docs.langchain.com/langgraph"
            ),
            "evaluation": {
                "score": 9,
                "problems": [],
                "suggestions": [],
                "improvement_actions": [],
            },
            "revision_count": 0,
        }


class EvaluationRunnerTests(unittest.TestCase):
    """Verify evaluation execution with a fake graph."""

    def test_runner_collects_case_metrics(self) -> None:
        cases = [
            {
                "id": "state_management",
                "query": "LangGraph支持状态管理吗？",
                "expected_terms": ["状态"],
                "expected_domain": "docs.langchain.com",
            },
        ]

        records = run_cases(FakeGraph(), cases)

        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["metrics"]["success"])
        self.assertEqual(records[0]["input_tokens"], 0)
        self.assertEqual(records[0]["output_tokens"], 0)
        self.assertEqual(records[0]["total_tokens"], 0)
        self.assertEqual(records[0]["estimated_cost_usd"], 0)
        self.assertGreaterEqual(
            records[0]["elapsed_seconds"],
            0,
        )

    def test_summary_contains_latency(self) -> None:
        records = run_cases(
            FakeGraph(),
            [
                {
                    "id": "state_management",
                    "query": "LangGraph支持状态管理吗？",
                    "expected_terms": ["状态"],
                    "expected_domain": "docs.langchain.com",
                },
            ],
        )

        summary = build_summary(records)

        self.assertEqual(summary["total_cases"], 1)
        self.assertEqual(summary["success_rate"], 1.0)
        self.assertIn("average_latency_seconds", summary)
        self.assertEqual(summary["total_input_tokens"], 0)
        self.assertEqual(summary["total_output_tokens"], 0)
        self.assertEqual(summary["total_tokens"], 0)
        self.assertEqual(summary["average_total_tokens"], 0)
        self.assertEqual(summary["total_estimated_cost_usd"], 0)
        self.assertEqual(summary["average_estimated_cost_usd"], 0)

    def test_token_usage_callback_collects_usage(self) -> None:
        callback = TokenUsageCallback()
        response = SimpleNamespace(
            generations=[
                [
                    SimpleNamespace(
                        message=SimpleNamespace(
                            usage_metadata={
                                "input_tokens": 120,
                                "output_tokens": 30,
                            },
                        ),
                    ),
                ],
            ],
        )

        callback.on_llm_end(response)

        self.assertEqual(callback.input_tokens, 120)
        self.assertEqual(callback.output_tokens, 30)
        self.assertEqual(callback.total_tokens, 150)

    def test_estimate_cost_usd(self) -> None:
        cost = estimate_cost_usd(
            input_tokens=1_000_000,
            output_tokens=1_000_000,
            input_price_per_million=0.15,
            output_price_per_million=0.60,
        )

        self.assertEqual(cost, 0.75)

    def test_reevaluate_records_accepts_official_domain(self) -> None:
        records = [
            {
                "id": "checkpointing",
                "final_report": (
                    "LangGraph支持检查点。"
                    "https://reference.langchain.com/python/langgraph/checkpoints"
                ),
                "metrics": {
                    "success": False,
                    "citation_valid": False,
                    "keyword_matched": True,
                    "score": 9,
                    "revision_count": 0,
                },
            },
        ]
        cases = [
            {
                "id": "checkpointing",
                "expected_terms": ["检查点", "checkpoint"],
                "expected_domains": [
                    "docs.langchain.com",
                    "reference.langchain.com",
                    "langchain-ai.github.io",
                ],
            },
        ]

        updated_records = reevaluate_records(records, cases)

        self.assertTrue(updated_records[0]["metrics"]["success"])
        self.assertTrue(
            updated_records[0]["metrics"]["citation_valid"],
        )
        self.assertFalse(records[0]["metrics"]["citation_valid"])


if __name__ == "__main__":
    unittest.main()