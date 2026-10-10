"""Tests for offline ResearchPilot metrics."""

import unittest

from research_agent.evaluation import (
    evaluate_result,
    summarize_metrics,
)
from research_agent.state import ResearchState


class EvaluationMetricsTests(unittest.TestCase):
    """Verify metrics without external API calls."""

    def test_successful_result_passes(self) -> None:
        state: ResearchState = {
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

        metrics = evaluate_result(
            state=state,
            expected_terms=["状态", "state"],
            expected_domains=["docs.langchain.com"],
        )

        self.assertTrue(metrics["success"])
        self.assertTrue(metrics["citation_valid"])
        self.assertTrue(metrics["keyword_matched"])

    def test_unofficial_source_fails_success(self) -> None:
        state: ResearchState = {
            "final_report": (
                "LangGraph支持状态管理。"
                "https://example.com/article"
            ),
            "evaluation": {
                "score": 9,
                "problems": [],
                "suggestions": [],
                "improvement_actions": [],
            },
        }

        metrics = evaluate_result(
            state=state,
            expected_terms=["状态"],
            expected_domains=["docs.langchain.com"],
        )

        self.assertFalse(metrics["success"])
        self.assertFalse(metrics["citation_valid"])

    def test_summary_aggregates_results(self) -> None:
        results = [
            {
                "success": True,
                "citation_valid": True,
                "keyword_matched": True,
                "score": 9,
                "revision_count": 0,
            },
            {
                "success": False,
                "citation_valid": False,
                "keyword_matched": True,
                "score": 7,
                "revision_count": 1,
            },
        ]

        summary = summarize_metrics(results)

        self.assertEqual(summary["total_cases"], 2)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["citation_validity_rate"], 0.5)
        self.assertEqual(summary["average_score"], 8.0)
        self.assertEqual(summary["average_revisions"], 0.5)

    def test_negative_revision_count_is_normalized(self) -> None:
        state: ResearchState = {
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
            "revision_count": -1,
        }

        metrics = evaluate_result(
            state=state,
            expected_terms=["状态"],
            expected_domains=["docs.langchain.com"],
        )

        self.assertEqual(metrics["revision_count"], 0)

if __name__ == "__main__":
    unittest.main()