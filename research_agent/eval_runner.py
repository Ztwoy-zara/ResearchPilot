"""Evaluation runner utilities for ResearchPilot."""

import json
from pathlib import Path
from time import perf_counter
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

from research_agent.evaluation import (
    CaseMetrics,
    evaluate_result,
    summarize_metrics,
)


DEFAULT_INPUT_PRICE_PER_MILLION = 0.15
DEFAULT_OUTPUT_PRICE_PER_MILLION = 0.60


def load_cases(path: Path) -> list[dict[str, Any]]:
    """Load evaluation cases from JSON."""

    return json.loads(path.read_text(encoding="utf-8"))


class TokenUsageCallback(BaseCallbackHandler):
    """Collect token usage reported by chat-model responses."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.output_tokens = 0

    @property
    def total_tokens(self) -> int:
        """Return total input and output tokens."""

        return self.input_tokens + self.output_tokens

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Collect usage metadata after each model call."""

        for generation_list in response.generations:
            for generation in generation_list:
                message = getattr(generation, "message", None)
                usage = getattr(message, "usage_metadata", None)

                if not usage:
                    continue

                self.input_tokens += usage.get("input_tokens", 0)
                self.output_tokens += usage.get("output_tokens", 0)


def estimate_cost_usd(
    input_tokens: int,
    output_tokens: int,
    input_price_per_million: float,
    output_price_per_million: float,
) -> float:
    """Estimate model cost from token counts and configured prices."""

    values = (
        input_tokens,
        output_tokens,
        input_price_per_million,
        output_price_per_million,
    )

    if any(value < 0 for value in values):
        message = "Token counts and prices cannot be negative."
        raise ValueError(message)

    cost = (
        input_tokens * input_price_per_million
        + output_tokens * output_price_per_million
    ) / 1_000_000

    return round(cost, 8)


def run_cases(
    graph: Any,
    cases: list[dict[str, Any]],
    *,
    input_price_per_million: float = DEFAULT_INPUT_PRICE_PER_MILLION,
    output_price_per_million: float = DEFAULT_OUTPUT_PRICE_PER_MILLION,
) -> list[dict[str, Any]]:
    """Run selected cases and collect metrics."""

    records = []

    for case in cases:
        token_usage = TokenUsageCallback()
        started_at = perf_counter()

        state = graph.invoke(
            {
                "query": case["query"],
            },
            config={
                "callbacks": [token_usage],
            },
        )
        elapsed_seconds = perf_counter() - started_at
        expected_domains = case.get("expected_domains")

        if expected_domains is None:
            expected_domains = [case["expected_domain"]]

        metrics = evaluate_result(
            state=state,
            expected_terms=case["expected_terms"],
            expected_domains=expected_domains,
        )

        records.append(
            {
                "id": case["id"],
                "query": case["query"],
                "elapsed_seconds": round(elapsed_seconds, 3),
                "input_tokens": token_usage.input_tokens,
                "output_tokens": token_usage.output_tokens,
                "total_tokens": token_usage.total_tokens,
                "estimated_cost_usd": estimate_cost_usd(
                    input_tokens=token_usage.input_tokens,
                    output_tokens=token_usage.output_tokens,
                    input_price_per_million=input_price_per_million,
                    output_price_per_million=output_price_per_million,
                ),
                "metrics": metrics,
                "final_report": state.get("final_report", ""),
            },
        )

    return records


def reevaluate_records(
    records: list[dict[str, Any]],
    cases: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Recalculate saved metrics without calling external APIs."""

    cases_by_id = {
        case["id"]: case
        for case in cases
    }
    updated_records = []

    for record in records:
        case = cases_by_id.get(record["id"])

        if case is None:
            message = f"Missing evaluation case: {record['id']}"
            raise ValueError(message)

        previous_metrics = record["metrics"]
        expected_domains = case.get("expected_domains")

        if expected_domains is None:
            expected_domains = [case["expected_domain"]]

        state = {
            "final_report": record.get("final_report", ""),
            "evaluation": {
                "score": previous_metrics["score"],
                "problems": [],
                "suggestions": [],
                "improvement_actions": [],
            },
            "revision_count": previous_metrics["revision_count"],
        }

        updated_record = dict(record)
        updated_record["metrics"] = evaluate_result(
            state=state,
            expected_terms=case["expected_terms"],
            expected_domains=expected_domains,
        )
        updated_records.append(updated_record)

    return updated_records


def build_summary(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build summary metrics including average latency."""

    if not records:
        message = "At least one evaluation record is required."
        raise ValueError(message)

    case_metrics: list[CaseMetrics] = [
        record["metrics"]
        for record in records
    ]

    summary = dict(summarize_metrics(case_metrics))
    summary["average_latency_seconds"] = round(
        sum(record["elapsed_seconds"] for record in records)
        / len(records),
        3,
    )
    summary["total_input_tokens"] = sum(
        record["input_tokens"]
        for record in records
    )
    summary["total_output_tokens"] = sum(
        record["output_tokens"]
        for record in records
    )
    summary["total_tokens"] = sum(
        record["total_tokens"]
        for record in records
    )
    summary["average_input_tokens"] = round(
        summary["total_input_tokens"] / len(records),
        2,
    )
    summary["average_output_tokens"] = round(
        summary["total_output_tokens"] / len(records),
        2,
    )
    summary["average_total_tokens"] = round(
        summary["total_tokens"] / len(records),
        2,
    )
    summary["total_estimated_cost_usd"] = round(
        sum(
            record["estimated_cost_usd"]
            for record in records
        ),
        8,
    )
    summary["average_estimated_cost_usd"] = round(
        summary["total_estimated_cost_usd"] / len(records),
        8,
    )

    return summary


def save_results(
    path: Path,
    records: list[dict[str, Any]],
) -> None:
    """Save evaluation records and summary as JSON."""

    output = {
        "summary": build_summary(records),
        "records": records,
    }

    path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )