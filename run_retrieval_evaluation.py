"""Offline evidence retrieval evaluation; no model or search API calls."""

import json
import argparse
from pathlib import Path

from research_agent.rag import retrieve_webpage_passages


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline retrieval evaluation")
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path(__file__).resolve().parent / "evals" / "evidence_cases.json",
        help="Path to evidence cases JSON",
    )
    case_path = parser.parse_args().cases
    cases = json.loads(case_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list) or not cases:
        raise ValueError("evidence_cases.json must contain a non-empty list")

    hit1_count = 0
    hit2_count = 0

    for case in cases:
        gold = case["gold_evidence"]
        page = case["page_text"]
        if not gold or gold not in page:
            raise ValueError(f"Invalid gold evidence: {case['id']}")

        passages, matched = retrieve_webpage_passages(
            page, case["query"], top_k=2
        )
        hit1 = bool(passages and gold in passages[0][1])
        hit2 = any(gold in text for _, text in passages)
        hit1_count += int(hit1)
        hit2_count += int(hit2)

        print(f"{case['id']}: matched={matched}, Hit@1={hit1}, Hit@2={hit2}")

    total = len(cases)
    print(f"Hit@1: {hit1_count}/{total} ({hit1_count / total:.1%})")
    print(f"Hit@2: {hit2_count}/{total} ({hit2_count / total:.1%})")


if __name__ == "__main__":
    main()