"""Run a small ResearchPilot evaluation safely."""

import argparse
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CASES_FILE = BASE_DIR / "evals" / "cases.json"
RESULTS_FILE = BASE_DIR / "evals" / "results.json"


def main() -> None:
    """Preview or execute evaluation cases."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Maximum number of cases to run.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually call the model and search API.",
    )
    args = parser.parse_args()

    if args.limit < 1:
        raise ValueError("--limit must be at least 1.")

    from research_agent.eval_runner import (
        build_summary,
        load_cases,
        run_cases,
        save_results,
    )

    cases = load_cases(CASES_FILE)[: args.limit]

    if not args.execute:
        print("Dry run only. No API calls were made.")
        for case in cases:
            print(f"- {case['id']}: {case['query']}")
        print("Add --execute to run these cases.")
        return

    from dotenv import load_dotenv

    load_dotenv()

    from explicit_agent import explicit_graph

    records = run_cases(
        graph=explicit_graph,
        cases=cases,
    )
    save_results(RESULTS_FILE, records)
    summary = build_summary(records)

    print(f"Saved {len(records)} result(s) to {RESULTS_FILE}")
    print(f"Success rate: {summary['success_rate']:.1%}")
    print(
        "Citation validity rate: "
        f"{summary['citation_validity_rate']:.1%}",
    )
    print(
        "Average latency: "
        f"{summary['average_latency_seconds']:.3f} seconds",
    )
    print(
        "Average tokens: "
        f"{summary['average_total_tokens']:.2f}",
    )
    print(
        "Average estimated cost: "
        f"${summary['average_estimated_cost_usd']:.8f}",
    )


if __name__ == "__main__":
    main()