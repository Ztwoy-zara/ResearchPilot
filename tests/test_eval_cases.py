"""Tests for the ResearchPilot evaluation dataset."""

import json
import unittest
from pathlib import Path


EVAL_FILE = (
    Path(__file__).resolve().parents[1]
    / "evals"
    / "cases.json"
)


class EvaluationDatasetTests(unittest.TestCase):
    """Validate evaluation cases without calling external APIs."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = json.loads(
            EVAL_FILE.read_text(encoding="utf-8"),
        )

    def test_dataset_contains_ten_cases(self) -> None:
        self.assertEqual(len(self.cases), 10)

    def test_case_ids_are_unique(self) -> None:
        case_ids = [case["id"] for case in self.cases]

        self.assertEqual(
            len(case_ids),
            len(set(case_ids)),
        )

    def test_cases_have_required_fields(self) -> None:
        for case in self.cases:
            with self.subTest(case=case["id"]):
                self.assertTrue(case["query"])
                self.assertTrue(case["expected_terms"])
                self.assertEqual(
                    case["expected_domains"],
                    [
                        "docs.langchain.com",
                        "reference.langchain.com",
                        "langchain-ai.github.io",
                    ],
                )


if __name__ == "__main__":
    unittest.main()