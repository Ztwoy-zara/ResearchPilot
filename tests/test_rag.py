"""Offline tests for webpage chunking and sparse retrieval."""

import unittest
import json
from pathlib import Path

from research_agent.rag import chunk_webpage, retrieve_webpage_passages


#验证排序、长正文截取、中文词项、无匹配及过滤无关片段
class WebRagTests(unittest.TestCase):
    def test_retrieval_prefers_relevant_passage_over_navigation(self) -> None:
        page = (
            "Home Products Pricing About us\n\n"
            "LangGraph checkpoints persist graph state between steps and allow recovery.\n\n"
            "Contact the sales team for more information."
        )

        passages, matched = retrieve_webpage_passages(page, "checkpoint persist state", top_k=1)

        self.assertTrue(matched)
        self.assertEqual(passages[0][0], 2)
        self.assertIn("persist graph state", passages[0][1])

    def test_long_page_and_evidence_budget(self) -> None:
        page = "StateGraph checkpoint " + "A" * 5000

        chunks = chunk_webpage(page)
        passages, _ = retrieve_webpage_passages(page, "StateGraph checkpoint")

        self.assertTrue(all(len(chunk) <= 500 for chunk in chunks))
        self.assertLessEqual(sum(len(text) for _, text in passages), 1200)

    def test_no_lexical_match_is_reported(self) -> None:
        passages, matched = retrieve_webpage_passages("Unrelated page content.", "checkpoint")

        self.assertFalse(matched)
        self.assertEqual(len(passages), 1)

    def test_unrelated_second_chunk_is_not_sent_to_writer(self) -> None:
        page = "Checkpoint saves state.\n\nContact sales for pricing."

        passages, matched = retrieve_webpage_passages(page, "checkpoint", top_k=2)

        self.assertTrue(matched)
        self.assertEqual(len(passages), 1)
        self.assertNotIn("pricing", passages[0][1])

    def test_chinese_character_pairs_match(self) -> None:
        page = "价格说明和联系方式。\n\n图状态支持检查点与恢复机制。"

        passages, matched = retrieve_webpage_passages(page, "检查点恢复", top_k=1)

        self.assertTrue(matched)
        self.assertEqual(passages[0][0], 2)

    def test_chinese_query_finds_english_evidence_when_not_first(self) -> None:
        case_path = Path(__file__).resolve().parents[1] / "evals" / "evidence_cases.json"
        case = json.loads(case_path.read_text(encoding="utf-8"))[0]
        sections = case["page_text"].split("\n\n")
        self.assertEqual(len(sections), 3)

        # 只在测试中调整顺序：Nodes、State、Edges
        page = "\n\n".join([sections[1], sections[0], sections[2]])
        passages, matched = retrieve_webpage_passages(page, case["query"], top_k=1)

        self.assertTrue(matched)
        self.assertTrue(
            any(case["gold_evidence"] in text for _, text in passages)
        )

    def test_empty_page_has_no_evidence(self) -> None:
        self.assertEqual(retrieve_webpage_passages(" \n ", "state"), ([], False))

    def test_chinese_session_query_prefers_checkpointer(self) -> None:
        page = (
            "Stores keep shared state across users.\n\n"
            "Checkpointers save a thread's graph state as checkpoints."
        )

        passages, matched = retrieve_webpage_passages(
            page, "如何保存会话状态？", top_k=1
        )

        self.assertTrue(matched)
        self.assertEqual(passages[0][0], 2)

if __name__ == "__main__":
    unittest.main()
