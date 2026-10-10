"""Unit tests for ResearchPilot workflow nodes."""

import unittest

from research_agent.nodes import (
    build_planner_node,
    build_researcher_node,
    build_writer_node,
)
from research_agent.state import ResearchState


class FakeResponse:
    """Minimal chat-model response."""

    def __init__(self, content: str) -> None:
        self.content = content


class FakeTextModel:
    """Fake model that does not make API requests."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.messages = []

    def invoke(self, messages):
        self.messages = messages
        return FakeResponse(self.content)

class FakeSearchTool:
    """Fake search tool that does not call Tavily."""

    def __init__(self, result: str) -> None:
        self.result = result
        self.tool_input = None

    def invoke(self, tool_input):
        self.tool_input = tool_input
        return self.result

class PlannerNodeTests(unittest.TestCase):
    """Verify the Planner node without using an external API."""

    def test_planner_saves_plan_to_state(self) -> None:
        model = FakeTextModel(
            "1. Find official documentation.\n"
            "2. Identify the main feature.",
        )
        planner_node = build_planner_node(model)

        state: ResearchState = {
            "query": "What is LangGraph?",
        }

        update = planner_node(state)

        self.assertIn("Find official documentation", update["plan"])
        self.assertIn("What is LangGraph?", str(model.messages))

    def test_planner_rejects_missing_query(self) -> None:
        model = FakeTextModel("Unused plan")
        planner_node = build_planner_node(model)

        with self.assertRaises(ValueError):
            planner_node({})

class ResearcherNodeTests(unittest.TestCase):
    """Verify the Researcher node without calling Tavily."""

    def test_researcher_performs_one_search(self) -> None:
        search_tool = FakeSearchTool(
            "Title: LangGraph documentation\n"
            "URL: https://docs.langchain.com/langgraph\n"
            "Summary: LangGraph supports stateful workflows.",
        )

        researcher_node = build_researcher_node(
            search_tool=search_tool,
            include_domains=["docs.langchain.com"],
            fetch_page=lambda _url: "LangGraph supports stateful workflows.",
        )

        state: ResearchState = {
            "query": "What is LangGraph?",
            "plan": "Find the official LangGraph documentation.",
        }

        update = researcher_node(state)

        self.assertIn(
            "LangGraph supports stateful workflows",
            update["research"],
        )
        self.assertEqual(
            search_tool.tool_input["include_domains"],
            ["docs.langchain.com"],
        )
        self.assertIn(
            "official documentation",
            search_tool.tool_input["query"],
        )

    def test_researcher_requires_planner_output(self) -> None:
        search_tool = FakeSearchTool("Unused result")
        researcher_node = build_researcher_node(
            search_tool=search_tool,
            include_domains=["docs.langchain.com"],
            fetch_page=lambda _url: "LangGraph supports stateful workflows.",
        )
        state: ResearchState = {
            "query": "What is LangGraph?",
        }

        with self.assertRaises(ValueError):
            researcher_node(state)

    def test_researcher_keeps_fetched_webpage_evidence(self) -> None:
        search_tool = FakeSearchTool(
            "URL: https://docs.example.org/state\n"
            "Summary: Search-only summary."
        )
        fetched_urls: list[str] = []

        def fake_fetch(url: str) -> str:
            fetched_urls.append(url)
            return (
                "Navigation links\n\n"
                "LangGraph state is shared between graph nodes and "
                "updated as the workflow executes."
            )

        node = build_researcher_node(search_tool, fetch_page=fake_fetch)
        research = node({"query": "LangGraph state", "plan": "Read docs"})["research"]

        self.assertEqual(fetched_urls, ["https://docs.example.org/state"])
        self.assertIn("LangGraph state is shared", research)
        self.assertIn("https://docs.example.org/state", research)
        self.assertNotIn("Search-only summary", research)

    def test_researcher_marks_failed_fetch(self) -> None:
        url = "https://docs.example.org/state"
        search_tool = FakeSearchTool(f"URL: {url}\nSummary: Unverified claim.")
        node = build_researcher_node(
            search_tool,
            fetch_page=lambda _url: f"Error fetching content from {url}: timeout",
        )

        research = node({"query": "LangGraph state", "plan": "Read docs"})["research"]

        self.assertIn("Webpage evidence unavailable", research)
        self.assertNotIn("Unverified claim", research)

    def test_researcher_limits_long_webpage_content(self) -> None:
        search_tool = FakeSearchTool("URL: https://docs.example.org/state")
        node = build_researcher_node(
            search_tool,
            fetch_page=lambda _url: "LangGraph state: " + "A" * 5000,
        )

        research = node({"query": "LangGraph state", "plan": "Read docs"})["research"]

        self.assertIn("LangGraph state:", research)
        self.assertLessEqual(len(research), 1800)

    def test_researcher_does_not_claim_lexical_match_when_none_exists(self) -> None:
        node = build_researcher_node(
            FakeSearchTool("URL: https://docs.example.org/other"),
            fetch_page=lambda _url: "Unrelated page content.",
        )

        research = node({"query": "checkpoint", "plan": "Read docs"})["research"]

        self.assertIn("Lexical match: none", research)

    def test_researcher_rejects_source_outside_allowed_domains(self) -> None:
        search_tool = FakeSearchTool(
            "URL: https://www.reddit.com/r/example\n"
            "Summary: Untrusted search result."
        )
        fetched_urls: list[str] = []

        def fake_fetch(url: str) -> str:
            fetched_urls.append(url)
            return "Content that must not be used."

        node = build_researcher_node(
            search_tool,
            include_domains=["docs.langchain.com"],
            fetch_page=fake_fetch,
        )

        research = node(
            {
                "query": "LangGraph State",
                "plan": "Read official documentation",
            }
        )["research"]

        self.assertEqual(fetched_urls, [])
        self.assertIn("no usable source URL", research)
        self.assertNotIn("Untrusted search result", research)

    def test_researcher_tries_next_source_after_fetch_failure(self) -> None:
        first_url = "https://docs.langchain.com/failed"
        second_url = "https://docs.langchain.com/oss/python/langgraph/graph-api"

        search_tool = FakeSearchTool(
            f"URL: {first_url}\n"
            "Summary: First result.\n\n"
            f"URL: {second_url}\n"
            "Summary: Second result."
        )
        fetched_urls: list[str] = []

        def fake_fetch(url: str) -> str:
            fetched_urls.append(url)

            if url == first_url:
                return f"Error fetching content from {url}: timeout"

            return (
                "LangGraph State represents the current snapshot "
                "of the application."
            )

        node = build_researcher_node(
            search_tool,
            include_domains=["docs.langchain.com"],
            fetch_page=fake_fetch,
        )

        research = node(
            {
                "query": "LangGraph State",
                "plan": "Read official documentation",
            }
        )["research"]

        self.assertEqual(fetched_urls, [first_url, second_url])
        self.assertIn(second_url, research)
        self.assertIn("current snapshot", research)
        self.assertNotIn(first_url, research)

class WriterNodeTests(unittest.TestCase):
    """Verify initial writing and evaluator-guided revision."""

    def test_writer_creates_initial_draft(self) -> None:
        model = FakeTextModel(
            "LangGraph supports stateful agent workflows.",
        )
        writer_node = build_writer_node(model)

        state: ResearchState = {
            "query": "What is LangGraph?",
            "plan": "Read the official documentation.",
            "research": (
                "URL: https://docs.langchain.com/langgraph\n"
                "LangGraph supports stateful workflows."
            ),
            "revision_count": 0,
        }

        update = writer_node(state)

        self.assertEqual(
            update["draft"],
            "LangGraph supports stateful agent workflows.",
        )
        self.assertIn(
            "https://docs.langchain.com/langgraph",
            str(model.messages),
        )

    def test_writer_uses_evaluator_actions_during_revision(self) -> None:
        model = FakeTextModel(
            "Revised report with an official source.",
        )
        writer_node = build_writer_node(model)

        state: ResearchState = {
            "query": "What is LangGraph?",
            "research": "Official LangGraph documentation.",
            "draft": "Previous incomplete report.",
            "revision_count": 1,
            "evaluation": {
                "score": 6,
                "problems": ["Missing citation"],
                "suggestions": ["Add an official source"],
                "improvement_actions": [
                    "Include the official documentation URL",
                ],
            },
        }

        update = writer_node(state)

        self.assertEqual(
            update["draft"],
            "Revised report with an official source.",
        )
        self.assertIn(
            "Include the official documentation URL",
            str(model.messages),
        )
        self.assertIn(
            "Previous incomplete report",
            str(model.messages),
        )

    def test_writer_rejects_missing_research(self) -> None:
        model = FakeTextModel("Unused report")
        writer_node = build_writer_node(model)

        state: ResearchState = {
            "query": "What is LangGraph?",
        }

        with self.assertRaises(ValueError):
            writer_node(state)

if __name__ == "__main__":
    unittest.main()
