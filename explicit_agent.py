"""Explicit StateGraph entry point for ResearchPilot."""

from dotenv import load_dotenv

load_dotenv()

from research_agent.models import get_model
from research_agent.nodes import (
    build_evaluator_node,
    build_planner_node,
    build_researcher_node,
    build_writer_node,
)
from research_agent.tools import tavily_search
from research_agent.workflow import build_research_graph


model = get_model("deepseek")

planner_node = build_planner_node(model)
researcher_node = build_researcher_node(
    search_tool=tavily_search,
    include_domains=["docs.langchain.com"],
)
writer_node = build_writer_node(model)
evaluator_node = build_evaluator_node(model)

explicit_graph = build_research_graph(
    planner_node=planner_node,
    researcher_node=researcher_node,
    writer_node=writer_node,
    evaluator_node=evaluator_node,
)