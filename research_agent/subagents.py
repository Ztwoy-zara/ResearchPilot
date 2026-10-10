"""
Sub-agent definitions for ResearchPilot Lite.
"""
from datetime import datetime

from .prompts import (
    PLANNER_INSTRUCTIONS,
    RESEARCHER_INSTRUCTIONS,
    WRITER_INSTRUCTIONS,
    EVALUATOR_INSTRUCTIONS,
)

from .tools import (
    tavily_search,
    think_tool,
)

current_date = datetime.now().strftime("%Y-%m-%d")

def get_subagents(model):
    """
    Create ResearchPilot Lite sub-agents.

    Agents:
    1. Planner
    2. Researcher
    3. Writer
    4. Evaluator
    """

    return [

        # =====================
        # Planner Agent
        # =====================

        {
            "name": "planner",

           "description": (
                "First step of every research task. "
                "Analyze user requests, identify research goals, "
                "create key questions and a structured research plan. "
                "Do not search the web and do not write reports."
            ),

            "system_prompt": PLANNER_INSTRUCTIONS,

            "model": model,

            "tools": [],
        },


        # =====================
        # Research Agent
        # =====================

        {
            "name": "researcher",

            "description": (
                "Use this agent after planner. "
                "Execute the research plan, "
                "collect evidence and sources. "
                "Use search tools when necessary."
            ),

            "system_prompt":
            RESEARCHER_INSTRUCTIONS.format(
                date=current_date
            ),

            "model": model,

            "tools": [
                tavily_search,
                think_tool,
            ],
        },


        # =====================
        # Writer Agent
        # =====================

        {
            "name": "writer",

            "description": (
                "Use after researcher. "
                "Transform research findings into a structured Markdown report."
            ),

            "system_prompt": WRITER_INSTRUCTIONS,

            "model": model,

            "tools": [],
        },


        # =====================
        # Evaluator Agent
        # =====================

        {
            "name": "evaluator",

            "description": (
                "Use after writer. "
                "Evaluate report completeness, accuracy and structure."
            ),

            "system_prompt": EVALUATOR_INSTRUCTIONS,

            "model": model,

            "tools": [],
        },

    ]