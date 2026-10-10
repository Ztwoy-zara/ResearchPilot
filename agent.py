"""
ResearchPilot Lite Agent

Multi-Agent research workflow:
Planner -> Researcher -> Writer -> Evaluator
"""


from deepagents import create_deep_agent

from datetime import datetime

from research_agent.prompts import (
    RESEARCH_WORKFLOW_INSTRUCTIONS,
    SUBAGENT_DELEGATION_INSTRUCTIONS,
)


from research_agent.subagents import get_subagents


from research_agent.models import get_model

current_date = datetime.now().strftime("%Y-%m-%d")

# ==========================
# Main Agent Instructions
# ==========================

INSTRUCTIONS = (
    RESEARCH_WORKFLOW_INSTRUCTIONS
    + "\n\n"
    + SUBAGENT_DELEGATION_INSTRUCTIONS
    + f"\n\nCurrent date: {current_date}"
)



# ==========================
# Model
# ==========================

# 默认 GPT-4o-mini
#model = get_model("openai")


# 如果想测试 DeepSeek：
model = get_model("deepseek")



# ==========================
# Create Multi-Agent
# ==========================

subs = get_subagents(model)

agent = create_deep_agent(

    # Main Agent的大脑
    model=model,


    # 主Agent不直接搜索
    # 搜索能力属于Research Agent
    tools=[],


    # 主Agent负责协调
    system_prompt=INSTRUCTIONS,


    # 四个SubAgent
    subagents=subs,

)