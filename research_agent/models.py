import os

from langchain_openai import ChatOpenAI


def get_gpt4o_mini():
    """
    GPT-4o-mini
    """

    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def get_deepseek():
    """
    DeepSeek Chat
    """

    return ChatOpenAI(
        model="deepseek-chat",
        temperature=0,
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL"),
    )


def get_model(provider="openai"):

    if provider == "openai":
        return get_gpt4o_mini()

    elif provider == "deepseek":
        return get_deepseek()

    else:
        raise ValueError(
            f"Unknown model provider: {provider}"
        )