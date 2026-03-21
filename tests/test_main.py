from argparse import Namespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_study.main import (
    DEFAULT_USER_INPUT,
    extract_last_ai_text,
    get_user_input,
    invoke_agent,
)


def test_get_user_input_from_args() -> None:
    args = Namespace(
        input="帮我看看上海天气",
        no_prompt=False,
        show_mermaid=False,
    )

    result = get_user_input(args, prompt=lambda _: "")

    assert result == "帮我看看上海天气"


def test_get_user_input_without_prompt_uses_default() -> None:
    args = Namespace(
        input=None,
        no_prompt=True,
        show_mermaid=False,
    )

    result = get_user_input(args, prompt=lambda _: "")

    assert result == DEFAULT_USER_INPUT


def test_extract_last_ai_text() -> None:
    messages = [
        HumanMessage(content="北京天气怎么样"),
        AIMessage(content="北京今天天气晴。"),
    ]

    assert extract_last_ai_text(messages) == "北京今天天气晴。"


@pytest.mark.anyio
async def test_invoke_agent_uses_async_graph() -> None:
    class FakeGraph:
        async def ainvoke(self, payload):
            return payload

    messages = [HumanMessage(content="北京天气怎么样")]

    result = await invoke_agent(FakeGraph(), messages)

    assert result == {"messages": messages}
