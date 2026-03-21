from argparse import Namespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_study.main import (
    DEFAULT_USER_INPUT,
    extract_last_ai_text,
    get_user_input,
    invoke_agent,
    make_graph_config,
    next_thread_id,
)


def test_get_user_input_from_args() -> None:
    args = Namespace(
        input="帮我看看上海天气",
        thread_id=None,
        no_prompt=False,
        show_mermaid=False,
    )

    result = get_user_input(args, prompt=lambda _: "")

    assert result == "帮我看看上海天气"


def test_get_user_input_without_prompt_uses_default() -> None:
    args = Namespace(
        input=None,
        thread_id=None,
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


def test_make_graph_config_uses_thread_id() -> None:
    assert make_graph_config("thread-123") == {"configurable": {"thread_id": "thread-123"}}


def test_next_thread_id_returns_non_empty_string() -> None:
    thread_id = next_thread_id()

    assert isinstance(thread_id, str)
    assert thread_id


@pytest.mark.anyio
async def test_invoke_agent_uses_async_graph_and_thread_config() -> None:
    class FakeGraph:
        async def ainvoke(self, payload, config):
            return {"payload": payload, "config": config}

    result = await invoke_agent(FakeGraph(), "北京天气怎么样", "thread-42")

    assert result == {
        "payload": {"messages": [HumanMessage(content="北京天气怎么样")]},
        "config": {"configurable": {"thread_id": "thread-42"}},
    }
