import pytest
from langchain_core.messages import AIMessage, HumanMessage

from langgraph_study.app.cli import (
    ask_show_mermaid,
    ask_thread_id,
    extract_last_ai_text,
    get_user_input,
    invoke_agent,
    make_graph_config,
    next_thread_id,
)
from langgraph_study.core.config import DEFAULT_USER_INPUT


def test_get_user_input_returns_prompt_value() -> None:
    result = get_user_input(prompt=lambda _: "帮我看看上海天气")
    assert result == "帮我看看上海天气"


def test_get_user_input_returns_default_on_empty() -> None:
    result = get_user_input(prompt=lambda _: "")
    assert result == DEFAULT_USER_INPUT


def test_ask_show_mermaid_returns_true_for_yes() -> None:
    assert ask_show_mermaid(prompt=lambda _: "y") is True


def test_ask_show_mermaid_returns_false_for_empty() -> None:
    assert ask_show_mermaid(prompt=lambda _: "") is False


def test_ask_thread_id_reuses_existing_value() -> None:
    assert ask_thread_id(prompt=lambda _: "thread-demo") == "thread-demo"


def test_ask_thread_id_generates_new_value_when_empty() -> None:
    thread_id = ask_thread_id(prompt=lambda _: "")
    assert isinstance(thread_id, str)
    assert thread_id


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
