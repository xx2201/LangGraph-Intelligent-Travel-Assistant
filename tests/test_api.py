from langchain_core.messages import AIMessage, HumanMessage

from langgraph_study.backend.api import build_state_view, chunk_text


def test_chunk_text_splits_reply_into_small_pieces() -> None:
    chunks = chunk_text("abcdef", chunk_size=2)

    assert chunks == ["ab", "cd", "ef"]


def test_build_state_view_counts_messages_and_tool_calls() -> None:
    state = build_state_view(
        {
            "query_context": {"needs_clarification": False, "intent": "travel"},
            "messages": [
                HumanMessage(content="帮我做一个成都计划"),
                AIMessage(
                    content="",
                    tool_calls=[{"name": "weather", "args": {"city": "成都"}, "id": "1", "type": "tool_call"}],
                ),
                AIMessage(content="这是最终回答"),
            ],
        }
    )

    assert state.message_count == 3
    assert state.user_message_count == 1
    assert state.assistant_message_count == 1
    assert state.tool_call_count == 1
    assert state.next_route_hint == "assistant"
