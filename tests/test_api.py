from langgraph_study.backend.api import build_process_updates, chunk_text, extract_stream_text


def test_chunk_text_splits_reply_into_small_pieces() -> None:
    chunks = chunk_text("abcdef", chunk_size=2)

    assert chunks == ["ab", "cd", "ef"]


def test_extract_stream_text_handles_plain_string_content() -> None:
    class FakeChunk:
        content = "hello"

    assert extract_stream_text(FakeChunk()) == "hello"


def test_extract_stream_text_handles_list_content() -> None:
    class FakeChunk:
        content = [{"text": "hel"}, {"text": "lo"}]

    assert extract_stream_text(FakeChunk()) == "hello"


def test_build_process_updates_marks_route_choice() -> None:
    tracker = {"assistant_round": 0, "current_assistant_key": "", "tool_round": 0, "tool_runs": {}}

    updates = build_process_updates(
        {
            "event": "on_chain_end",
            "name": "route_after_analysis",
            "data": {"output": "select_agent"},
        },
        tracker,
    )

    assert updates[0]["key"] == "route_after_analysis"
    assert updates[0]["status"] == "done"
    assert "选择专职 agent" in updates[0]["detail"]


def test_build_process_updates_marks_agent_selection() -> None:
    tracker = {"assistant_round": 0, "current_assistant_key": "", "tool_round": 0, "tool_runs": {}}

    updates = build_process_updates(
        {
            "event": "on_chain_end",
            "name": "select_agent",
            "data": {
                "output": {
                    "active_agent": "weather_agent",
                    "agent_selection_reason": "识别到天气意图。",
                }
            },
        },
        tracker,
    )

    assert updates[0]["key"] == "select_agent"
    assert updates[0]["status"] == "done"
    assert "weather_agent" in updates[0]["detail"]


def test_build_process_updates_tracks_tool_lifecycle() -> None:
    tracker = {"assistant_round": 0, "current_assistant_key": "", "tool_round": 0, "tool_runs": {}}

    start_updates = build_process_updates(
        {
            "event": "on_tool_start",
            "name": "weather",
            "run_id": "tool-run-1",
            "data": {"input": {"city": "北京"}},
        },
        tracker,
    )
    end_updates = build_process_updates(
        {
            "event": "on_tool_end",
            "name": "weather",
            "run_id": "tool-run-1",
            "data": {"output": {"city": "北京", "weather": "晴"}},
        },
        tracker,
    )

    assert start_updates[0]["key"] == "tool_1"
    assert start_updates[0]["status"] == "running"
    assert end_updates[0]["key"] == "tool_1"
    assert end_updates[0]["status"] == "done"
