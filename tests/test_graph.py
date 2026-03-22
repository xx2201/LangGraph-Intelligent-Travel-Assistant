from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
import pytest

from langgraph_study.assistant.graph import (
    build_studio_graph,
    build_graph_with_checkpointer,
    build_persistent_graph,
)
from langgraph_study.assistant.nodes import run_bound_model


def make_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


class EchoModel:
    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        return AIMessage(content="你好，我是旅行助手。")


class ToolCallingModel:
    def __init__(self) -> None:
        self.calls = 0

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        self.calls += 1
        if not any(isinstance(message, ToolMessage) for message in messages):
            assert any(isinstance(message, SystemMessage) for message in messages)
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "weather",
                        "args": {"city": "北京"},
                        "id": "call_weather_1",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="北京今天天气晴，适合出门。")


class GuardModel:
    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        raise AssertionError("需要澄清的问题不应进入模型节点。")


class HistoryModel:
    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, messages):
        human_count = sum(isinstance(message, HumanMessage) for message in messages)
        return AIMessage(content=f"human_count={human_count}")


class StreamingHistoryModel:
    def bind_tools(self, tools):
        self.tools = tools
        return self

    async def astream(self, messages, config=None, **kwargs):
        yield AIMessageChunk(content="hello ")
        yield AIMessageChunk(content="world")


class NoEmptyBindToolsModel:
    def bind_tools(self, tools):
        if not tools:
            raise AssertionError("empty tool binding should be skipped")
        self.tools = tools
        return self

    def invoke(self, messages):
        return AIMessage(content="普通问答无需工具。")


def test_studio_graph_builds_without_real_mcp_tools(monkeypatch) -> None:
    import langgraph_study.assistant.graph as graph_module

    monkeypatch.setattr(graph_module, "get_qwen_model", lambda: EchoModel())
    monkeypatch.setattr(
        graph_module,
        "get_amap_tools",
        lambda: (_ for _ in ()).throw(AssertionError("studio graph should not load runtime MCP tools")),
    )

    graph = build_studio_graph()

    assert graph is not None


def test_graph_returns_direct_answer_without_tools() -> None:
    graph = build_graph_with_checkpointer(
        model=EchoModel(),
        tools=[],
        checkpointer=InMemorySaver(),
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="你好")]},
        config=make_config("thread-echo"),
    )

    assert result["messages"][-1].content == "你好，我是旅行助手。"
    assert result["query_context"]["intent"] == "general"
    assert result["active_agent"] == "general_agent"


def test_general_agent_skips_empty_bind_tools() -> None:
    graph = build_graph_with_checkpointer(
        model=NoEmptyBindToolsModel(),
        tools=[],
        checkpointer=InMemorySaver(),
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="你好")]},
        config=make_config("thread-general-no-bind"),
    )

    assert result["messages"][-1].content == "普通问答无需工具。"
    assert result["active_agent"] == "general_agent"


def test_graph_completes_tool_loop() -> None:
    @tool
    def weather(city: str) -> str:
        """查询城市天气。"""
        return f"{city}今天天气晴。"

    model = ToolCallingModel()
    graph = build_graph_with_checkpointer(
        model=model,
        tools=[weather],
        checkpointer=InMemorySaver(),
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="北京天气怎么样？")]},
        config=make_config("thread-weather"),
    )

    assert "北京今天天气晴" in result["messages"][-1].content
    assert result["query_context"]["normalized_city"] == "北京"
    assert result["active_agent"] == "weather_agent"


def test_graph_requests_clarification_for_ambiguous_weather_question() -> None:
    graph = build_graph_with_checkpointer(
        model=GuardModel(),
        tools=[],
        checkpointer=InMemorySaver(),
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="今天天气怎么样？")]},
        config=make_config("thread-clarify"),
    )

    assert "哪个城市或地点" in result["messages"][-1].content
    assert result["query_context"]["needs_clarification"] is True


def test_graph_routes_geocode_requests_to_geo_agent() -> None:
    graph = build_graph_with_checkpointer(
        model=EchoModel(),
        tools=[],
        checkpointer=InMemorySaver(),
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="帮我把天安门广场转成坐标")]},
        config=make_config("thread-geo"),
    )

    assert result["query_context"]["intent"] == "geocode"
    assert result["active_agent"] == "geo_agent"


def test_graph_routes_travel_requests_to_travel_agent() -> None:
    graph = build_graph_with_checkpointer(
        model=EchoModel(),
        tools=[],
        checkpointer=InMemorySaver(),
    )
    result = graph.invoke(
        {"messages": [HumanMessage(content="上海和杭州这周末哪个更适合出行？")]},
        config=make_config("thread-travel"),
    )

    assert result["query_context"]["intent"] == "travel"
    assert result["active_agent"] == "travel_agent"


def test_graph_restores_context_with_same_thread_id() -> None:
    graph = build_graph_with_checkpointer(
        model=HistoryModel(),
        tools=[],
        checkpointer=InMemorySaver(),
    )
    config = make_config("thread-history")

    first = graph.invoke(
        {"messages": [HumanMessage(content="北京天气怎么样？")]},
        config=config,
    )
    second = graph.invoke(
        {"messages": [HumanMessage(content="那上海呢？")]},
        config=config,
    )

    assert first["messages"][-1].content == "human_count=1"
    assert second["messages"][-1].content == "human_count=2"


@pytest.mark.anyio
async def test_persistent_graph_restores_context_across_graph_instances(tmp_path) -> None:
    db_path = tmp_path / "checkpoints.sqlite"
    config = make_config("thread-persistent")

    first_graph = await build_persistent_graph(
        model=HistoryModel(),
        tools=[],
        db_path=str(db_path),
    )
    try:
        first = await first_graph.ainvoke(
            {"messages": [HumanMessage(content="北京天气怎么样？")]},
            config=config,
        )

        second_graph = await build_persistent_graph(
            model=HistoryModel(),
            tools=[],
            db_path=str(db_path),
        )
        try:
            second = await second_graph.ainvoke(
                {"messages": [HumanMessage(content="那上海呢？")]},
                config=config,
            )
        finally:
            await second_graph.checkpointer.conn.close()
    finally:
        await first_graph.checkpointer.conn.close()

    assert first["messages"][-1].content == "human_count=1"
    assert second["messages"][-1].content == "human_count=2"


@pytest.mark.anyio
async def test_run_bound_model_prefers_astream_and_merges_chunks() -> None:
    model = StreamingHistoryModel()

    result = await run_bound_model(model, [HumanMessage(content="hi")])

    assert isinstance(result, AIMessage)
    assert result.content == "hello world"
