from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from langgraph_study.graph import build_graph


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


def test_graph_returns_direct_answer_without_tools() -> None:
    graph = build_graph(model=EchoModel(), tools=[])
    result = graph.invoke({"messages": [HumanMessage(content="你好")]})

    assert result["messages"][-1].content == "你好，我是旅行助手。"
    assert result["query_context"]["intent"] == "general"


def test_graph_completes_tool_loop() -> None:
    @tool
    def weather(city: str) -> str:
        """查询城市天气。"""
        return f"{city}今天天气晴。"

    model = ToolCallingModel()
    graph = build_graph(model=model, tools=[weather])
    result = graph.invoke({"messages": [HumanMessage(content="北京天气怎么样？")]})

    assert "北京今天天气晴" in result["messages"][-1].content
    assert result["query_context"]["normalized_city"] == "北京"
    assert model.calls == 2


def test_graph_requests_clarification_for_ambiguous_weather_question() -> None:
    graph = build_graph(model=GuardModel(), tools=[])
    result = graph.invoke({"messages": [HumanMessage(content="今天天气怎么样？")]})

    assert "哪个城市或地点" in result["messages"][-1].content
    assert result["query_context"]["needs_clarification"] is True
