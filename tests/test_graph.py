from langgraph_study.graph import build_graph


def test_state_route(monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    graph = build_graph()
    result = graph.invoke(
        {
            "input": "我想重点理解 state",
            "background": "学过 LangChain",
            "notes": [],
        }
    )

    assert result["route"] == "state"
    assert "State" in result["answer"]
    assert result["response_source"] == "fallback"
    assert result["topic"] == "我想重点理解 state"
    assert len(result["notes"]) >= 5


def test_control_flow_route(monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    graph = build_graph()
    result = graph.invoke(
        {
            "input": "条件边和 edge 是怎么工作的",
            "background": "学过 LangChain",
            "notes": [],
        }
    )

    assert result["route"] == "control_flow"
    assert "Conditional Edge" in result["answer"]
    assert result["response_source"] == "fallback"


def test_default_route(monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    graph = build_graph()
    result = graph.invoke(
        {
            "input": "LangGraph 和 LangChain 的关系",
            "background": "学过 LangChain",
            "notes": [],
        }
    )

    assert result["route"] == "overview"
    assert "状态机" in result["answer"]


def test_input_defaults_when_missing(monkeypatch) -> None:
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)
    graph = build_graph()
    result = graph.invoke({"notes": []})

    assert result["topic"] == "我想先理解 LangGraph 的 state"
    assert result["background"] == "已学习过 LangChain"
