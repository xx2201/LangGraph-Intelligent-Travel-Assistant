from langgraph_study.graph import build_graph


def test_state_route() -> None:
    graph = build_graph()
    result = graph.invoke(
        {
            "topic": "我想重点理解 state",
            "background": "学过 LangChain",
            "notes": [],
        }
    )

    assert result["route"] == "state"
    assert "State" in result["answer"]
    assert len(result["notes"]) >= 3


def test_control_flow_route() -> None:
    graph = build_graph()
    result = graph.invoke(
        {
            "topic": "条件边和 edge 是怎么工作的",
            "background": "学过 LangChain",
            "notes": [],
        }
    )

    assert result["route"] == "control_flow"
    assert "Conditional Edge" in result["answer"]


def test_default_route() -> None:
    graph = build_graph()
    result = graph.invoke(
        {
            "topic": "LangGraph 和 LangChain 的关系",
            "background": "学过 LangChain",
            "notes": [],
        }
    )

    assert result["route"] == "overview"
    assert "状态机" in result["answer"]

