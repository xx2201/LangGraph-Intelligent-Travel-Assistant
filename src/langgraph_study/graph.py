from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    analyze_topic,
    explain_control_flow,
    explain_memory,
    explain_overview,
    explain_state,
    finalize,
    normalize_input,
)
from .state import LearningState, Route


def route_topic(state: LearningState) -> Route:
    return state["route"]


def build_graph():
    builder = StateGraph(LearningState)

    builder.add_node("normalize_input", normalize_input)
    builder.add_node("analyze_topic", analyze_topic)
    builder.add_node("overview", explain_overview)
    builder.add_node("state", explain_state)
    builder.add_node("control_flow", explain_control_flow)
    builder.add_node("memory", explain_memory)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "normalize_input")
    builder.add_edge("normalize_input", "analyze_topic")
    builder.add_conditional_edges(
        "analyze_topic",
        route_topic,
        {
            "overview": "overview",
            "state": "state",
            "control_flow": "control_flow",
            "memory": "memory",
        },
    )

    builder.add_edge("overview", "finalize")
    builder.add_edge("state", "finalize")
    builder.add_edge("control_flow", "finalize")
    builder.add_edge("memory", "finalize")
    builder.add_edge("finalize", END)

    return builder.compile()


graph = build_graph()
