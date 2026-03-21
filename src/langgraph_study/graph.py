from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .llm import get_qwen_model
from .mcp_tools import get_amap_tools
from .nodes import (
    analyze_query,
    clarify_query,
    create_assistant_node,
    route_after_analysis,
)
from .state import TravelAssistantState


def build_graph(model=None, tools=None):
    resolved_tools = list(tools) if tools is not None else list(get_amap_tools())
    resolved_model = model or get_qwen_model()
    assistant_node = create_assistant_node(resolved_model.bind_tools(resolved_tools))

    builder = StateGraph(TravelAssistantState)
    builder.add_node("analyze_query", analyze_query)
    builder.add_node("clarify", clarify_query)
    builder.add_node("assistant", assistant_node)
    builder.add_node("tools", ToolNode(resolved_tools))
    builder.add_edge(START, "analyze_query")
    builder.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "clarify": "clarify",
            "assistant": "assistant",
        },
    )
    builder.add_edge("clarify", END)
    builder.add_conditional_edges("assistant", tools_condition)
    builder.add_edge("tools", "assistant")

    return builder.compile()
