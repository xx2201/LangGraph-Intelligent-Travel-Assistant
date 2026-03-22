from __future__ import annotations

from copy import copy
from pathlib import Path
from typing import Iterable

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ..core.config import CHECKPOINT_DB_PATH
from ..integrations.llm import get_qwen_model
from ..integrations.mcp_tools import get_amap_tools, load_amap_tools
from ..integrations.studio_tools import get_studio_tools
from .nodes import (
    create_specialist_node,
    analyze_query,
    clarify_query,
    route_after_analysis,
    route_to_specialist,
    select_specialist_agent,
)
from .state import TravelAssistantState


async def create_sqlite_checkpointer(db_path: str = CHECKPOINT_DB_PATH) -> AsyncSqliteSaver:
    """Open the SQLite checkpoint store used to persist conversation state."""

    checkpoint_path = Path(db_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(checkpoint_path)
    return AsyncSqliteSaver(connection)


def tool_name(tool) -> str:
    """Return the public tool name used by LangChain/LangGraph."""

    return getattr(tool, "name", getattr(tool, "__name__", str(tool)))


def select_tools(tools: Iterable, allowed_names: set[str]) -> list:
    """Pick the tool subset owned by a specialist agent."""

    return [tool for tool in tools if tool_name(tool) in allowed_names]


def clone_model(model):
    """Create a shallow model copy so each agent can bind its own tool subset."""

    try:
        return copy(model)
    except Exception:  # pragma: no cover - defensive fallback
        return model


def bind_agent_model(model, tools):
    """Bind one specialist tool subset onto a model copy."""

    agent_model = clone_model(model)
    return agent_model.bind_tools(list(tools))


def compile_graph(model=None, tools=None, checkpointer=None):
    """Compile the multi-agent graph from a model, tools, and optional checkpointer."""

    resolved_tools = list(tools or [])
    resolved_model = model or get_qwen_model()

    weather_tools = select_tools(resolved_tools, {"weather", "input_tips"})
    geo_tools = select_tools(resolved_tools, {"geocode", "reverse_geocode", "input_tips"})
    travel_tools = select_tools(
        resolved_tools,
        {"weather", "input_tips", "geocode", "reverse_geocode"},
    )

    weather_agent = create_specialist_node(
        bind_agent_model(resolved_model, weather_tools),
        "weather_agent",
    )
    geo_agent = create_specialist_node(
        bind_agent_model(resolved_model, geo_tools),
        "geo_agent",
    )
    travel_agent = create_specialist_node(
        bind_agent_model(resolved_model, travel_tools),
        "travel_agent",
    )
    general_agent = create_specialist_node(
        bind_agent_model(resolved_model, []),
        "general_agent",
    )

    builder = StateGraph(TravelAssistantState)
    builder.add_node("analyze_query", analyze_query)
    builder.add_node("clarify", clarify_query)
    builder.add_node("select_agent", select_specialist_agent)
    builder.add_node("weather_agent", weather_agent)
    builder.add_node("geo_agent", geo_agent)
    builder.add_node("travel_agent", travel_agent)
    builder.add_node("general_agent", general_agent)
    builder.add_node("weather_tools", ToolNode(weather_tools))
    builder.add_node("geo_tools", ToolNode(geo_tools))
    builder.add_node("travel_tools", ToolNode(travel_tools))

    builder.add_edge(START, "analyze_query")
    builder.add_conditional_edges(
        "analyze_query",
        route_after_analysis,
        {
            "clarify": "clarify",
            "select_agent": "select_agent",
        },
    )
    builder.add_edge("clarify", END)
    builder.add_conditional_edges(
        "select_agent",
        route_to_specialist,
        {
            "weather_agent": "weather_agent",
            "geo_agent": "geo_agent",
            "travel_agent": "travel_agent",
            "general_agent": "general_agent",
        },
    )
    builder.add_conditional_edges(
        "weather_agent",
        tools_condition,
        {
            "tools": "weather_tools",
            "__end__": END,
        },
    )
    builder.add_conditional_edges(
        "geo_agent",
        tools_condition,
        {
            "tools": "geo_tools",
            "__end__": END,
        },
    )
    builder.add_conditional_edges(
        "travel_agent",
        tools_condition,
        {
            "tools": "travel_tools",
            "__end__": END,
        },
    )
    builder.add_conditional_edges(
        "general_agent",
        tools_condition,
        {
            "tools": END,
            "__end__": END,
        },
    )
    builder.add_edge("weather_tools", "weather_agent")
    builder.add_edge("geo_tools", "geo_agent")
    builder.add_edge("travel_tools", "travel_agent")

    return builder.compile(checkpointer=checkpointer)


def build_studio_graph():
    """Build the graph used by LangGraph Studio."""

    return compile_graph(model=None, tools=get_studio_tools(), checkpointer=None)


def build_runtime_graph(model=None, tools=None):
    """Build the runtime graph that uses the real MCP-backed tool set."""

    resolved_tools = list(tools) if tools is not None else list(get_amap_tools())
    return compile_graph(model=model, tools=resolved_tools, checkpointer=None)


def build_graph():
    """Backward-compatible alias that points Studio to the safe graph factory."""

    return build_studio_graph()


def build_graph_for_test(model=None, tools=None):
    """Build a graph for unit tests where model and tools are injected explicitly."""

    return compile_graph(model=model, tools=tools, checkpointer=None)


def build_runtime_graph_with_checkpointer(model=None, tools=None, checkpointer=None):
    """Build the real runtime graph with an externally provided checkpointer."""

    resolved_tools = list(tools) if tools is not None else list(get_amap_tools())
    return compile_graph(model=model, tools=resolved_tools, checkpointer=checkpointer)


def build_graph_with_checkpointer(model=None, tools=None, checkpointer=None):
    """Backward-compatible wrapper kept for the existing tests."""

    return build_runtime_graph_with_checkpointer(
        model=model,
        tools=tools,
        checkpointer=checkpointer,
    )


async def build_persistent_graph(model=None, tools=None, db_path: str = CHECKPOINT_DB_PATH):
    """Build the real runtime graph and attach a SQLite checkpointer to it."""

    checkpointer = await create_sqlite_checkpointer(db_path)
    resolved_tools = list(tools) if tools is not None else list(await load_amap_tools())
    return compile_graph(model=model, tools=resolved_tools, checkpointer=checkpointer)
