from __future__ import annotations

from pathlib import Path

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from ..core.config import CHECKPOINT_DB_PATH
from ..integrations.llm import get_qwen_model
from ..integrations.mcp_tools import get_amap_tools, load_amap_tools
from ..integrations.studio_tools import get_studio_tools
from .nodes import analyze_query, clarify_query, create_assistant_node, route_after_analysis
from .state import TravelAssistantState


async def create_sqlite_checkpointer(db_path: str = CHECKPOINT_DB_PATH) -> AsyncSqliteSaver:
    """Open the SQLite checkpoint store used to persist conversation state."""

    checkpoint_path = Path(db_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    connection = await aiosqlite.connect(checkpoint_path)
    return AsyncSqliteSaver(connection)


def compile_graph(model=None, tools=None, checkpointer=None):
    """Compile the common graph structure from a model, tools, and optional checkpointer.

    This function defines nodes and edges only. It does not decide whether the tools
    are real MCP tools or lightweight studio placeholders. That choice is made by the
    caller before entering this function.
    """

    resolved_tools = list(tools or [])
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

    return builder.compile(checkpointer=checkpointer)


def build_studio_graph():
    """Build the graph used by LangGraph Studio.

    Studio should be able to draw the graph without starting the real MCP stdio
    subprocess during import. For that reason this entrypoint uses placeholder tools.
    """

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
