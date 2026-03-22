from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Annotated, TypedDict

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


"""
和 tewts01.py 的核心区别主要有三点：
1. tewts01.py 直接在当前进程里用 @tool 定义工具。
2. 这个文件通过 MCP client 启动 math_server.py，再把远端工具加载成 LangChain tools。
3. MCP 工具默认走异步调用，所以这里用 graph.ainvoke()，而不是 graph.invoke()。

图结构本身不变，仍然是：
chatbot -> tools -> chatbot
"""


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_math_server_config() -> dict:
    """告诉 MCP client 如何启动本地的 math_server.py。"""

    server_path = Path(__file__).resolve().with_name("math_server.py")
    return {
        "transport": "stdio",
        "command": sys.executable,
        "args": ["-u", str(server_path)],
    }


async def load_math_tools():
    """向 MCP server 查询工具列表，并转成 LangChain tools。"""

    client = MultiServerMCPClient(
        {
            "math": build_math_server_config(),
        }
    )
    return tuple(await client.get_tools(server_name="math"))


def build_graph(tools):
    model = ChatTongyi(model="qwen3-max", temperature=0)
    model_with_tools = model.bind_tools(tools)

    async def chatbot(state: State):
        response = await model_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    graph_builder = StateGraph(State)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_node("tools", ToolNode(tools))

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("chatbot", tools_condition)
    graph_builder.add_edge("tools", "chatbot")

    return graph_builder.compile()


async def main() -> None:
    tools = await load_math_tools()
    print("loaded MCP tools =>", [tool.name for tool in tools])

    graph = build_graph(tools)
    result = await graph.ainvoke(
        {
            "messages": [
                HumanMessage(content="请帮我计算 10 加上 5，最后用一句中文回答我。")
            ]
        }
    )

    for msg in result["messages"]:
        print(type(msg).__name__, "=>", getattr(msg, "content", ""))
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print("tool_calls =>", msg.tool_calls)
        if getattr(msg, "name", None):
            print("tool_name =>", msg.name)


if __name__ == "__main__":
    asyncio.run(main())
