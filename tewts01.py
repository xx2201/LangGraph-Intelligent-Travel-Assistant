from typing import TypedDict, Annotated

from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# =========================
# 1) 定义工具（本地 Python 函数）
# =========================
@tool
def multiply(a: int, b: int) -> int:
    """把两个整数相乘。"""
    return a * b



# =========================
# =========================
@tool
def add(a: int, b: int) -> int:
    """把两个整数相加。"""
    return a + b



# =========================
# =========================
@tool
def div(a: int, b: int) -> int:
    """把两个整数相除。"""
    return a / b



# =========================
# =========================
@tool
def sub(a: int, b: int) -> int:
    """把两个整数相减。"""
    return a - b


tools = [sub,multiply,add,div]


# =========================
# 2) 定义状态
# LangGraph 里状态就是节点之间传递的数据
# =========================
class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# =========================
# 3) 初始化模型，并绑定工具
# bind_tools = 把工具 schema 告诉模型
# =========================
model = ChatTongyi(model="qwen3-max", temperature=0)
model_with_tools = model.bind_tools(tools)


# =========================
# 4) 定义“模型节点”
# 作用：让模型看当前 messages，并决定
# - 直接回答
# - 还是输出 tool call
# =========================
def chatbot(state: State):
    response = model_with_tools.invoke(state["messages"])
    return {"messages": [response]}


# =========================
# 5) 构图
# chatbot -> 如果模型要调工具 -> tools
# tools -> 再回 chatbot
# 如果模型不需要工具 -> END
# =========================
graph_builder = StateGraph(State)

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools))

graph_builder.add_edge(START, "chatbot")
graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,   # 自动判断 AIMessage 里是否包含 tool_calls
)
graph_builder.add_edge("tools", "chatbot")

graph = graph_builder.compile()


# =========================
# 6) 运行
# =========================
result = graph.invoke(
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