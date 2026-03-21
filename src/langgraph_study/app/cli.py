from __future__ import annotations

import asyncio
import sys
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from ..assistant.graph import build_persistent_graph
from ..core.config import DEFAULT_USER_INPUT


def ask_yes_no(question: str, prompt=input) -> bool:
    return prompt(question).strip().lower() in {"y", "yes"}


def ask_show_mermaid(prompt=input) -> bool:
    return ask_yes_no("是否先打印 Mermaid 图结构？(y/N)：", prompt=prompt)


def ask_thread_id(prompt=input) -> str:
    existing_thread_id = prompt("输入已有 thread_id 可继续旧会话；直接回车创建新会话：").strip()
    return existing_thread_id or next_thread_id()


def get_user_input(prompt=input) -> str:
    return (
        prompt(f"你想咨询什么旅行问题？（回车使用默认值：{DEFAULT_USER_INPUT}）：").strip()
        or DEFAULT_USER_INPUT
    )


def extract_last_ai_text(messages) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage):
            if isinstance(message.content, str):
                return message.content
            return str(message.content)
    return ""


def print_console_text(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sanitized = text.encode(encoding, errors="replace").decode(encoding)
        print(sanitized)


def make_graph_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


async def invoke_agent(graph, user_input: str, thread_id: str):
    return await graph.ainvoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=make_graph_config(thread_id),
    )


def next_thread_id() -> str:
    return uuid4().hex


async def run_cli(prompt=input) -> None:
    graph = await build_persistent_graph()
    show_mermaid = ask_show_mermaid(prompt=prompt)
    thread_id = ask_thread_id(prompt=prompt)

    if show_mermaid:
        print(graph.get_graph().draw_mermaid())
        print()

    print(f"当前会话 thread_id: {thread_id}")

    while True:
        user_input = get_user_input(prompt=prompt)
        if user_input.lower() in {"exit", "quit", "/exit"}:
            break
        if user_input.lower() == "/reset":
            thread_id = next_thread_id()
            print(f"会话已清空。新的 thread_id: {thread_id}")
            continue

        result = await invoke_agent(graph, user_input, thread_id)

        print("=== Assistant ===")
        print_console_text(extract_last_ai_text(result["messages"]))
        print()


def main() -> None:
    asyncio.run(run_cli())


if __name__ == "__main__":
    main()
