from __future__ import annotations

import argparse
import asyncio
import sys
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage

from ..assistant.graph import build_persistent_graph
from ..core.config import DEFAULT_USER_INPUT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the travel assistant agent demo.")
    parser.add_argument(
        "--input",
        help="One-shot travel question to feed into the agent.",
    )
    parser.add_argument(
        "--thread-id",
        help="Reuse an existing thread id to continue a previous conversation.",
    )
    parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Run without interactive prompts and fall back to defaults.",
    )
    parser.add_argument(
        "--show-mermaid",
        action="store_true",
        help="Print the Mermaid definition of the compiled graph.",
    )
    return parser.parse_args()


def get_user_input(args: argparse.Namespace, prompt=input) -> str:
    user_input = (args.input or "").strip()

    if user_input:
        return user_input

    if args.no_prompt:
        return DEFAULT_USER_INPUT

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


async def run_cli(args: argparse.Namespace) -> None:
    graph = await build_persistent_graph()
    thread_id = args.thread_id or next_thread_id()

    if args.show_mermaid:
        print(graph.get_graph().draw_mermaid())
        print()

    one_shot = bool(args.input or args.no_prompt)
    print(f"当前会话 thread_id: {thread_id}")

    while True:
        user_input = get_user_input(args)
        if user_input.lower() in {"exit", "quit", "/exit"}:
            break
        if user_input.lower() == "/reset":
            thread_id = next_thread_id()
            print(f"会话已清空。新的 thread_id: {thread_id}")
            if one_shot:
                break
            continue

        result = await invoke_agent(graph, user_input, thread_id)

        print("=== Assistant ===")
        print_console_text(extract_last_ai_text(result["messages"]))
        print()

        if one_shot:
            break


def main() -> None:
    args = parse_args()
    asyncio.run(run_cli(args))
