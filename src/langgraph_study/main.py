from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from langgraph_study.config import DEFAULT_USER_INPUT
    from langgraph_study.graph import build_graph
else:
    from .config import DEFAULT_USER_INPUT
    from .graph import build_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the travel assistant agent demo.")
    parser.add_argument(
        "--input",
        help="One-shot travel question to feed into the agent.",
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


async def invoke_agent(graph, messages):
    return await graph.ainvoke({"messages": messages})


def main() -> None:
    args = parse_args()
    graph = build_graph()

    if args.show_mermaid:
        print(graph.get_graph().draw_mermaid())
        print()

    messages = []
    one_shot = bool(args.input or args.no_prompt)

    while True:
        user_input = get_user_input(args)
        if user_input.lower() in {"exit", "quit", "/exit"}:
            break
        if user_input.lower() == "/reset":
            messages = []
            print("会话已清空。")
            if one_shot:
                break
            continue

        messages.append(HumanMessage(content=user_input))
        result = asyncio.run(invoke_agent(graph, messages))
        messages = result["messages"]

        print("=== Assistant ===")
        print(extract_last_ai_text(messages))
        print()

        if one_shot:
            break


if __name__ == "__main__":
    main()
