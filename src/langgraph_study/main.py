from __future__ import annotations

import argparse

from .config import DEFAULT_BACKGROUND, DEFAULT_INPUT
from .graph import build_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal LangGraph study demo.")
    parser.add_argument(
        "--input",
        help="Natural language learning request to feed into the graph.",
    )
    parser.add_argument(
        "--topic",
        help="Backward-compatible alias of --input.",
    )
    parser.add_argument(
        "--background",
        help="Current background information.",
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


def build_initial_state(
    args: argparse.Namespace,
    prompt=input,
) -> dict[str, object]:
    user_input = (args.input or args.topic or "").strip()
    background = (args.background or "").strip()

    if not user_input:
        if args.no_prompt:
            user_input = DEFAULT_INPUT
        else:
            user_input = prompt(
                f"请输入你当前想学习的问题或主题（回车使用默认值：{DEFAULT_INPUT}）："
            ).strip()
            if not user_input:
                user_input = DEFAULT_INPUT

    if not background:
        if args.no_prompt:
            background = DEFAULT_BACKGROUND
        else:
            background = prompt(
                f"请输入你的背景（回车使用默认值：{DEFAULT_BACKGROUND}）："
            ).strip()
            if not background:
                background = DEFAULT_BACKGROUND

    return {
        "input": user_input,
        "background": background,
        "notes": [],
    }


def main() -> None:
    args = parse_args()
    graph = build_graph()

    if args.show_mermaid:
        print(graph.get_graph().draw_mermaid())
        print()

    initial_state = build_initial_state(args)
    result = graph.invoke(initial_state)

    print("=== Topic ===")
    print(result["topic"])
    print()
    print("=== Source ===")
    print(result["response_source"])
    print()
    if result.get("response_error"):
        print("=== Model Error ===")
        print(result["response_error"])
        print()
    print("=== Route ===")
    print(result["route"])
    print()
    print("=== Answer ===")
    print(result["answer"])
    print()
    print("=== Next Step ===")
    print(result["next_step"])
    print()
    print("=== Notes ===")
    for note in result["notes"]:
        print(f"- {note}")


if __name__ == "__main__":
    main()
