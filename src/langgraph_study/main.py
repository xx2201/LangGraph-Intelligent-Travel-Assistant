from __future__ import annotations

import argparse

from .graph import build_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the minimal LangGraph study demo.")
    parser.add_argument(
        "--topic",
        default="我想先理解 LangGraph 的 state",
        help="Learning topic to route through the graph.",
    )
    parser.add_argument(
        "--background",
        default="已学习过 LangChain",
        help="Current background information.",
    )
    parser.add_argument(
        "--show-mermaid",
        action="store_true",
        help="Print the Mermaid definition of the compiled graph.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = build_graph()

    if args.show_mermaid:
        print(graph.get_graph().draw_mermaid())
        print()

    result = graph.invoke(
        {
            "topic": args.topic,
            "background": args.background,
            "notes": [],
        }
    )

    print("=== Topic ===")
    print(result["topic"])
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

