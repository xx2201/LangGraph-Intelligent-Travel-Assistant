from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from langgraph_study.app.cli import (
        DEFAULT_USER_INPUT,
        extract_last_ai_text,
        get_user_input,
        invoke_agent,
        main,
        make_graph_config,
        next_thread_id,
        parse_args,
        print_console_text,
    )
else:
    from .app.cli import (
        DEFAULT_USER_INPUT,
        extract_last_ai_text,
        get_user_input,
        invoke_agent,
        main,
        make_graph_config,
        next_thread_id,
        parse_args,
        print_console_text,
    )

__all__ = [
    "DEFAULT_USER_INPUT",
    "extract_last_ai_text",
    "get_user_input",
    "invoke_agent",
    "main",
    "make_graph_config",
    "next_thread_id",
    "parse_args",
    "print_console_text",
]


if __name__ == "__main__":
    main()
