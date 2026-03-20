from argparse import Namespace

from langgraph_study.main import (
    DEFAULT_BACKGROUND,
    DEFAULT_INPUT,
    build_initial_state,
)


def test_build_initial_state_from_args() -> None:
    args = Namespace(
        input="我想学习条件边",
        topic="我想学习条件边",
        background="已学习过 LangChain",
        no_prompt=False,
        show_mermaid=False,
    )

    state = build_initial_state(args, prompt=lambda _: "")

    assert state["input"] == "我想学习条件边"
    assert state["background"] == "已学习过 LangChain"
    assert state["notes"] == []


def test_build_initial_state_from_prompt() -> None:
    answers = iter(["我想先理解 state", ""])
    args = Namespace(
        input=None,
        topic=None,
        background=None,
        no_prompt=False,
        show_mermaid=False,
    )

    state = build_initial_state(args, prompt=lambda _: next(answers))

    assert state["input"] == "我想先理解 state"
    assert state["background"] == DEFAULT_BACKGROUND


def test_build_initial_state_without_prompt_uses_defaults() -> None:
    args = Namespace(
        input=None,
        topic=None,
        background=None,
        no_prompt=True,
        show_mermaid=False,
    )

    state = build_initial_state(args, prompt=lambda _: "")

    assert state["input"] == DEFAULT_INPUT
    assert state["background"] == DEFAULT_BACKGROUND
