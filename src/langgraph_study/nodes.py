from __future__ import annotations

from .state import LearningState


def analyze_topic(state: LearningState) -> LearningState:
    topic = state["topic"].lower()

    if "state" in topic:
        route = "state"
    elif "edge" in topic or "条件边" in topic or "branch" in topic:
        route = "control_flow"
    elif "memory" in topic or "checkpoint" in topic:
        route = "memory"
    else:
        route = "overview"

    return {
        "route": route,
        "notes": [
            f"主题已识别为: {route}",
            f"已有背景: {state.get('background', '未提供')}",
        ],
    }


def explain_overview(_: LearningState) -> LearningState:
    return {
        "answer": (
            "LangGraph 的核心不是提示词，而是显式状态机。"
            "你需要先理解状态如何在节点之间流动，再去理解 agent。"
        ),
        "next_step": "继续学习 StateGraph、节点返回值和 compile() 的关系。",
        "notes": ["进入 overview 节点。"],
    }


def explain_state(_: LearningState) -> LearningState:
    return {
        "answer": (
            "State 是 LangGraph 的中心。每个节点读取共享状态，并返回部分更新。"
            "图执行的本质是状态在节点之间逐步演化。"
        ),
        "next_step": "下一步建议观察 reducer，例如 Annotated[list[str], operator.add]。",
        "notes": ["进入 state 节点。"],
    }


def explain_control_flow(_: LearningState) -> LearningState:
    return {
        "answer": (
            "Conditional Edge 用于把图从固定链路提升为显式分支控制。"
            "这也是 LangGraph 相对普通链式调用更重要的差异之一。"
        ),
        "next_step": "下一步建议学习 add_conditional_edges() 的返回路由机制。",
        "notes": ["进入 control_flow 节点。"],
    }


def explain_memory(_: LearningState) -> LearningState:
    return {
        "answer": (
            "Memory 在 LangGraph 里更准确地说是 checkpointing 和持久状态恢复。"
            "它不是简单聊天记录，而是运行图的可恢复执行上下文。"
        ),
        "next_step": "下一步建议学习 checkpointer 与 thread/session 的关系。",
        "notes": ["进入 memory 节点。"],
    }


def finalize(state: LearningState) -> LearningState:
    return {
        "notes": [
            "图执行结束。",
            f"建议下一步: {state['next_step']}",
        ]
    }

