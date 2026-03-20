from __future__ import annotations

from .config import DEFAULT_BACKGROUND, DEFAULT_INPUT
from .llm import generate_answer_for_route
from .state import LearningState


def normalize_input(state: LearningState) -> LearningState:
    raw_input = (state.get("input") or state.get("topic") or "").strip()
    topic = raw_input or DEFAULT_INPUT
    background = (state.get("background") or DEFAULT_BACKGROUND).strip()

    return {
        "input": raw_input or topic,
        "topic": topic,
        "background": background,
        "notes": [
            f"输入主题: {topic}",
            f"学习者背景: {background}",
        ],
    }


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
        ],
    }


def explain_overview(state: LearningState) -> LearningState:
    response = generate_answer_for_route(state, "overview")
    notes = [
        "进入 overview 节点。",
        f"回答来源: {response['response_source']}",
    ]
    if response.get("response_error"):
        notes.append(f"模型调用信息: {response['response_error']}")
    response["notes"] = notes
    return response


def explain_state(state: LearningState) -> LearningState:
    response = generate_answer_for_route(state, "state")
    notes = [
        "进入 state 节点。",
        f"回答来源: {response['response_source']}",
    ]
    if response.get("response_error"):
        notes.append(f"模型调用信息: {response['response_error']}")
    response["notes"] = notes
    return response


def explain_control_flow(state: LearningState) -> LearningState:
    response = generate_answer_for_route(state, "control_flow")
    notes = [
        "进入 control_flow 节点。",
        f"回答来源: {response['response_source']}",
    ]
    if response.get("response_error"):
        notes.append(f"模型调用信息: {response['response_error']}")
    response["notes"] = notes
    return response


def explain_memory(state: LearningState) -> LearningState:
    response = generate_answer_for_route(state, "memory")
    notes = [
        "进入 memory 节点。",
        f"回答来源: {response['response_source']}",
    ]
    if response.get("response_error"):
        notes.append(f"模型调用信息: {response['response_error']}")
    response["notes"] = notes
    return response


def finalize(state: LearningState) -> LearningState:
    return {
        "notes": [
            "图执行结束。",
            f"实际回答来源: {state.get('response_source', 'unknown')}",
            f"建议下一步: {state['next_step']}",
        ]
    }
