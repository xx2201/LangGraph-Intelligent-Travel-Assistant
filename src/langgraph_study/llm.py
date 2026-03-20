from __future__ import annotations

import os
from functools import lru_cache
from typing import Final

from langchain_qwq import ChatQwen

from .config import DEFAULT_QWEN_MODEL
from .state import LearningState, Route


FALLBACK_ANSWERS: Final[dict[Route, str]] = {
    "overview": (
        "LangGraph 的核心不是提示词，而是显式状态机。"
        "你需要先理解状态如何在节点之间流动，再去理解 agent。"
    ),
    "state": (
        "State 是 LangGraph 的中心。每个节点读取共享状态，并返回部分更新。"
        "图执行的本质是状态在节点之间逐步演化。"
    ),
    "control_flow": (
        "Conditional Edge 用于把图从固定链路提升为显式分支控制。"
        "这也是 LangGraph 相对普通链式调用更重要的差异之一。"
    ),
    "memory": (
        "Memory 在 LangGraph 里更准确地说是 checkpointing 和持久状态恢复。"
        "它不是简单聊天记录，而是运行图的可恢复执行上下文。"
    ),
}


NEXT_STEPS: Final[dict[Route, str]] = {
    "overview": "继续学习 StateGraph、节点返回值和 compile() 的关系。",
    "state": "下一步建议观察 reducer，例如 Annotated[list[str], operator.add]。",
    "control_flow": "下一步建议学习 add_conditional_edges() 的返回路由机制。",
    "memory": "下一步建议学习 checkpointer 与 thread/session 的关系。",
}


ROUTE_LABELS: Final[dict[Route, str]] = {
    "overview": "总览与定位",
    "state": "状态设计",
    "control_flow": "条件边与控制流",
    "memory": "记忆与持久化",
}


def _coerce_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def has_qwen_api_key() -> bool:
    return bool(os.getenv("DASHSCOPE_API_KEY"))


@lru_cache(maxsize=4)
def get_qwen_model(model_name: str = DEFAULT_QWEN_MODEL) -> ChatQwen:
    return ChatQwen(
        model=model_name,
        max_retries=2,
        timeout=None,
    )


def generate_answer_for_route(state: LearningState, route: Route) -> LearningState:
    next_step = NEXT_STEPS[route]
    fallback_answer = FALLBACK_ANSWERS[route]

    if not has_qwen_api_key():
        return {
            "answer": fallback_answer,
            "next_step": next_step,
            "response_source": "fallback",
            "response_error": "",
        }

    model_name = os.getenv("QWEN_MODEL", DEFAULT_QWEN_MODEL)
    llm = get_qwen_model(model_name)

    system_prompt = (
        "你是一个严谨的 LangGraph 学习导师。"
        "你的任务是结合学习者背景，围绕给定主题给出系统、准确、便于学习的解释。"
        "回答使用中文，避免空话，不要编造未被提及的 API。"
    )
    human_prompt = (
        f"学习主题：{state['topic']}\n"
        f"学习者背景：{state.get('background', '未提供')}\n"
        f"当前路由方向：{ROUTE_LABELS[route]}\n"
        f"建议下一步：{next_step}\n"
        "请输出一段 120 到 220 字的解释，重点说明这个主题在 LangGraph 学习路径里的作用。"
        "不要使用 Markdown 标题。"
    )

    try:
        ai_msg = llm.invoke(
            [
                ("system", system_prompt),
                ("human", human_prompt),
            ]
        )
        answer = _coerce_text(ai_msg.content).strip()
        if not answer:
            answer = fallback_answer
            source = f"{model_name} (empty-response-fallback)"
            error = "模型返回空内容，已回退到本地答案。"
        else:
            source = model_name
            error = ""
    except Exception as exc:
        answer = fallback_answer
        source = f"{model_name} (error-fallback)"
        error = f"{type(exc).__name__}: {exc}"

    return {
        "answer": answer,
        "next_step": next_step,
        "response_source": source,
        "response_error": error,
    }
