from __future__ import annotations

import os
from functools import lru_cache

from langchain_community.chat_models.tongyi import ChatTongyi

from .config import DEFAULT_QWEN_MODEL


def has_qwen_api_key() -> bool:
    return bool(os.getenv("DASHSCOPE_API_KEY"))


@lru_cache(maxsize=4)
def get_qwen_model(model_name: str = DEFAULT_QWEN_MODEL) -> ChatTongyi:
    if not has_qwen_api_key():
        raise RuntimeError("未检测到 DASHSCOPE_API_KEY，无法创建 qwen3-max 模型。")
    return ChatTongyi(model=model_name)
