from __future__ import annotations

import re
from typing import Literal

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage, message_chunk_to_message
from langchain_core.runnables import RunnableLambda

from ..core.config import TRAVEL_AGENT_SYSTEM_PROMPT
from .state import QueryContext, TravelAssistantState

WEATHER_KEYWORDS = ("天气", "气温", "温度", "下雨", "下雪", "冷不冷", "热不热", "适合出门")
GEOCODE_KEYWORDS = ("坐标", "经纬度", "定位", "地址", "地理编码")
PLACE_SEARCH_KEYWORDS = ("附近", "景点", "餐厅", "酒店", "机场", "地铁站", "poi")
TRAVEL_KEYWORDS = ("出行", "旅游", "旅行", "行程", "周末去哪", "怎么玩")
TIME_PATTERNS = ("今天", "明天", "后天", "这周末", "周末", "本周", "下周")
CITY_ALIASES = {
    "北京": "北京",
    "北京市": "北京",
    "上海": "上海",
    "上海市": "上海",
    "杭州": "杭州",
    "杭州市": "杭州",
    "广州": "广州",
    "广州市": "广州",
    "深圳": "深圳",
    "深圳市": "深圳",
    "南京": "南京",
    "南京市": "南京",
    "苏州": "苏州",
    "苏州市": "苏州",
    "成都": "成都",
    "成都市": "成都",
    "重庆": "重庆",
    "重庆市": "重庆",
    "武汉": "武汉",
    "武汉市": "武汉",
    "西安": "西安",
    "西安市": "西安",
    "天津": "天津",
    "天津市": "天津",
    "长沙": "长沙",
    "长沙市": "长沙",
    "青岛": "青岛",
    "青岛市": "青岛",
    "厦门": "厦门",
    "厦门市": "厦门",
}
POI_HINTS = (
    "区",
    "路",
    "街",
    "广场",
    "大厦",
    "公园",
    "景区",
    "车站",
    "站",
    "机场",
    "酒店",
    "商场",
    "大学",
    "医院",
    "小区",
    "村",
    "镇",
    "乡",
    "山",
    "湖",
    "寺",
    "桥",
    "塔",
    "园区",
    "SOHO",
    "soho",
)


def create_assistant_node(bound_model):
    """Create the graph node that calls the LLM with the current state."""

    def build_messages(state: TravelAssistantState):
        context_message = build_query_context_message(state.get("query_context", {}))
        return [
            SystemMessage(content=TRAVEL_AGENT_SYSTEM_PROMPT),
            SystemMessage(content=context_message),
            *state["messages"],
        ]

    def assistant(state: TravelAssistantState):
        """Return one AI message produced from prompts, context, and history."""

        response = run_bound_model_sync(bound_model, build_messages(state))
        return {"messages": [response]}

    async def assistant_async(state: TravelAssistantState):
        """Async version of the assistant node that preserves model stream events."""

        response = await run_bound_model(bound_model, build_messages(state))
        return {"messages": [response]}

    return RunnableLambda(assistant, afunc=assistant_async, name="assistant")


async def run_bound_model(bound_model, messages) -> AIMessage:
    """Run the bound chat model with streaming support when available.

    The node first tries to consume ``astream()`` so LangGraph can expose model stream
    events to outer callers such as the web frontend. If streaming is not available,
    it falls back to ``ainvoke()`` and then to ``invoke()``.
    """

    if hasattr(bound_model, "astream"):
        accumulated_chunk = None
        async for chunk in bound_model.astream(messages):
            if accumulated_chunk is None:
                accumulated_chunk = chunk
            else:
                accumulated_chunk = accumulated_chunk + chunk
        if accumulated_chunk is not None:
            final_message = message_chunk_to_message(accumulated_chunk)
            if isinstance(final_message, AIMessage):
                return final_message

    if hasattr(bound_model, "ainvoke"):
        response = await bound_model.ainvoke(messages)
        if isinstance(response, AIMessage):
            return response

    return bound_model.invoke(messages)


def run_bound_model_sync(bound_model, messages) -> AIMessage:
    """Synchronous companion to ``run_bound_model`` for ``graph.invoke()`` callers."""

    if hasattr(bound_model, "stream"):
        accumulated_chunk = None
        for chunk in bound_model.stream(messages):
            if accumulated_chunk is None:
                accumulated_chunk = chunk
            else:
                accumulated_chunk = accumulated_chunk + chunk
        if accumulated_chunk is not None:
            final_message = message_chunk_to_message(accumulated_chunk)
            if isinstance(final_message, AIMessage):
                return final_message

    return bound_model.invoke(messages)


def analyze_query(state: TravelAssistantState) -> dict[str, QueryContext]:
    """Convert the latest user message into structured query context."""

    latest_user_text = extract_latest_user_text(state)
    context = build_query_context(latest_user_text)
    return {"query_context": context}


def route_after_analysis(state: TravelAssistantState) -> Literal["clarify", "assistant"]:
    """Choose the next node after ``analyze_query``.

    This function is a pure router. It reads the ``query_context`` that was already
    written into the state and then returns the label of the next node.
    """

    query_context = state.get("query_context", {})
    if query_context.get("needs_clarification"):
        return "clarify"
    return "assistant"


def clarify_query(state: TravelAssistantState):
    """Generate a follow-up question when the original request is ambiguous."""

    query_context = state.get("query_context", {})
    location_text = query_context.get("location_text", "")
    reason = query_context.get("clarification_reason", "")
    intent = query_context.get("intent", "general")

    if intent == "weather" and not location_text:
        content = (
            "你想查询哪个城市或地点的天气？例如“北京”“上海迪士尼”或“西湖景区”。"
        )
    elif intent == "weather":
        content = (
            f"你提到的是“{location_text}”，这更像具体地点而不是标准城市名。"
            "请补充城市，或直接给出更完整地点，例如“北京市朝阳区望京SOHO天气怎么样？”。"
        )
    elif intent == "geocode":
        content = "请给我更完整的地址信息，我才能继续做坐标解析。"
    else:
        content = "请再具体一点描述你的地点或旅行需求，我再继续帮你处理。"

    if reason:
        content = f"{content}\n当前判断依据：{reason}"

    return {"messages": [AIMessage(content=content)]}


def extract_latest_user_text(state: TravelAssistantState) -> str:
    """Read the newest human message from the message history."""

    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else str(message.content)
    return ""


def build_query_context(user_text: str) -> QueryContext:
    """Assemble a structured view of the user's latest request."""

    text = user_text.strip()
    intent = detect_intent(text)
    time_text = detect_time_text(text)
    location_text = extract_location_text(text, intent)
    normalized_city = normalize_city(location_text)
    clarification_required, clarification_reason = assess_clarification_need(
        intent=intent,
        location_text=location_text,
        normalized_city=normalized_city,
        text=text,
    )

    context: QueryContext = {
        "raw_user_input": text,
        "intent": intent,
        "time_text": time_text,
        "needs_clarification": clarification_required,
        "clarification_reason": clarification_reason,
    }
    if location_text:
        context["location_text"] = location_text
    if normalized_city:
        context["normalized_city"] = normalized_city
    if clarification_required and location_text:
        context["suggested_tool"] = "input_tips"
    return context


def build_query_context_message(query_context: QueryContext) -> str:
    """Turn structured context into a plain-text prompt block for the model."""

    lines = [
        "你会收到一个前置解析层生成的查询上下文。",
        "如果上下文里已有标准城市名，天气查询优先使用该城市作为工具参数。",
        "如果上下文提示地点不清晰，不要自行猜测城市。",
    ]
    raw_user_input = query_context.get("raw_user_input")
    if raw_user_input:
        lines.append(f"最新用户问题：{raw_user_input}")
    intent = query_context.get("intent")
    if intent:
        lines.append(f"解析意图：{intent}")
    location_text = query_context.get("location_text")
    if location_text:
        lines.append(f"原始地点片段：{location_text}")
    normalized_city = query_context.get("normalized_city")
    if normalized_city:
        lines.append(f"标准城市名：{normalized_city}")
    time_text = query_context.get("time_text")
    if time_text:
        lines.append(f"时间线索：{time_text}")
    if query_context.get("needs_clarification"):
        lines.append(f"澄清原因：{query_context.get('clarification_reason', '')}")
    return "\n".join(lines)


def detect_intent(text: str):
    """Guess the user's high-level intent from keyword rules."""

    if any(keyword in text for keyword in WEATHER_KEYWORDS):
        return "weather"
    if any(keyword in text for keyword in GEOCODE_KEYWORDS):
        return "geocode"
    if any(keyword in text for keyword in PLACE_SEARCH_KEYWORDS):
        return "place_search"
    if any(keyword in text for keyword in TRAVEL_KEYWORDS):
        return "travel"
    return "general"


def detect_time_text(text: str) -> str:
    """Extract a simple time hint from the user text."""

    for token in TIME_PATTERNS:
        if token in text:
            return token
    return ""


def extract_location_text(text: str, intent: str) -> str:
    """Extract the location fragment that matches the detected intent."""

    if intent == "weather":
        patterns = [
            r"(?:帮我看看|帮我查查|帮我查下|查一下|查下|看看|看下)?(?P<location>[\u4e00-\u9fa5A-Za-z0-9·]{2,20}?)(?:今天天气|明天天气|后天天气|这周末天气|天气|气温|温度)",
            r"(?P<location>[\u4e00-\u9fa5A-Za-z0-9·]{2,20}?)(?:适合出门吗|适合旅游吗)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return clean_location_text(match.group("location"))
    if intent == "geocode":
        patterns = [
            r"把(?P<location>.+?)(?:转成坐标|转换成坐标|转成经纬度|解析成坐标)",
            r"(?P<location>.+?)(?:的坐标|的经纬度)",
        ]
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return clean_location_text(match.group("location"))
    return ""


def clean_location_text(location_text: str) -> str:
    """Remove helper phrases and punctuation from a raw location string."""
    text = location_text.strip("，。！？,.!? ")
    prefixes = ("帮我看看", "帮我查查", "帮我查下", "查一下", "查下", "看看", "看下")
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text.removeprefix(prefix)
    if text in TIME_PATTERNS:
        return ""
    return text.strip()


def normalize_city(location_text: str) -> str:
    """Normalize known city aliases into a standard city name."""
    if not location_text:
        return ""
    return CITY_ALIASES.get(location_text, "")


def assess_clarification_need(
    intent: str,
    location_text: str,
    normalized_city: str,
    text: str,
) -> tuple[bool, str]:
    """Decide whether the graph should ask the user for more details."""

    if intent == "weather":
        if not location_text:
            return True, "天气问题缺少明确地点。"
        if normalized_city:
            return False, ""
        if any(hint in location_text for hint in POI_HINTS):
            return True, "检测到地点更像区县、商圈或 POI，天气接口通常需要标准城市名或更完整地点。"
        if len(location_text) > 6 and "市" not in location_text:
            return True, "地点过长但未识别为标准城市名，存在地点歧义。"
    if intent == "geocode" and not location_text:
        return True, "坐标解析缺少可用地址文本。"
    if intent == "general" and "天气" in text and not location_text:
        return True, "问题提到天气，但未提取到地点。"
    return False, ""
