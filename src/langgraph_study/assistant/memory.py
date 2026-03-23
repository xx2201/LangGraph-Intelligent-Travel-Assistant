from __future__ import annotations
import re
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, RemoveMessage, SystemMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.types import RunnableConfig

from ..core.config import (
    MEMORY_MAX_SUMMARY_CHARS,
    MEMORY_SHORT_TERM_WINDOW,
    MEMORY_SUMMARY_TRIGGER_MESSAGES,
)
from ..integrations.memory import LongTermMemoryManager, NoOpLongTermMemoryStore
from .state import QueryContext, TaskMemory, TravelAssistantState


BUDGET_PATTERN = re.compile(r"(预算[^\n，。,.!?；;]{0,16})")
TRIP_DAYS_PATTERN = re.compile(r"([0-9一二三四五六七八九十两]+天)")
PREFERENCE_PATTERN = re.compile(r"(喜欢[^，。.!?\n]{1,18}|偏好[^，。.!?\n]{1,18}|尽量不要[^，。.!?\n]{1,18}|不要[^，。.!?\n]{1,18})")


def build_noop_memory_manager() -> LongTermMemoryManager:
    return LongTermMemoryManager(store=NoOpLongTermMemoryStore(), top_k=0)


def ensure_unique_items(values: list[str], limit: int = 6) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = " ".join(str(value).split())
        if not cleaned or cleaned in seen:
            continue
        ordered.append(cleaned)
        seen.add(cleaned)
    if len(ordered) <= limit:
        return ordered
    return ordered[-limit:]


def extract_latest_assistant_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, AIMessage) and isinstance(message.content, str) and message.content.strip():
            return message.content.strip()
    return ""


def summarize_message(message: BaseMessage) -> str:
    if isinstance(message, HumanMessage):
        role = "User"
    elif isinstance(message, AIMessage):
        role = "Assistant"
    else:
        role = type(message).__name__.removesuffix("Message") or type(message).__name__
    content = getattr(message, "content", "")
    text = content if isinstance(content, str) else str(content)
    compact = " ".join(text.split())
    if len(compact) > 120:
        compact = f"{compact[:117]}..."
    return f"{role}: {compact}"


def summarize_archived_messages(
    existing_summary: str,
    archived_messages: list[BaseMessage],
    max_chars: int = MEMORY_MAX_SUMMARY_CHARS,
) -> str:
    archived_lines = [summarize_message(message) for message in archived_messages if summarize_message(message)]
    if not archived_lines:
        return existing_summary
    sections: list[str] = []
    if existing_summary.strip():
        sections.append(existing_summary.strip())
    sections.append("Archived history: " + " | ".join(archived_lines))
    combined = "\n".join(sections)
    if len(combined) <= max_chars:
        return combined
    return combined[-max_chars:]


def extract_budget_text(text: str) -> str:
    match = BUDGET_PATTERN.search(text)
    return match.group(1).strip() if match else ""


def extract_trip_days(text: str) -> str:
    match = TRIP_DAYS_PATTERN.search(text)
    return match.group(1).strip() if match else ""


def extract_preferences(text: str) -> list[str]:
    return [match.strip() for match in PREFERENCE_PATTERN.findall(text)]


def update_task_memory(state: TravelAssistantState) -> TaskMemory:
    task_memory: TaskMemory = dict(state.get("task_memory", {}))
    query_context: QueryContext = state.get("query_context", {})
    messages = state.get("messages", [])
    latest_user_request = query_context.get("raw_user_input", "")
    latest_assistant_reply = extract_latest_assistant_text(messages)

    if latest_user_request:
        task_memory["latest_user_request"] = latest_user_request
        task_memory["current_goal"] = latest_user_request
    if latest_assistant_reply:
        task_memory["latest_assistant_reply"] = latest_assistant_reply
    if query_context.get("intent"):
        task_memory["latest_intent"] = query_context["intent"]
    if query_context.get("time_text"):
        task_memory["latest_time_text"] = query_context["time_text"]
    if state.get("active_agent"):
        task_memory["last_active_agent"] = state["active_agent"]

    city = query_context.get("normalized_city") or query_context.get("location_text", "")
    if city:
        task_memory["confirmed_city"] = city
        recent_cities = list(task_memory.get("recent_cities", []))
        recent_cities.append(city)
        task_memory["recent_cities"] = ensure_unique_items(recent_cities)

    budget_text = extract_budget_text(latest_user_request)
    if budget_text:
        task_memory["budget_text"] = budget_text

    trip_days = extract_trip_days(latest_user_request)
    if trip_days:
        task_memory["trip_days"] = trip_days

    preferences = list(task_memory.get("user_preferences", []))
    preferences.extend(extract_preferences(latest_user_request))
    task_memory["user_preferences"] = ensure_unique_items(preferences)
    return task_memory


def format_task_memory(task_memory: TaskMemory) -> str:
    if not task_memory:
        return ""
    lines = ["Current structured task memory:"]
    for key, label in (
        ("current_goal", "Current goal"),
        ("latest_intent", "Latest intent"),
        ("confirmed_city", "Confirmed city"),
        ("recent_cities", "Recent cities"),
        ("budget_text", "Budget"),
        ("trip_days", "Trip length"),
        ("latest_time_text", "Time hint"),
        ("user_preferences", "Preferences"),
    ):
        value = task_memory.get(key)
        if not value:
            continue
        if isinstance(value, list):
            rendered = ", ".join(str(item) for item in value)
        else:
            rendered = str(value)
        lines.append(f"- {label}: {rendered}")
    return "\n".join(lines)


def build_memory_context_messages(state: TravelAssistantState) -> list[SystemMessage]:
    messages: list[SystemMessage] = []
    conversation_summary = state.get("conversation_summary", "").strip()
    if conversation_summary:
        messages.append(
            SystemMessage(content=f"Conversation summary from earlier turns:\n{conversation_summary}")
        )
    task_memory = format_task_memory(state.get("task_memory", {}))
    if task_memory:
        messages.append(SystemMessage(content=task_memory))
    recalled_memories = state.get("recalled_memories", [])
    if recalled_memories:
        recall_lines = "\n".join(f"- {item}" for item in recalled_memories)
        messages.append(SystemMessage(content=f"Relevant long-term memories:\n{recall_lines}"))
    return messages


def resolve_memory_scope(
    state: TravelAssistantState,
    config: RunnableConfig | None,
) -> str:
    if state.get("memory_scope"):
        return state["memory_scope"]
    configurable = (config or {}).get("configurable", {})
    return str(configurable.get("memory_scope") or configurable.get("thread_id") or "")


def build_recall_query(state: TravelAssistantState) -> str:
    task_memory = state.get("task_memory", {})
    query_context = state.get("query_context", {})
    fragments = [
        query_context.get("raw_user_input", ""),
        task_memory.get("current_goal", ""),
        " ".join(task_memory.get("user_preferences", [])),
    ]
    return " ".join(fragment for fragment in fragments if fragment).strip()


def route_after_specialist_response(
    state: TravelAssistantState,
) -> Literal["tools", "finalize_memory"]:
    messages = state.get("messages", [])
    if messages:
        latest = messages[-1]
        if isinstance(latest, AIMessage) and getattr(latest, "tool_calls", None):
            return "tools"
    return "finalize_memory"


def create_recall_memory_node(memory_manager: LongTermMemoryManager):
    def recall_sync(state: TravelAssistantState, config: RunnableConfig | None = None):
        scope = resolve_memory_scope(state, config)
        query = build_recall_query(state)
        recalled = memory_manager.recall(scope=scope, query=query) if scope else []
        return {
            "memory_scope": scope,
            "recalled_memories": recalled,
        }

    async def recall_async(state: TravelAssistantState, config: RunnableConfig | None = None):
        scope = resolve_memory_scope(state, config)
        query = build_recall_query(state)
        recalled = await memory_manager.arecall(scope=scope, query=query) if scope else []
        return {
            "memory_scope": scope,
            "recalled_memories": recalled,
        }

    from langchain_core.runnables import RunnableLambda

    return RunnableLambda(recall_sync, afunc=recall_async, name="recall_memory")


def create_finalize_memory_node(memory_manager: LongTermMemoryManager):
    def finalize_sync(state: TravelAssistantState, config: RunnableConfig | None = None):
        return _finalize_state_sync(state, config, memory_manager)

    async def finalize_async(state: TravelAssistantState, config: RunnableConfig | None = None):
        return await _finalize_state(state, config, memory_manager)

    from langchain_core.runnables import RunnableLambda

    return RunnableLambda(finalize_sync, afunc=finalize_async, name="finalize_memory")


async def _finalize_state(
    state: TravelAssistantState,
    config: RunnableConfig | None,
    memory_manager: LongTermMemoryManager,
) -> dict[str, Any]:
    messages = list(state.get("messages", []))
    updated_task_memory = update_task_memory(state)
    conversation_summary = state.get("conversation_summary", "")
    if len(messages) > MEMORY_SUMMARY_TRIGGER_MESSAGES:
        keep_count = max(MEMORY_SHORT_TERM_WINDOW, 1)
        archive_count = len(messages) - keep_count
        archived_messages = messages[:archive_count]
        kept_messages = messages[archive_count:]
        conversation_summary = summarize_archived_messages(conversation_summary, archived_messages)
        message_update: list[Any] = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept_messages]
    else:
        message_update = []

    scope = resolve_memory_scope(state, config)
    if scope:
        await memory_manager.aremember(
            scope=scope,
            conversation_summary=conversation_summary,
            task_memory=updated_task_memory,
        )

    result: dict[str, Any] = {
        "task_memory": updated_task_memory,
        "conversation_summary": conversation_summary,
        "memory_scope": scope,
    }
    if message_update:
        result["messages"] = message_update
    return result


def _finalize_state_sync(
    state: TravelAssistantState,
    config: RunnableConfig | None,
    memory_manager: LongTermMemoryManager,
) -> dict[str, Any]:
    messages = list(state.get("messages", []))
    updated_task_memory = update_task_memory(state)
    conversation_summary = state.get("conversation_summary", "")
    if len(messages) > MEMORY_SUMMARY_TRIGGER_MESSAGES:
        keep_count = max(MEMORY_SHORT_TERM_WINDOW, 1)
        archive_count = len(messages) - keep_count
        archived_messages = messages[:archive_count]
        kept_messages = messages[archive_count:]
        conversation_summary = summarize_archived_messages(conversation_summary, archived_messages)
        message_update: list[Any] = [RemoveMessage(id=REMOVE_ALL_MESSAGES), *kept_messages]
    else:
        message_update = []

    scope = resolve_memory_scope(state, config)
    if scope:
        memory_manager.remember(
            scope=scope,
            conversation_summary=conversation_summary,
            task_memory=updated_task_memory,
        )

    result: dict[str, Any] = {
        "task_memory": updated_task_memory,
        "conversation_summary": conversation_summary,
        "memory_scope": scope,
    }
    if message_update:
        result["messages"] = message_update
    return result
