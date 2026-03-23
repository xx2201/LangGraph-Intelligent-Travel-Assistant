from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import MessagesState


TravelIntent = Literal["weather", "geocode", "place_search", "travel", "general"]
SpecialistAgent = Literal["weather_agent", "geo_agent", "travel_agent", "general_agent"]


class QueryContext(TypedDict, total=False):
    raw_user_input: str
    intent: TravelIntent
    location_text: str
    normalized_city: str
    time_text: str
    needs_clarification: bool
    clarification_reason: str
    suggested_tool: str
    candidate_locations: list[str]


class TaskMemory(TypedDict, total=False):
    """Structured mid-term memory distilled from recent turns."""

    current_goal: str
    latest_intent: TravelIntent
    latest_user_request: str
    latest_assistant_reply: str
    confirmed_city: str
    recent_cities: list[str]
    budget_text: str
    trip_days: str
    latest_time_text: str
    last_active_agent: SpecialistAgent
    user_preferences: list[str]


class TravelAssistantState(MessagesState, total=False):
    """Message-based state for the travel assistant agent."""

    query_context: QueryContext
    active_agent: SpecialistAgent
    agent_selection_reason: str
    conversation_summary: str
    task_memory: TaskMemory
    recalled_memories: list[str]
    memory_scope: str
