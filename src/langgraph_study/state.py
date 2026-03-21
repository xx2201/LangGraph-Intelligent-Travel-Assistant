from __future__ import annotations

from typing import Literal, TypedDict

from langgraph.graph import MessagesState


TravelIntent = Literal["weather", "geocode", "place_search", "travel", "general"]


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


class TravelAssistantState(MessagesState, total=False):
    """Message-based state for the travel assistant agent."""

    query_context: QueryContext
