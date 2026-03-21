from __future__ import annotations

from langchain_core.messages import SystemMessage

from .config import TRAVEL_AGENT_SYSTEM_PROMPT
from .state import TravelAssistantState


def create_assistant_node(bound_model):
    def assistant(state: TravelAssistantState):
        response = bound_model.invoke(
            [
                SystemMessage(content=TRAVEL_AGENT_SYSTEM_PROMPT),
                *state["messages"],
            ]
        )
        return {"messages": [response]}

    return assistant
