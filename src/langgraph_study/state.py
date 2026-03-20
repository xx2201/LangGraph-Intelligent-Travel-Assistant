from __future__ import annotations

import operator
from typing import Annotated, Literal, TypeAlias
from typing_extensions import TypedDict


Route: TypeAlias = Literal["overview", "state", "control_flow", "memory"]


class LearningState(TypedDict, total=False):
    """Shared state flowing through the learning graph."""

    input: str
    topic: str
    background: str
    route: Route
    notes: Annotated[list[str], operator.add]
    answer: str
    next_step: str
    response_source: str
    response_error: str
