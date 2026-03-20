from __future__ import annotations

import operator
from typing import Annotated
from typing_extensions import TypedDict


class LearningState(TypedDict, total=False):
    """Shared state flowing through the learning graph."""

    topic: str
    background: str
    route: str
    notes: Annotated[list[str], operator.add]
    answer: str
    next_step: str

