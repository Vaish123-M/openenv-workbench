from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field


class Observation(BaseModel):
    task_name: str
    difficulty: str
    objective: str
    task_input: Dict[str, Any]
    submission_format: str
    step: int
    max_steps: int
    remaining_steps: int
    done: bool
    latest_score: float = 0.0
    history: List[Dict[str, Any]] = Field(default_factory=list)


class Action(BaseModel):
    content: str = Field(..., min_length=1)
    format: Literal["text", "json"] = "text"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class Reward(BaseModel):
    value: float
    score: float
    penalty: float = 0.0
    breakdown: Dict[str, float] = Field(default_factory=dict)
