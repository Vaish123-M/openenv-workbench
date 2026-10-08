from __future__ import annotations

from typing import Any, Dict, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class RunError(BaseModel):
    """Safe, structured information about a failed run."""

    error_type: str
    message: str


class RunResult(BaseModel):
    """Outcome of one agent/environment episode."""

    run_id: str = Field(default_factory=lambda: uuid4().hex)
    task_id: str | None = None
    environment: str
    completed: bool = False
    termination_reason: Literal[
        "task_completed",
        "max_steps_reached",
        "timeout",
        "cancelled",
        "agent_error",
        "environment_error",
        "invalid_action",
        "missing_observation",
        "unexpected_termination",
    ]
    steps: int = 0
    final_observation: Any = None
    final_action: Any = None
    error: RunError | None = None
    info: Dict[str, Any] = Field(default_factory=dict)
