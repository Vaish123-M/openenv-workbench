from __future__ import annotations

from typing import Any, Dict, Literal
from time import monotonic
from uuid import uuid4

from pydantic import BaseModel, Field


class RunError(BaseModel):
    """Safe, structured information about a failed run."""

    error_type: str
    message: str


FailureCategory = Literal[
    "invalid_action",
    "agent_error",
    "environment_error",
    "constraint_violation",
    "incorrect_output",
    "incomplete_task",
    "timeout",
    "repeated_failed_action",
]


class TrajectoryStep(BaseModel):
    """One attempted agent/environment interaction."""

    step_number: int
    observation: Any = None
    action: Any = None
    environment_response: Any = None
    timestamp: float
    duration: float = 0.0
    error: RunError | None = None


class FailureSummary(BaseModel):
    primary_reason: FailureCategory | None = None
    categories: list[FailureCategory] = Field(default_factory=list)
    error_count: int = 0
    repeated_action_count: int = 0
    wasted_steps: int = 0


class ExecutionMetrics(BaseModel):
    total_steps: int = 0
    failed_steps: int = 0
    elapsed_time: float = 0.0
    invalid_action_count: int = 0
    agent_errors: list[RunError] = Field(default_factory=list)
    environment_errors: list[RunError] = Field(default_factory=list)
    timed_out: bool = False
    model_name: str | None = None
    token_usage: dict[str, int] | None = None
    repeated_action_count: int = 0
    wasted_steps: int = 0


class GradingSummary(BaseModel):
    score: float | None = None
    reward: float | None = None
    penalty: float | None = None
    breakdown: dict[str, float] = Field(default_factory=dict)


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
    metrics: ExecutionMetrics = Field(default_factory=ExecutionMetrics)
    grading: GradingSummary = Field(default_factory=GradingSummary)
    trajectory: list[TrajectoryStep] = Field(default_factory=list)
    failure: FailureSummary = Field(default_factory=FailureSummary)
    _started_at: float = monotonic()
