from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from agent.models import RunResult


class BenchmarkConfig(BaseModel):
    environment: str
    task_ids: list[str]
    agent: str | None = None
    agents: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=10, ge=1, le=1000)
    timeout: float | None = Field(default=None, gt=0)
    runs: int = Field(default=1, ge=1, le=1000)
    seed: int | None = None
    actions: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_agents(self) -> "BenchmarkConfig":
        if self.agent is not None and self.agents and self.agent not in self.agents:
            raise ValueError("agent must be included in agents when both are provided")
        if not self.agents:
            if self.agent is None:
                raise ValueError("agent or agents is required")
            self.agents = [self.agent]
        elif self.agent is None:
            self.agent = self.agents[0]
        return self

    @property
    def selected_agents(self) -> list[str]:
        return list(self.agents)


class BenchmarkRecord(BaseModel):
    benchmark_run_id: str
    run_index: int
    environment: str
    task_id: str
    agent: str
    model_name: str | None = None
    difficulty: str | None = None
    result: RunResult


class BenchmarkSummary(BaseModel):
    total_runs: int = 0
    successful_runs: int = 0
    failed_runs: int = 0
    timeout_runs: int = 0
    success_rate: float = 0.0
    average_score: float = 0.0
    average_steps: float = 0.0
    average_execution_time: float = 0.0
    failure_rate: float = 0.0
    timeout_rate: float = 0.0
    failure_categories: dict[str, int] = Field(default_factory=dict)
    repeated_action_count: int = 0
    wasted_steps: int = 0


class AgentComparisonSummary(BenchmarkSummary):
    agent: str
    model_name: str | None = None


class BenchmarkResult(BaseModel):
    benchmark_run_id: str = Field(default_factory=lambda: uuid4().hex)
    config: BenchmarkConfig
    records: list[BenchmarkRecord] = Field(default_factory=list)
    summary: BenchmarkSummary = Field(default_factory=BenchmarkSummary)


class ComparisonResult(BaseModel):
    """Results and per-agent summaries for one multi-agent benchmark."""

    benchmark_run_id: str = Field(default_factory=lambda: uuid4().hex)
    config: BenchmarkConfig
    records: list[BenchmarkRecord] = Field(default_factory=list)
    summaries: dict[str, AgentComparisonSummary] = Field(default_factory=dict)
