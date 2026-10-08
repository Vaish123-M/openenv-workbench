from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from agent.models import RunResult


class BenchmarkConfig(BaseModel):
    environment: str
    task_ids: list[str]
    agent: str
    max_steps: int = Field(default=10, ge=1, le=1000)
    timeout: float | None = Field(default=None, gt=0)
    runs: int = Field(default=1, ge=1, le=1000)
    seed: int | None = None
    actions: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)


class BenchmarkRecord(BaseModel):
    benchmark_run_id: str
    run_index: int
    environment: str
    task_id: str
    agent: str
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


class BenchmarkResult(BaseModel):
    benchmark_run_id: str = Field(default_factory=lambda: uuid4().hex)
    config: BenchmarkConfig
    records: list[BenchmarkRecord] = Field(default_factory=list)
    summary: BenchmarkSummary = Field(default_factory=BenchmarkSummary)

