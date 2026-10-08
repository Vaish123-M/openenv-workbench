from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable
from uuid import uuid4

from agent import AgentRunner, LLMBackedAgent, MockAgent, OpenAIProvider
from agent.interface import Agent
from agent.models import RunError, RunResult
from coding import CodingEnvironment, get_coding_task
from environment.core import OpenEnv
from tasks import get_task

from .models import (
    BenchmarkConfig,
    BenchmarkRecord,
    BenchmarkResult,
    BenchmarkSummary,
    AgentComparisonSummary,
    ComparisonResult,
)

EnvironmentFactory = Callable[[], Any]
AgentFactory = Callable[[str, int | None], Agent]


def _default_environment_factories() -> dict[str, EnvironmentFactory]:
    return {"openenv": OpenEnv, "coding": CodingEnvironment}


def _default_agent_factories(config: BenchmarkConfig) -> dict[str, AgentFactory]:
    return {
        "mock": lambda task_id, seed: MockAgent(config.actions.get(task_id, [])),
        "openai": lambda task_id, seed: LLMBackedAgent(OpenAIProvider()),
    }


def _difficulty(environment: str, task_id: str) -> str | None:
    if environment == "openenv":
        return get_task(task_id).difficulty
    if environment == "coding":
        return get_coding_task(task_id).difficulty
    return None


class BenchmarkRunner:
    """Run named benchmark configurations through the existing AgentRunner."""

    def __init__(
        self,
        environment_factories: dict[str, EnvironmentFactory] | None = None,
        agent_factories: dict[str, AgentFactory] | None = None,
    ) -> None:
        self.environment_factories = environment_factories or _default_environment_factories()
        self._agent_factories = agent_factories

    def run(self, config: BenchmarkConfig) -> BenchmarkResult:
        if len(config.selected_agents) != 1:
            raise ValueError("run() requires exactly one configured agent; use run_comparison()")
        return self._run_for_agents(config, config.selected_agents)[0]

    def run_comparison(self, config: BenchmarkConfig) -> ComparisonResult:
        """Run identical benchmark cases for every configured agent."""
        agents = config.selected_agents
        if len(agents) < 2:
            raise ValueError("run_comparison() requires at least two configured agents")
        results = self._run_for_agents(config, agents)
        benchmark_run_id = uuid4().hex
        records = [
            record.model_copy(update={"benchmark_run_id": benchmark_run_id})
            for result in results
            for record in result.records
        ]
        return ComparisonResult(
            benchmark_run_id=benchmark_run_id,
            config=config,
            records=records,
            summaries={
                agent: AgentComparisonSummary(
                    agent=agent,
                    model_name=self._model_name(
                        [record for record in records if record.agent == agent]
                    ),
                    **self._summarize(
                        [record for record in records if record.agent == agent]
                    ).model_dump(),
                )
                for agent in agents
            },
        )

    def _run_for_agents(
        self, config: BenchmarkConfig, agents: list[str]
    ) -> list[BenchmarkResult]:
        environment_factory = self.environment_factories.get(config.environment)
        if environment_factory is None:
            raise ValueError(f"Unknown registered environment: {config.environment}")
        agent_factories = self._agent_factories or _default_agent_factories(config)
        unknown = [agent for agent in agents if agent not in agent_factories]
        if unknown:
            raise ValueError(f"Unknown registered agent: {unknown[0]}")

        benchmarks: list[BenchmarkResult] = []
        for agent_name in agents:
            benchmark = BenchmarkResult(config=config)
            run_index = 0
            agent_factory = agent_factories[agent_name]
            for task_id in config.task_ids:
                difficulty = _difficulty(config.environment, task_id)
                for _ in range(config.runs):
                    run_index += 1
                    environment = environment_factory()
                    seed = None if config.seed is None else config.seed + run_index - 1
                    try:
                        agent = agent_factory(task_id, seed)
                        result = AgentRunner().run(
                            environment=environment,
                            agent=agent,
                            task_id=task_id,
                            max_steps=config.max_steps,
                            timeout=config.timeout,
                        )
                    except Exception as exc:
                        result = RunResult(
                            task_id=task_id,
                            environment=config.environment,
                            termination_reason="agent_error",
                            error=RunError(
                                error_type=type(exc).__name__,
                                message="Agent setup failed",
                            ),
                            metrics={
                                "agent_errors": [
                                    {
                                        "error_type": type(exc).__name__,
                                        "message": "Agent setup failed",
                                    }
                                ]
                            },
                        )
                    benchmark.records.append(
                        BenchmarkRecord(
                            benchmark_run_id=benchmark.benchmark_run_id,
                            run_index=run_index,
                            environment=config.environment,
                            task_id=task_id,
                            agent=agent_name,
                            model_name=result.metrics.model_name,
                            difficulty=difficulty,
                            result=result,
                        )
                    )
                    close = getattr(environment, "close", None)
                    if close is not None:
                        close()
            benchmark.summary = self._summarize(benchmark.records)
            benchmarks.append(benchmark)
        return benchmarks

    @staticmethod
    def _summarize(records: list[BenchmarkRecord]) -> BenchmarkSummary:
        total = len(records)
        successful = sum(record.result.completed for record in records)
        timeouts = sum(record.result.termination_reason == "timeout" for record in records)
        scores = [record.result.grading.score or 0.0 for record in records]
        steps = [record.result.steps for record in records]
        elapsed = [record.result.metrics.elapsed_time for record in records]
        failed = total - successful
        return BenchmarkSummary(
            total_runs=total,
            successful_runs=successful,
            failed_runs=failed,
            timeout_runs=timeouts,
            success_rate=successful / total if total else 0.0,
            average_score=sum(scores) / total if total else 0.0,
            average_steps=sum(steps) / total if total else 0.0,
            average_execution_time=sum(elapsed) / total if total else 0.0,
            failure_rate=failed / total if total else 0.0,
            timeout_rate=timeouts / total if total else 0.0,
        )

    @staticmethod
    def _model_name(records: list[BenchmarkRecord]) -> str | None:
        names = {record.model_name for record in records if record.model_name}
        return next(iter(names)) if len(names) == 1 else None

    @staticmethod
    def filter(
        benchmark: BenchmarkResult,
        environment: str | None = None,
        task: str | None = None,
        agent: str | None = None,
        difficulty: str | None = None,
    ) -> list[BenchmarkRecord]:
        return [
            record
            for record in benchmark.records
            if (environment is None or record.environment == environment)
            and (task is None or record.task_id == task)
            and (agent is None or record.agent == agent)
            and (difficulty is None or record.difficulty == difficulty)
        ]

    @staticmethod
    def export_json(benchmark: BenchmarkResult, path: str | Path) -> None:
        Path(path).write_text(benchmark.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def export_csv(benchmark: BenchmarkResult, path: str | Path) -> None:
        fields = [
            "benchmark_run_id", "run_index", "environment", "task_id", "agent",
            "model_name",
            "difficulty", "completed", "termination_reason", "steps", "score",
            "elapsed_time", "timed_out",
        ]
        with Path(path).open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for record in benchmark.records:
                writer.writerow(
                    {
                        "benchmark_run_id": record.benchmark_run_id,
                        "run_index": record.run_index,
                        "environment": record.environment,
                        "task_id": record.task_id,
                        "agent": record.agent,
                        "model_name": record.model_name or "",
                        "difficulty": record.difficulty or "",
                        "completed": record.result.completed,
                        "termination_reason": record.result.termination_reason,
                        "steps": record.result.steps,
                        "score": record.result.grading.score,
                        "elapsed_time": record.result.metrics.elapsed_time,
                        "timed_out": record.result.metrics.timed_out,
                    }
                )

    @staticmethod
    def export_comparison_json(comparison: ComparisonResult, path: str | Path) -> None:
        Path(path).write_text(comparison.model_dump_json(indent=2), encoding="utf-8")

    @staticmethod
    def export_comparison_csv(comparison: ComparisonResult, path: str | Path) -> None:
        fields = [
            "benchmark_run_id",
            "run_index",
            "environment",
            "task_id",
            "agent",
            "model_name",
            "difficulty",
            "completed",
            "termination_reason",
            "steps",
            "score",
            "elapsed_time",
            "timed_out",
        ]
        with Path(path).open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for record in comparison.records:
                writer.writerow(
                    {
                        "benchmark_run_id": record.benchmark_run_id,
                        "run_index": record.run_index,
                        "environment": record.environment,
                        "task_id": record.task_id,
                        "agent": record.agent,
                        "model_name": record.model_name or "",
                        "difficulty": record.difficulty or "",
                        "completed": record.result.completed,
                        "termination_reason": record.result.termination_reason,
                        "steps": record.result.steps,
                        "score": record.result.grading.score,
                        "elapsed_time": record.result.metrics.elapsed_time,
                        "timed_out": record.result.metrics.timed_out,
                    }
                )
