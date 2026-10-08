from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmark.models import BenchmarkResult, ComparisonResult
from agent.models import RunResult


class ResultRepository:
    """SQLite data-access layer for benchmark and agent evaluation results."""

    def __init__(self, path: str | Path = "openenv_results.db") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._initialize()

    def _initialize(self) -> None:
        self._connection.executescript(
            """
            PRAGMA foreign_keys = ON;
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                benchmark_run_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                config_json TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS task_runs (
                run_id TEXT PRIMARY KEY,
                benchmark_run_id TEXT NOT NULL,
                environment TEXT NOT NULL,
                task_id TEXT,
                agent TEXT NOT NULL,
                model_name TEXT,
                termination_reason TEXT NOT NULL,
                created_at TEXT NOT NULL,
                result_json TEXT NOT NULL,
                record_json TEXT NOT NULL,
                FOREIGN KEY (benchmark_run_id) REFERENCES benchmark_runs(benchmark_run_id)
            );
            CREATE INDEX IF NOT EXISTS idx_task_runs_agent ON task_runs(agent);
            CREATE INDEX IF NOT EXISTS idx_task_runs_environment_task
                ON task_runs(environment, task_id);
            """
        )
        columns = {
            row["name"]
            for row in self._connection.execute("PRAGMA table_info(task_runs)").fetchall()
        }
        if "record_json" not in columns:
            self._connection.execute(
                "ALTER TABLE task_runs ADD COLUMN record_json TEXT NOT NULL DEFAULT '{}'"
            )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()

    def save_benchmark(self, benchmark: BenchmarkResult) -> str:
        return self._save(
            benchmark.benchmark_run_id,
            "benchmark",
            benchmark.config.model_dump_json(),
            benchmark.summary.model_dump_json(),
            benchmark.records,
        )

    def save_comparison(self, comparison: ComparisonResult) -> str:
        summary = {agent: value.model_dump() for agent, value in comparison.summaries.items()}
        return self._save(
            comparison.benchmark_run_id,
            "comparison",
            comparison.config.model_dump_json(),
            json.dumps(summary),
            comparison.records,
        )

    def save_run(
        self, result: RunResult, agent: str = "unknown", model_name: str | None = None
    ) -> str:
        """Persist a standalone task run without requiring a benchmark."""
        from benchmark.models import BenchmarkRecord, BenchmarkResult, BenchmarkConfig

        benchmark = BenchmarkResult(
            benchmark_run_id=result.run_id,
            config=BenchmarkConfig(
                environment=result.environment,
                task_ids=[result.task_id] if result.task_id else [],
                agent=agent,
            ),
            records=[
                BenchmarkRecord(
                    benchmark_run_id=result.run_id,
                    run_index=1,
                    environment=result.environment,
                    task_id=result.task_id or "",
                    agent=agent,
                    model_name=model_name or result.metrics.model_name,
                    result=result,
                )
            ],
        )
        return self.save_benchmark(benchmark)

    def _save(
        self,
        benchmark_run_id: str,
        kind: str,
        config_json: str,
        summary_json: str,
        records: list[Any],
    ) -> str:
        created_at = _now()
        with self._connection:
            self._connection.execute(
                """
                INSERT OR REPLACE INTO benchmark_runs
                (benchmark_run_id, kind, config_json, summary_json, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (benchmark_run_id, kind, config_json, summary_json, created_at),
            )
            self._connection.executemany(
                """
                INSERT OR REPLACE INTO task_runs
                (run_id, benchmark_run_id, environment, task_id, agent, model_name,
                 termination_reason, created_at, result_json, record_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        record.result.run_id,
                        benchmark_run_id,
                        record.environment,
                        record.task_id,
                        record.agent,
                        record.model_name,
                        record.result.termination_reason,
                        created_at,
                        record.result.model_dump_json(),
                        record.model_dump_json(),
                    )
                    for record in records
                ],
            )
        return benchmark_run_id

    def get_benchmark(self, benchmark_run_id: str) -> dict[str, Any] | None:
        row = self._connection.execute(
            "SELECT * FROM benchmark_runs WHERE benchmark_run_id = ?",
            (benchmark_run_id,),
        ).fetchone()
        if row is None:
            return None
        return self._benchmark_payload(row)

    def get_run(self, run_id: str) -> RunResult | None:
        row = self._connection.execute(
            "SELECT result_json FROM task_runs WHERE run_id = ?", (run_id,)
        ).fetchone()
        return RunResult.model_validate_json(row["result_json"]) if row else None

    def results_by_agent(self, agent: str) -> list[RunResult]:
        return self._runs(
            "SELECT result_json FROM task_runs WHERE agent = ? ORDER BY created_at",
            (agent,),
        )

    def results_by_environment_task(
        self, environment: str, task_id: str | None = None
    ) -> list[RunResult]:
        if task_id is None:
            return self._runs(
                "SELECT result_json FROM task_runs WHERE environment = ? ORDER BY created_at",
                (environment,),
            )
        return self._runs(
            """
            SELECT result_json FROM task_runs
            WHERE environment = ? AND task_id = ? ORDER BY created_at
            """,
            (environment, task_id),
        )

    def _runs(self, query: str, parameters: tuple[Any, ...]) -> list[RunResult]:
        return [
            RunResult.model_validate_json(row["result_json"])
            for row in self._connection.execute(query, parameters).fetchall()
        ]

    def _benchmark_payload(self, row: sqlite3.Row) -> dict[str, Any]:
        records = [
            json.loads(item["record_json"])
            for item in self._connection.execute(
                "SELECT * FROM task_runs WHERE benchmark_run_id = ? ORDER BY run_id",
                (row["benchmark_run_id"],),
            ).fetchall()
        ]
        return {
            "benchmark_run_id": row["benchmark_run_id"],
            "kind": row["kind"],
            "config": json.loads(row["config_json"]),
            "summary": json.loads(row["summary_json"]),
            "created_at": row["created_at"],
            "records": records,
        }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
