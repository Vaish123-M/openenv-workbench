from __future__ import annotations

import json

from benchmark import BenchmarkConfig, BenchmarkRunner


EMAIL_ACTION = {
    "content": '{"label": "spam", "reason": "Verify suspension card click immediately form."}',
    "format": "json",
}
CODING_ACTIONS = {
    "fix_addition": [
        {
            "content": json.dumps(
                {
                    "tool": "edit_file",
                    "path": "solution.py",
                    "content": "def add_numbers(first: int, second: int) -> int:\n    return first + second\n",
                }
            ),
            "format": "json",
        },
        {"content": '{"tool":"submit"}', "format": "json"},
    ]
}


def test_benchmark_runs_multiple_tasks_and_repeats_with_aggregation() -> None:
    benchmark = BenchmarkRunner().run(
        BenchmarkConfig(
            environment="openenv",
            task_ids=["email_classification", "email_classification"],
            agent="mock",
            runs=2,
            seed=42,
            actions={"email_classification": [EMAIL_ACTION]},
        )
    )

    assert len(benchmark.benchmark_run_id) == 32
    assert len(benchmark.records) == 4
    assert len({record.result.run_id for record in benchmark.records}) == 4
    assert benchmark.summary.total_runs == 4
    assert benchmark.summary.success_rate == 1.0
    assert benchmark.summary.average_score == 1.0
    assert benchmark.summary.average_steps == 1.0
    assert benchmark.summary.failure_rate == 0.0
    assert benchmark.summary.timeout_rate == 0.0
    assert [record.run_index for record in benchmark.records] == [1, 2, 3, 4]


def test_benchmark_runs_coding_task_through_existing_agent_runner() -> None:
    benchmark = BenchmarkRunner().run(
        BenchmarkConfig(
            environment="coding",
            task_ids=["fix_addition"],
            agent="mock",
            actions=CODING_ACTIONS,
            max_steps=4,
            seed=7,
        )
    )

    record = benchmark.records[0]
    assert record.result.completed is True
    assert record.result.grading.score == 1.0
    assert record.result.steps == 2
    assert benchmark.summary.success_rate == 1.0


def test_benchmark_aggregates_failures_and_filters_records() -> None:
    benchmark = BenchmarkRunner().run(
        BenchmarkConfig(
            environment="openenv",
            task_ids=["email_classification"],
            agent="mock",
            runs=2,
            max_steps=1,
            actions={"email_classification": [{"content": "wrong", "format": "text"}]},
        )
    )

    assert benchmark.summary.success_rate == 0.0
    assert benchmark.summary.failure_rate == 1.0
    assert benchmark.summary.average_score == 0.0
    assert len(BenchmarkRunner.filter(benchmark, environment="openenv")) == 2
    assert len(BenchmarkRunner.filter(benchmark, task="email_classification")) == 2
    assert len(BenchmarkRunner.filter(benchmark, agent="mock")) == 2
    assert len(BenchmarkRunner.filter(benchmark, difficulty="easy")) == 2
    assert BenchmarkRunner.filter(benchmark, difficulty="hard") == []


def test_benchmark_exports_json_and_csv(tmp_path) -> None:
    benchmark = BenchmarkRunner().run(
        BenchmarkConfig(
            environment="openenv",
            task_ids=["email_classification"],
            agent="mock",
            actions={"email_classification": [EMAIL_ACTION]},
        )
    )
    json_path = tmp_path / "benchmark.json"
    csv_path = tmp_path / "benchmark.csv"

    BenchmarkRunner.export_json(benchmark, json_path)
    BenchmarkRunner.export_csv(benchmark, csv_path)

    exported = json.loads(json_path.read_text(encoding="utf-8"))
    csv_lines = csv_path.read_text(encoding="utf-8").splitlines()
    assert exported["benchmark_run_id"] == benchmark.benchmark_run_id
    assert exported["summary"]["success_rate"] == 1.0
    assert csv_lines[0].startswith("benchmark_run_id,run_index,environment")
    assert len(csv_lines) == 2


def test_benchmark_rejects_unknown_registered_names() -> None:
    runner = BenchmarkRunner()
    config = BenchmarkConfig(environment="unknown", task_ids=["task"], agent="mock")
    try:
        runner.run(config)
    except ValueError as exc:
        assert "environment" in str(exc)
    else:
        raise AssertionError("unknown environment should be rejected")
