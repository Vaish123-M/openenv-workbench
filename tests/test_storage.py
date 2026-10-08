from __future__ import annotations

from agent import AgentRunner, MockAgent
from benchmark import BenchmarkConfig, BenchmarkRunner
from environment.core import OpenEnv
from storage import ResultRepository


SUCCESSFUL_EMAIL = {
    "content": '{"label": "spam", "reason": "Verify suspension card click immediately form."}',
    "format": "json",
}


def test_repository_saves_and_retrieves_benchmark_and_runs(tmp_path) -> None:
    repository = ResultRepository(tmp_path / "results.db")
    benchmark = BenchmarkRunner(repository=repository).run(
        BenchmarkConfig(
            environment="openenv",
            task_ids=["email_classification"],
            agent="mock",
            actions={"email_classification": [SUCCESSFUL_EMAIL]},
        )
    )

    stored_benchmark = repository.get_benchmark(benchmark.benchmark_run_id)
    stored_run = repository.get_run(benchmark.records[0].result.run_id)

    assert stored_benchmark is not None
    assert stored_benchmark["benchmark_run_id"] == benchmark.benchmark_run_id
    assert stored_benchmark["records"][0]["result"]["trajectory"]
    assert stored_run is not None
    assert stored_run.failure.primary_reason is None
    assert len(repository.results_by_agent("mock")) == 1
    assert len(repository.results_by_environment_task("openenv", "email_classification")) == 1
    repository.close()


def test_repository_persists_standalone_run(tmp_path) -> None:
    repository = ResultRepository(tmp_path / "standalone.db")
    result = AgentRunner().run(
        OpenEnv(),
        MockAgent([SUCCESSFUL_EMAIL]),
    )

    repository.save_run(result, agent="mock")

    assert repository.get_benchmark(result.run_id)["kind"] == "benchmark"
    assert repository.get_run(result.run_id).run_id == result.run_id
    repository.close()
