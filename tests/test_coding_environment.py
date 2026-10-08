from __future__ import annotations

import json

import pytest

from coding import CodingEnvironment


def action(tool: str, **payload: object) -> dict[str, str]:
    return {"content": json.dumps({"tool": tool, **payload}), "format": "json"}


def test_coding_environment_controls_tools_and_workspace_paths() -> None:
    environment = CodingEnvironment()
    observation = environment.reset("fix_addition")

    assert observation.task_name == "fix_addition"
    _, _, _, info = environment.step(action("list_files"))
    assert info["result"]["files"] == ["solution.py", "test_solution.py"]

    _, _, _, read_info = environment.step(action("read_file", path="solution.py"))
    assert "return first - second" in read_info["result"]["content"]

    _, _, _, invalid_info = environment.step(action("read_file", path="../outside.py"))
    assert "escapes" in invalid_info["result"]["error"] or "relative" in invalid_info["result"]["error"]

    environment.close()


def test_coding_environment_runs_tests_and_submits_successfully() -> None:
    environment = CodingEnvironment()
    environment.reset("fix_addition")
    environment.step(
        action(
            "edit_file",
            path="solution.py",
            content="def add_numbers(first: int, second: int) -> int:\n    return first + second\n",
        )
    )
    _, reward, done, info = environment.step(action("submit"))

    assert done is True
    assert reward == 1.0
    assert info["result"]["passed"] is True
    assert info["result"]["termination_reason"] == "task_completed"


def test_coding_environment_failed_submission_is_retryable() -> None:
    environment = CodingEnvironment()
    environment.reset("fix_safe_division")

    _, reward, done, info = environment.step(action("submit"))

    assert done is False
    assert reward == 0.0
    assert info["result"]["termination_reason"] == "submission_failed"


def test_coding_environment_rejects_arbitrary_tools() -> None:
    environment = CodingEnvironment()
    environment.reset("fix_addition")

    _, _, _, info = environment.step(action("run_shell", command="whoami"))

    assert "Unknown coding tool" in info["result"]["error"]


@pytest.mark.parametrize("task_name", ["fix_addition", "fix_email_normalization", "fix_safe_division"])
def test_all_coding_tasks_have_deterministic_grading(task_name: str) -> None:
    environment = CodingEnvironment()
    observation = environment.reset(task_name)

    assert observation.difficulty in {"easy", "medium", "hard"}
    assert environment.available_tasks() == [
        "fix_addition",
        "fix_email_normalization",
        "fix_safe_division",
    ]
