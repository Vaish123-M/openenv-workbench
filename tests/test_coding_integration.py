from __future__ import annotations

import json

import pytest

from agent import AgentRunner, MockAgent
from coding import CodingEnvironment


SUCCESSFUL_ACTIONS = {
    "fix_addition": [
        {"tool": "edit_file", "path": "solution.py", "content": "def add_numbers(first: int, second: int) -> int:\n    return first + second\n"},
        {"tool": "submit"},
    ],
    "fix_email_normalization": [
        {"tool": "edit_file", "path": "solution.py", "content": "def normalize_email(value: str) -> str:\n    return value.strip().lower()\n"},
        {"tool": "submit"},
    ],
    "fix_safe_division": [
        {"tool": "edit_file", "path": "solution.py", "content": "def safe_divide(numerator: float, denominator: float) -> float:\n    return 0 if denominator == 0 else numerator / denominator\n"},
        {"tool": "run_tests"},
        {"tool": "submit"},
    ],
}


@pytest.mark.parametrize("task_name", list(SUCCESSFUL_ACTIONS))
def test_mock_agent_completes_coding_task(task_name: str) -> None:
    actions = [
        {"content": json.dumps(payload), "format": "json"}
        for payload in SUCCESSFUL_ACTIONS[task_name]
    ]
    result = AgentRunner().run(CodingEnvironment(), MockAgent(actions), task_id=task_name)

    assert result.completed is True
    assert result.termination_reason == "task_completed"
    assert result.grading.score == 1.0
    assert result.steps == len(actions)
    assert result.metrics.total_steps == len(actions)


def test_coding_runner_handles_invalid_action_and_max_steps() -> None:
    invalid = AgentRunner().run(
        CodingEnvironment(),
        MockAgent([{"content": "not-json", "format": "text"}]),
        task_id="fix_addition",
    )
    assert invalid.termination_reason == "invalid_action"
    assert invalid.error is not None

    maxed = AgentRunner().run(
        CodingEnvironment(),
        MockAgent(
            [{"content": json.dumps({"tool": "list_files"}), "format": "json"}],
            repeat_last=True,
        ),
        task_id="fix_addition",
        max_steps=2,
    )
    assert maxed.termination_reason == "max_steps_reached"
    assert maxed.steps == 2
