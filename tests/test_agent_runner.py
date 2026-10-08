from __future__ import annotations

import pytest

from agent import AgentRunner, MockAgent
from environment.core import OpenEnv


SUCCESSFUL_EMAIL = {
    "content": '{"label": "spam", "reason": "Verify suspension card click immediately form."}',
    "format": "json",
}


def test_successful_run_completes_against_existing_environment() -> None:
    result = AgentRunner().run(OpenEnv(), MockAgent([SUCCESSFUL_EMAIL]), max_steps=3)

    assert result.completed is True
    assert result.termination_reason == "task_completed"
    assert result.steps == 1
    assert result.task_id == "email_classification"
    assert result.environment == "email_classification"
    assert result.error is None


def test_max_steps_termination() -> None:
    result = AgentRunner().run(
        OpenEnv(),
        MockAgent([{"content": "not enough", "format": "text"}], repeat_last=True),
        max_steps=2,
    )

    assert result.completed is False
    assert result.termination_reason == "max_steps_reached"
    assert result.steps == 2


class CountingEnvironment:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.actions: list[dict[str, str]] = []

    def reset(self, task_name: str | None = None) -> dict[str, bool | str]:
        self.reset_calls += 1
        return {"task_name": task_name or "fixture", "done": False}

    def step(self, action: dict[str, str]) -> tuple[dict[str, bool], float, bool, dict[str, str]]:
        self.actions.append(action)
        return {"done": True}, 1.0, True, {}


def test_agent_action_is_passed_to_environment_and_reset_once() -> None:
    environment = CountingEnvironment()
    action = {"content": "finish", "format": "text"}

    result = AgentRunner().run(environment, MockAgent([action]), task_id="fixture")

    assert result.steps == 1
    assert environment.reset_calls == 1
    assert environment.actions == [action]


def test_agent_exception_is_structured() -> None:
    class FailingAgent:
        def observe(self, observation: object) -> object:
            raise RuntimeError("agent failed")

    result = AgentRunner().run(OpenEnv(), FailingAgent())

    assert result.completed is False
    assert result.termination_reason == "agent_error"
    assert result.error is not None
    assert result.error.error_type == "RuntimeError"


def test_environment_exception_is_structured() -> None:
    class FailingEnvironment(CountingEnvironment):
        def step(self, action: dict[str, str]) -> tuple[dict[str, bool], float, bool, dict[str, str]]:
            raise RuntimeError("environment failed")

    result = AgentRunner().run(FailingEnvironment(), MockAgent([{"content": "x", "format": "text"}]))

    assert result.termination_reason == "environment_error"
    assert result.steps == 0
    assert result.error is not None


@pytest.mark.parametrize(
    "action",
    [None, {"content": "", "format": "text"}, {"content": "x", "format": "invalid"}],
)
def test_invalid_action_is_structured(action: object) -> None:
    environment = CountingEnvironment()

    result = AgentRunner().run(environment, MockAgent([action]))

    assert result.termination_reason == "invalid_action"
    assert result.steps == 0
    assert environment.actions == []
    assert result.error is not None
