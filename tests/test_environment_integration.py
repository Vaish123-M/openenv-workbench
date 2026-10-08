from __future__ import annotations

import pytest

from agent import AgentRunner, MockAgent
from environment.core import OpenEnv


SUCCESSFUL_ACTIONS = {
    "email_classification": {
        "content": '{"label": "spam", "reason": "Verify suspension card click immediately form."}',
        "format": "json",
    },
    "data_cleaning": {
        "content": (
            '{"cleaned_rows": ['
            '{"id": "001", "name": "Alice Johnson", "email": "alice@example.com", '
            '"signup_date": "2026-04-01", "phone": "555-123-4567"}, '
            '{"id": "002", "name": "Bob Smith", "email": "bob.smith@example.com", '
            '"signup_date": "2026-04-01", "phone": "555-987-6543"}]}'
        ),
        "format": "json",
    },
    "customer_support_reply": {
        "content": (
            "I'm sorry your order arrived with a cracked screen. We can provide a replacement "
            "or refund. Please send your order number and a photo so support review can be "
            "completed within 30 days of delivery."
        ),
        "format": "text",
    },
}


@pytest.mark.parametrize("task_name", list(SUCCESSFUL_ACTIONS))
def test_mock_agent_completes_every_existing_environment(task_name: str) -> None:
    result = AgentRunner().run(
        environment=OpenEnv(),
        agent=MockAgent([SUCCESSFUL_ACTIONS[task_name]]),
        task_id=task_name,
        max_steps=3,
    )

    assert result.completed is True
    assert result.termination_reason == "task_completed"
    assert result.task_id == task_name
    assert result.environment == task_name
    assert result.steps == 1
    assert result.metrics.total_steps == 1
    assert result.grading.score == 1.0
    assert result.grading.breakdown
    assert result.error is None


@pytest.mark.parametrize("task_name", list(SUCCESSFUL_ACTIONS))
def test_invalid_mock_action_is_reported_for_every_environment(task_name: str) -> None:
    result = AgentRunner().run(
        environment=OpenEnv(),
        agent=MockAgent([{"content": "", "format": "text"}]),
        task_id=task_name,
        max_steps=3,
    )

    assert result.completed is False
    assert result.termination_reason == "invalid_action"
    assert result.steps == 0
    assert result.metrics.invalid_action_count == 1
    assert result.metrics.failed_steps == 1
    assert result.error is not None


@pytest.mark.parametrize(
    ("task_name", "action"),
    [
        ("email_classification", {"content": "uncertain", "format": "text"}),
        ("data_cleaning", {"content": "{}", "format": "json"}),
        ("customer_support_reply", {"content": "hello", "format": "text"}),
    ],
)
def test_mock_agent_stops_at_max_steps_for_every_environment(
    task_name: str, action: dict[str, str]
) -> None:
    result = AgentRunner().run(
        environment=OpenEnv(),
        agent=MockAgent([action], repeat_last=True),
        task_id=task_name,
        max_steps=2,
    )

    assert result.completed is False
    assert result.termination_reason == "max_steps_reached"
    assert result.steps == 2
    assert result.metrics.total_steps == 2
    assert result.grading.score is not None
    assert result.error is None


@pytest.mark.parametrize("task_name", list(SUCCESSFUL_ACTIONS))
def test_agent_error_is_structured_for_every_environment(task_name: str) -> None:
    class FailingAgent:
        def observe(self, observation: object) -> object:
            raise RuntimeError("deterministic agent failure")

    result = AgentRunner().run(OpenEnv(), FailingAgent(), task_id=task_name)

    assert result.termination_reason == "agent_error"
    assert result.steps == 0
    assert result.metrics.failed_steps == 1
    assert len(result.metrics.agent_errors) == 1
    assert result.error is not None


def test_environment_error_is_structured_in_complete_flow() -> None:
    class FailingEnvironment(OpenEnv):
        def step(self, action: object) -> object:
            raise RuntimeError("deterministic environment failure")

    result = AgentRunner().run(
        FailingEnvironment(),
        MockAgent([SUCCESSFUL_ACTIONS["email_classification"]]),
        task_id="email_classification",
    )

    assert result.termination_reason == "environment_error"
    assert result.steps == 0
    assert result.metrics.failed_steps == 1
    assert len(result.metrics.environment_errors) == 1
    assert result.error is not None
