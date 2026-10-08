from __future__ import annotations

import pytest

from agent import AgentRunner, LLMBackedAgent, LLMProviderError, OpenAIProvider
from environment.core import OpenEnv
from environment.models import Observation


class FakeProvider:
    def __init__(self, response: str) -> None:
        self.response = response
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.response


def test_llm_agent_returns_json_action_and_runs_environment() -> None:
    provider = FakeProvider(
        '{"label": "spam", "reason": "Verify suspension card click immediately form."}'
    )
    agent = LLMBackedAgent(provider)

    result = AgentRunner().run(OpenEnv(), agent, task_id="email_classification")

    assert result.completed is True
    assert result.steps == 1
    assert result.final_action.format == "json"
    assert provider.prompts


def test_llm_agent_accepts_explicit_action_envelope() -> None:
    observation = Observation(
        task_name="customer_support_reply",
        difficulty="hard",
        objective="reply",
        task_input={"customer_message": "The order arrived damaged."},
        submission_format="plain text",
        step=0,
        max_steps=3,
        remaining_steps=3,
        done=False,
    )
    agent = LLMBackedAgent(FakeProvider('{"content": "done", "format": "text"}'))

    action = agent.observe(observation)

    assert action.content == "done"
    assert action.format == "text"


def test_empty_llm_response_is_rejected() -> None:
    observation = Observation(
        task_name="email_classification",
        difficulty="easy",
        objective="classify",
        task_input={"email": {"subject": "Verify account", "body": "Provide card details."}},
        submission_format="Return JSON",
        step=0,
        max_steps=3,
        remaining_steps=3,
        done=False,
    )

    with pytest.raises(ValueError, match="empty"):
        LLMBackedAgent(FakeProvider(" ")).observe(observation)


def test_provider_errors_become_runner_agent_errors() -> None:
    class FailingProvider:
        def complete(self, prompt: str) -> str:
            raise LLMProviderError("request failed")

    result = AgentRunner().run(OpenEnv(), LLMBackedAgent(FailingProvider()))

    assert result.termination_reason == "agent_error"
    assert result.error is not None
    assert result.error.error_type == "LLMProviderError"


def test_openai_provider_uses_injected_client_without_api_key() -> None:
    class FakeCompletions:
        def create(self, **kwargs: object) -> object:
            class Message:
                content = "generated"

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    provider = OpenAIProvider(client=FakeClient())

    assert provider.complete("prompt") == "generated"
