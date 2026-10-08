from __future__ import annotations

import asyncio

from agent import AsyncLLMBackedAgent, AsyncOpenAIProvider, LLMProviderError
from environment.models import Observation


class FakeAsyncProvider:
    async def acomplete(self, prompt: str) -> str:
        return '{"label": "spam", "reason": "card verification request"}'


def test_async_llm_agent_returns_validated_action() -> None:
    observation = Observation(
        task_name="email_classification",
        difficulty="easy",
        objective="classify",
        task_input={"email": {"subject": "Verify", "body": "Provide card details."}},
        submission_format="Return JSON",
        step=0,
        max_steps=3,
        remaining_steps=3,
        done=False,
    )

    action = asyncio.run(AsyncLLMBackedAgent(FakeAsyncProvider()).aobserve(observation))

    assert action.format == "json"
    assert '"label"' in action.content


def test_async_openai_provider_uses_injected_client() -> None:
    class FakeCompletions:
        async def create(self, **kwargs: object) -> object:
            class Message:
                content = "generated"

            class Choice:
                message = Message()

            class Response:
                choices = [Choice()]

            return Response()

    class FakeClient:
        chat = type("Chat", (), {"completions": FakeCompletions()})()

    assert asyncio.run(AsyncOpenAIProvider(client=FakeClient()).acomplete("prompt")) == "generated"
