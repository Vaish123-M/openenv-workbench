from __future__ import annotations

import json
import os
from typing import Any, Callable, Protocol

from openai import AsyncOpenAI, OpenAI
from pydantic import ValidationError

from environment.models import Action, Observation
from models.prompts import build_task_prompt


class LLMProvider(Protocol):
    """Provider-neutral interface used by an LLM-backed agent."""

    def complete(self, prompt: str) -> str:
        """Generate one completion for a prompt."""


class AsyncLLMProvider(Protocol):
    """Provider-neutral interface for asynchronous completions."""

    async def acomplete(self, prompt: str) -> str:
        """Generate one completion asynchronously."""


class LLMProviderError(RuntimeError):
    """Raised when a provider cannot produce a usable completion."""


class OpenAIProvider:
    """OpenAI-compatible chat-completion provider.

    The API key is read from ``OPENAI_API_KEY`` or ``HF_TOKEN``. A client may
    be injected for tests or applications that manage client construction.
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.last_token_usage: dict[str, int] | None = None
        if client is not None:
            self._client = client
            return

        token = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("OPENAI_API_KEY or HF_TOKEN must be set")
        self._client = OpenAI(
            api_key=token,
            base_url=base_url or os.getenv("API_BASE_URL", "https://api.openai.com/v1"),
        )

    def complete(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return only the requested task response."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=400,
            )
            content = response.choices[0].message.content
            self.last_token_usage = _token_usage(response)
        except Exception as exc:
            raise LLMProviderError(f"LLM provider request failed: {type(exc).__name__}") from exc

        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("LLM provider returned an empty response")
        return content.strip()


class AsyncOpenAIProvider:
    """Asynchronous OpenAI-compatible chat-completion provider."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model or os.getenv("MODEL_NAME", "gpt-4o-mini")
        self.last_token_usage: dict[str, int] | None = None
        if client is not None:
            self._client = client
            return
        token = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
        if not token:
            raise ValueError("OPENAI_API_KEY or HF_TOKEN must be set")
        self._client = AsyncOpenAI(
            api_key=token,
            base_url=base_url or os.getenv("API_BASE_URL", "https://api.openai.com/v1"),
        )

    async def acomplete(self, prompt: str) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "Return only the requested task response."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=400,
            )
            content = response.choices[0].message.content
            self.last_token_usage = _token_usage(response)
        except Exception as exc:
            raise LLMProviderError(f"LLM provider request failed: {type(exc).__name__}") from exc
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("LLM provider returned an empty response")
        return content.strip()


def _expected_action_format(observation: Observation) -> str:
    return "json" if "json" in observation.submission_format.lower() else "text"


def _token_usage(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    values: dict[str, int] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = getattr(usage, key, None)
        if isinstance(value, int):
            values[key] = value
    return values or None


class LLMBackedAgent:
    """Agent that prompts an LLM and returns a validated environment action."""

    def __init__(
        self,
        provider: LLMProvider,
        prompt_builder: Callable[[Observation], str] = build_task_prompt,
    ) -> None:
        self._provider = provider
        self._prompt_builder = prompt_builder
        self.model_name = getattr(provider, "model", None)
        self.last_token_usage: dict[str, int] | None = None

    def observe(self, observation: Observation) -> Action:
        prompt = self._prompt_builder(observation)
        response = self._provider.complete(prompt)
        self.last_token_usage = getattr(self._provider, "last_token_usage", None)
        return self._to_action(response, observation)

    @staticmethod
    def _to_action(response: str, observation: Observation) -> Action:
        if not response.strip():
            raise ValueError("LLM returned an empty action response")

        try:
            parsed = json.loads(response)
        except json.JSONDecodeError:
            parsed = None

        if isinstance(parsed, dict) and "content" in parsed:
            try:
                return Action.model_validate(parsed)
            except ValidationError as exc:
                raise ValueError("LLM returned an invalid action envelope") from exc

        return Action(content=response, format=_expected_action_format(observation))


class AsyncLLMBackedAgent:
    """Asynchronous LLM agent for use with ``AgentRunner.run_async``."""

    def __init__(
        self,
        provider: AsyncLLMProvider,
        prompt_builder: Callable[[Observation], str] = build_task_prompt,
    ) -> None:
        self._provider = provider
        self._prompt_builder = prompt_builder
        self.model_name = getattr(provider, "model", None)
        self.last_token_usage: dict[str, int] | None = None

    async def aobserve(self, observation: Observation) -> Action:
        response = await self._provider.acomplete(self._prompt_builder(observation))
        self.last_token_usage = getattr(self._provider, "last_token_usage", None)
        return LLMBackedAgent._to_action(response, observation)
