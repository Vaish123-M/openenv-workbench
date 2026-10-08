from __future__ import annotations

import logging
import time
from typing import Any, Protocol

from pydantic import ValidationError

from environment.models import Action

from .interface import Agent
from .models import RunError, RunResult

logger = logging.getLogger(__name__)


class Environment(Protocol):
    """The subset of the OpenEnv interface required by AgentRunner."""

    def reset(self, task_name: str | None = None) -> Any:
        ...

    def step(self, action: Any) -> Any:
        ...


def _environment_name(environment: Environment) -> str:
    return environment.__class__.__name__


def _error(exc: Exception) -> RunError:
    return RunError(error_type=type(exc).__name__, message=str(exc)[:1000])


class AgentRunner:
    """Execute an agent against an OpenEnv-compatible environment.

    ``timeout`` is cooperative: the runner checks the deadline between reset,
    agent, and environment calls. Calls already in progress cannot be
    forcefully interrupted without changing the environment/agent architecture.
    """

    def run(
        self,
        environment: Environment,
        agent: Agent,
        task_id: str | None = None,
        max_steps: int = 10,
        timeout: float | None = None,
    ) -> RunResult:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be greater than 0")

        started = time.monotonic()
        result = RunResult(
            environment=_environment_name(environment),
            task_id=task_id,
            termination_reason="unexpected_termination",
        )

        if self._timed_out(started, timeout):
            return self._finish(result, "timeout")

        try:
            observation = environment.reset(task_id)
        except Exception as exc:
            return self._failure(result, "environment_error", exc, "Environment reset failed")

        if observation is None:
            return self._failure(
                result,
                "missing_observation",
                ValueError("reset returned no observation"),
                "Environment reset returned no observation",
            )
        result.final_observation = observation
        result.task_id = task_id or getattr(observation, "task_name", None)
        result.environment = getattr(observation, "task_name", result.environment)

        for _ in range(max_steps):
            if self._timed_out(started, timeout):
                return self._finish(result, "timeout")

            try:
                action = agent.observe(observation)
            except Exception as exc:
                return self._failure(result, "agent_error", exc, "Agent failed while choosing an action")

            try:
                Action.model_validate(action)
            except (ValidationError, TypeError, ValueError) as exc:
                return self._failure(result, "invalid_action", exc, "Agent returned an invalid action")

            result.final_action = action
            if self._timed_out(started, timeout):
                return self._finish(result, "timeout")

            try:
                step_result = environment.step(action)
                observation, _, done, info = self._unpack_step(step_result)
            except ValueError as exc:
                return self._failure(result, "unexpected_termination", exc, "Environment returned an invalid step result")
            except Exception as exc:
                return self._failure(result, "environment_error", exc, "Environment step failed")

            result.steps += 1
            result.info = info
            if observation is None:
                return self._failure(
                    result,
                    "missing_observation",
                    ValueError("step returned no observation"),
                    "Environment step returned no observation",
                )
            result.final_observation = observation

            if done or getattr(observation, "done", False):
                result.completed = True
                return self._finish(result, "task_completed")

        return self._finish(result, "max_steps_reached")

    @staticmethod
    def _unpack_step(step_result: Any) -> tuple[Any, float, bool, dict[str, Any]]:
        if not isinstance(step_result, tuple) or len(step_result) != 4:
            raise ValueError("environment.step must return (observation, reward, done, info)")
        observation, reward, done, info = step_result
        if not isinstance(done, bool):
            raise ValueError("environment.step returned a non-boolean done value")
        if not isinstance(info, dict):
            raise ValueError("environment.step returned non-dict info")
        return observation, reward, done, info

    @staticmethod
    def _timed_out(started: float, timeout: float | None) -> bool:
        return timeout is not None and time.monotonic() - started >= timeout

    @staticmethod
    def _finish(result: RunResult, reason: str) -> RunResult:
        result.termination_reason = reason  # type: ignore[assignment]
        return result

    @staticmethod
    def _failure(result: RunResult, reason: str, exc: Exception, log_message: str) -> RunResult:
        result.error = _error(exc)
        result.termination_reason = reason  # type: ignore[assignment]
        logger.error("%s: %s", log_message, result.error.error_type)
        return result
