from __future__ import annotations

import logging
import asyncio
import inspect
import time
from typing import Any, Protocol

from pydantic import ValidationError

from environment.models import Action

from .interface import Agent, AsyncAgent
from .models import ExecutionMetrics, GradingSummary, RunError, RunResult

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
        result._started_at = started

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
                self._record_agent_metadata(result, agent)
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
                if self._is_invalid_environment_action(info):
                    return self._failure(
                        result,
                        "invalid_action",
                        ValueError(info["result"].get("error", "environment rejected action")),
                        "Environment rejected an invalid action",
                    )
            except ValueError as exc:
                return self._failure(result, "unexpected_termination", exc, "Environment returned an invalid step result")
            except Exception as exc:
                return self._failure(result, "environment_error", exc, "Environment step failed")

            result.steps += 1
            result.info = info
            self._record_grading(result, info)
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

    async def run_async(
        self,
        environment: Environment,
        agent: Agent | AsyncAgent,
        task_id: str | None = None,
        max_steps: int = 10,
        timeout: float | None = None,
    ) -> RunResult:
        """Run sync or async components with cancellable asyncio deadlines.

        Awaitable reset, agent, and step calls are cancelled by the runner when
        the overall deadline expires. External task cancellation is propagated
        after best-effort cleanup. Synchronous calls remain non-preemptible.
        """
        self._validate_limits(max_steps, timeout)
        started = time.monotonic()
        result = RunResult(
            environment=_environment_name(environment),
            task_id=task_id,
            termination_reason="unexpected_termination",
        )
        result._started_at = started
        try:
            try:
                observation = await self._call_with_deadline(
                    lambda: environment.reset(task_id), started, timeout, "environment"
                )
            except _OperationTimeout as exc:
                return self._failure(result, "timeout", exc, "Environment reset timed out")
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
                try:
                    observer = getattr(agent, "aobserve", None)
                    if observer is None:
                        observer = getattr(agent, "observe")
                    action = await self._call_with_deadline(
                        lambda: observer(observation), started, timeout, "agent"
                    )
                    self._record_agent_metadata(result, agent)
                except _OperationTimeout as exc:
                    return self._failure(result, "timeout", exc, "Agent call timed out")
                except Exception as exc:
                    return self._failure(result, "agent_error", exc, "Agent failed while choosing an action")

                try:
                    Action.model_validate(action)
                except (ValidationError, TypeError, ValueError) as exc:
                    return self._failure(result, "invalid_action", exc, "Agent returned an invalid action")
                result.final_action = action

                try:
                    observation, _, done, info = await self._call_with_deadline(
                        lambda: environment.step(action), started, timeout, "environment"
                    )
                    if self._is_invalid_environment_action(info):
                        return self._failure(
                            result,
                            "invalid_action",
                            ValueError(info["result"].get("error", "environment rejected action")),
                            "Environment rejected an invalid action",
                        )
                except _OperationTimeout as exc:
                    return self._failure(result, "timeout", exc, "Environment step timed out")
                except ValueError as exc:
                    return self._failure(result, "unexpected_termination", exc, "Environment returned an invalid step result")
                except Exception as exc:
                    return self._failure(result, "environment_error", exc, "Environment step failed")

                result.steps += 1
                result.info = info
                self._record_grading(result, info)
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
        except asyncio.CancelledError:
            result.termination_reason = "cancelled"
            raise
        finally:
            await self._cleanup(environment, agent)

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
    def _validate_limits(max_steps: int, timeout: float | None) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if timeout is not None and timeout <= 0:
            raise ValueError("timeout must be greater than 0")

    @staticmethod
    async def _call_with_deadline(
        call: Any, started: float, timeout: float | None, phase: str
    ) -> Any:
        value = call()
        if not inspect.isawaitable(value):
            if timeout is not None and time.monotonic() - started >= timeout:
                raise _timeout_for_phase(phase)
            return value
        remaining = None if timeout is None else timeout - (time.monotonic() - started)
        if remaining is not None and remaining <= 0:
            value.close() if hasattr(value, "close") else None
            raise _timeout_for_phase(phase)
        try:
            return await asyncio.wait_for(value, timeout=remaining)
        except asyncio.TimeoutError as exc:
            raise _timeout_for_phase(phase) from exc

    @staticmethod
    async def _cleanup(environment: Environment, agent: Agent | AsyncAgent) -> None:
        for owner in (agent, environment):
            cleanup = getattr(owner, "aclose", None)
            if cleanup is None:
                cleanup = getattr(owner, "close", None)
            if cleanup is None:
                continue
            try:
                value = cleanup()
                if inspect.isawaitable(value):
                    await asyncio.shield(value)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Runner cleanup failed for %s", owner.__class__.__name__)

    @staticmethod
    def _finish(result: RunResult, reason: str) -> RunResult:
        result.termination_reason = reason  # type: ignore[assignment]
        AgentRunner._finalize_metrics(result, reason)
        return result

    @staticmethod
    def _failure(result: RunResult, reason: str, exc: Exception, log_message: str) -> RunResult:
        result.error = _error(exc)
        result.termination_reason = reason  # type: ignore[assignment]
        result.metrics.failed_steps += 1
        if reason == "invalid_action":
            result.metrics.invalid_action_count += 1
        elif reason == "agent_error":
            result.metrics.agent_errors.append(result.error)
        elif reason == "environment_error":
            result.metrics.environment_errors.append(result.error)
        logger.error("%s: %s", log_message, result.error.error_type)
        AgentRunner._finalize_metrics(result, reason)
        return result

    @staticmethod
    def _record_grading(result: RunResult, info: dict[str, Any]) -> None:
        result.grading = GradingSummary(
            score=info.get("score"),
            reward=info.get("reward"),
            penalty=info.get("penalty"),
            breakdown=info.get("breakdown", {}),
        )

    @staticmethod
    def _is_invalid_environment_action(info: dict[str, Any]) -> bool:
        nested = info.get("result")
        return isinstance(nested, dict) and nested.get("invalid_action") is True

    @staticmethod
    def _record_agent_metadata(result: RunResult, agent: Any) -> None:
        model_name = getattr(agent, "model_name", None)
        if model_name:
            result.metrics.model_name = model_name
        token_usage = getattr(agent, "last_token_usage", None)
        if isinstance(token_usage, dict):
            result.metrics.token_usage = {
                key: value for key, value in token_usage.items() if isinstance(value, int)
            }

    @staticmethod
    def _finalize_metrics(result: RunResult, reason: str) -> None:
        result.metrics.total_steps = result.steps
        result.metrics.elapsed_time = max(time.monotonic() - result._started_at, 0.0)
        result.metrics.timed_out = reason == "timeout"


class _OperationTimeout(TimeoutError):
    pass


class AgentTimeoutError(_OperationTimeout):
    pass


class EnvironmentTimeoutError(_OperationTimeout):
    pass


def _timeout_for_phase(phase: str) -> _OperationTimeout:
    if phase == "agent":
        return AgentTimeoutError("agent operation exceeded timeout")
    return EnvironmentTimeoutError("environment operation exceeded timeout")
