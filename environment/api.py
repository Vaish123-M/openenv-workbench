from __future__ import annotations

import os
from typing import Any, Callable, Dict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .core import OpenEnv
from .models import Action


class ResetRequest(BaseModel):
    task_name: str | None = Field(default=None, description="Task to load for the next episode")


class StepRequest(BaseModel):
    action: Action


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class RunRequest(BaseModel):
    environment: str = Field(default="openenv", description="Registered environment name")
    task: str = Field(default="email_classification", description="Registered task name")
    agent: str = Field(default="mock", description="Registered agent name")
    max_steps: int = Field(default=10, ge=1, le=100)
    timeout: float | None = Field(default=None, gt=0)
    actions: list[Action] | None = Field(
        default=None,
        description="Predefined actions for the mock agent",
    )


app = FastAPI(title="openenv-workbench", version="1.0.0")
env = OpenEnv()
_result_repositories: dict[str, Any] = {}


def _results_repository() -> Any:
    from storage import ResultRepository

    path = os.getenv("OPENENV_RESULTS_DB", "openenv_results.db")
    if path not in _result_repositories:
        _result_repositories[path] = ResultRepository(path)
    return _result_repositories[path]
def _coding_environment() -> Any:
    from coding import CodingEnvironment

    return CodingEnvironment()


ENVIRONMENT_FACTORIES: dict[str, Callable[[], Any]] = {
    "openenv": OpenEnv,
    "coding": _coding_environment,
}


def _build_agent(request: RunRequest) -> Any:
    from agent import LLMBackedAgent, MockAgent, OpenAIProvider

    if request.agent == "mock":
        if not request.actions:
            raise ValueError("actions are required for the mock agent")
        return MockAgent(request.actions)
    if request.agent == "openai":
        return LLMBackedAgent(OpenAIProvider())
    raise ValueError(f"Unknown registered agent: {request.agent}")


@app.post("/reset")
def reset(request: ResetRequest) -> Dict[str, Any]:
    observation = env.reset(request.task_name)
    return observation.model_dump()


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest) -> StepResponse:
    observation, reward, done, info = env.step(request.action)
    return StepResponse(observation=observation.model_dump(), reward=reward, done=done, info=info)


@app.get("/state")
def state() -> Dict[str, Any]:
    return env.state()


@app.post("/runs")
def run_agent(request: RunRequest) -> Any:
    from agent import AgentRunner

    environment_factory = ENVIRONMENT_FACTORIES.get(request.environment)
    if environment_factory is None:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown registered environment: {request.environment}",
        )

    try:
        agent = _build_agent(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    run_environment = environment_factory()
    result = AgentRunner().run(
        environment=run_environment,
        agent=agent,
        task_id=request.task,
        max_steps=request.max_steps,
        timeout=request.timeout,
    )
    _results_repository().save_run(result, agent=request.agent)
    return result


@app.get("/results/benchmarks/{benchmark_run_id}")
def get_benchmark_result(benchmark_run_id: str) -> Any:
    result = _results_repository().get_benchmark(benchmark_run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    return result


@app.get("/results/runs/{run_id}")
def get_task_result(run_id: str) -> Any:
    result = _results_repository().get_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Task run not found")
    return result


@app.get("/results/agents/{agent}")
def get_agent_results(agent: str) -> list[Any]:
    return _results_repository().results_by_agent(agent)


@app.get("/results/environments/{environment}")
def get_environment_results(environment: str, task: str | None = None) -> list[Any]:
    return _results_repository().results_by_environment_task(environment, task)
