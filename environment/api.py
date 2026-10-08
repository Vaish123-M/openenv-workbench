from __future__ import annotations

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
ENVIRONMENT_FACTORIES: dict[str, Callable[[], OpenEnv]] = {"openenv": OpenEnv}


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
    return AgentRunner().run(
        environment=run_environment,
        agent=agent,
        task_id=request.task,
        max_steps=request.max_steps,
        timeout=request.timeout,
    )
