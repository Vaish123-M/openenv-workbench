from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI
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


app = FastAPI(title="openenv-workbench", version="1.0.0")
env = OpenEnv()


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
