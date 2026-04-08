from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, Dict

from environment.models import Action, Observation, Reward


class BaseTask(ABC):
    name: str
    difficulty: str
    objective: str
    max_steps: int = 3

    @abstractmethod
    def scenario(self) -> Dict[str, Any]:
        raise NotImplementedError

    def initial_state(self) -> Dict[str, Any]:
        return {
            "step": 0,
            "done": False,
            "latest_score": 0.0,
            "latest_reward": 0.0,
            "last_action": "",
            "history": [],
            "scenario": deepcopy(self.scenario()),
        }

    def build_observation(self, state: Dict[str, Any]) -> Observation:
        return Observation(
            task_name=self.name,
            difficulty=self.difficulty,
            objective=self.objective,
            task_input=deepcopy(state["scenario"]),
            submission_format=self.submission_format(),
            step=state["step"],
            max_steps=self.max_steps,
            remaining_steps=max(self.max_steps - state["step"], 0),
            done=state["done"],
            latest_score=state["latest_score"],
            history=deepcopy(state["history"]),
        )

    def submission_format(self) -> str:
        return "text"

    @abstractmethod
    def grade(self, action: Action, state: Dict[str, Any]) -> Reward:
        raise NotImplementedError
