from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Tuple
from uuid import uuid4

from .models import Action, Observation, Reward
from tasks import get_task, list_tasks


def _normalize_action_content(content: str) -> str:
    return " ".join(content.lower().split())


class OpenEnv:
    def __init__(self, default_task: str = "email_classification") -> None:
        self.default_task = default_task
        self.task = None
        self._state: Dict[str, Any] = {}

    def reset(self, task_name: str | None = None) -> Observation:
        task = get_task(task_name or self.default_task)
        self.task = task
        self._state = task.initial_state()
        self._state["episode_id"] = uuid4().hex
        return task.build_observation(self._state)

    def step(self, action: Action | Dict[str, Any]) -> Tuple[Observation, float, bool, Dict[str, Any]]:
        if self.task is None:
            self.reset()

        assert self.task is not None
        action_model = action if isinstance(action, Action) else Action.model_validate(action)
        reward = self.task.grade(action_model, self._state)

        last_action = self._state.get("last_action", "")
        loop_penalty = -0.05 if last_action and _normalize_action_content(last_action) == _normalize_action_content(action_model.content) else 0.0

        self._state["step"] += 1
        self._state["last_action"] = action_model.content
        self._state["latest_score"] = reward.score
        self._state["latest_reward"] = reward.value + loop_penalty
        self._state["history"].append(
            {
                "step": self._state["step"],
                "action": action_model.model_dump(),
                "reward": reward.model_dump() | {"loop_penalty": loop_penalty},
            }
        )

        done = reward.score >= 0.999 or self._state["step"] >= self.task.max_steps
        self._state["done"] = done

        observation = self.task.build_observation(self._state)
        info = {
            "task_name": self.task.name,
            "difficulty": self.task.difficulty,
            "objective": self.task.objective,
            "score": reward.score,
            "reward": reward.value + loop_penalty,
            "penalty": reward.penalty + abs(loop_penalty),
            "breakdown": reward.breakdown,
            "remaining_steps": observation.remaining_steps,
            "loop_penalty": loop_penalty,
        }
        return observation, reward.value + loop_penalty, done, info

    def state(self) -> Dict[str, Any]:
        return deepcopy(self._state)

    def available_tasks(self) -> list[str]:
        return list_tasks()
