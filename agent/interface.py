from __future__ import annotations

from typing import Any, Protocol, Sequence


class Agent(Protocol):
    """Minimal interface for agents that act on environment observations."""

    def observe(self, observation: Any) -> Any:
        """Return the next action for an environment observation."""


class MockAgent:
    """Deterministic agent that returns predefined actions in sequence."""

    def __init__(self, actions: Sequence[Any], repeat_last: bool = False) -> None:
        self._actions = list(actions)
        self._repeat_last = repeat_last
        self._index = 0

    def observe(self, observation: Any) -> Any:
        del observation
        if self._index < len(self._actions):
            action = self._actions[self._index]
            self._index += 1
            return action
        if self._repeat_last and self._actions:
            return self._actions[-1]
        raise RuntimeError("MockAgent has no remaining actions")
