from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from environment.models import Action, Observation

from .tasks import CodingTask, get_coding_task, list_coding_tasks

MAX_FILE_BYTES = 100_000
MAX_OUTPUT_CHARS = 8_000
ALLOWED_TOOLS = frozenset({"list_files", "read_file", "edit_file", "run_tests", "submit"})


class CodingEnvironment:
    """A task-owned, tool-limited Python coding workspace."""

    def __init__(self, test_timeout: float = 5.0) -> None:
        if test_timeout <= 0:
            raise ValueError("test_timeout must be greater than 0")
        self.test_timeout = test_timeout
        self.task: CodingTask | None = None
        self._workspace: Path | None = None
        self._state: dict[str, Any] = {}

    def reset(self, task_name: str | None = None) -> Observation:
        if self._workspace is not None:
            shutil.rmtree(self._workspace, ignore_errors=True)
        self.task = get_coding_task(task_name or "fix_addition")
        workspace = Path(tempfile.mkdtemp(prefix="openenv-coding-")).resolve()
        self._workspace = workspace
        for relative, content in self.task.files.items():
            target = self._safe_path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        self._state = {
            "step": 0,
            "done": False,
            "latest_score": 0.0,
            "latest_reward": 0.0,
            "history": [],
            "last_result": None,
            "last_error": None,
        }
        return self._observation()

    def step(self, action: Action | dict[str, Any]) -> tuple[Observation, float, bool, dict[str, Any]]:
        if self.task is None:
            self.reset()
        assert self.task is not None
        self._state["step"] += 1
        tool: str | None = None
        payload: dict[str, Any] = {}
        try:
            action_model = action if isinstance(action, Action) else Action.model_validate(action)
            payload = self._parse_action(action_model)
            tool = payload.get("tool")
            if tool not in ALLOWED_TOOLS:
                raise ValueError(f"Unknown coding tool: {tool}")
            result = self._dispatch(tool, payload)
        except (ValueError, OSError) as exc:
            result = {"error": str(exc), "score": 0.0, "done": False, "invalid_action": True}
        score = float(result.get("score", self._state["latest_score"]))
        done = bool(result.get("done", False))
        self._state.update(
            done=done,
            latest_score=score,
            latest_reward=score,
            last_result=result,
            last_error=result.get("error"),
        )
        self._state["history"].append({"step": self._state["step"], "tool": tool, "result": result})
        if self._state["step"] >= self.task.max_steps and not done:
            done = True
            self._state["done"] = True
            result["termination_reason"] = "max_steps_reached"

        observation = self._observation()
        info = {
            "task_name": self.task.name,
            "difficulty": self.task.difficulty,
            "objective": self.task.objective,
            "score": score,
            "reward": score,
            "penalty": 0.0,
            "breakdown": result.get("breakdown", {}),
            "tool": payload.get("tool"),
            "result": result,
        }
        return observation, score, done, info

    def state(self) -> dict[str, Any]:
        return {
            "task": self.task.name if self.task else None,
            "workspace": "task-owned-temporary-workspace",
            "step": self._state.get("step", 0),
            "done": self._state.get("done", False),
            "last_result": self._state.get("last_result"),
        }

    def available_tasks(self) -> list[str]:
        return list_coding_tasks()

    def close(self) -> None:
        if self._workspace is not None:
            shutil.rmtree(self._workspace, ignore_errors=True)
            self._workspace = None

    def _safe_path(self, relative: str) -> Path:
        if self._workspace is None:
            raise RuntimeError("Coding environment is not reset")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or path.name == "":
            raise ValueError("path must be a relative workspace path")
        resolved = (self._workspace / path).resolve()
        if resolved != self._workspace and self._workspace not in resolved.parents:
            raise ValueError("path escapes the coding workspace")
        return resolved

    @staticmethod
    def _parse_action(action: Action) -> dict[str, Any]:
        if action.format != "json":
            raise ValueError("coding actions must use format=json")
        try:
            payload = json.loads(action.content)
        except json.JSONDecodeError as exc:
            raise ValueError("coding action content must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("coding action must be a JSON object")
        return payload

    def _dispatch(self, tool: str, payload: dict[str, Any]) -> dict[str, Any]:
        if tool == "list_files":
            return {"files": sorted(str(path.relative_to(self._workspace)) for path in self._workspace.rglob("*") if path.is_file())}
        if tool == "read_file":
            path = self._safe_path(self._required_string(payload, "path"))
            if not path.is_file():
                raise ValueError("file does not exist")
            return {"path": str(path.relative_to(self._workspace)), "content": path.read_text(encoding="utf-8")[:MAX_FILE_BYTES]}
        if tool == "edit_file":
            path = self._safe_path(self._required_string(payload, "path"))
            content = self._required_string(payload, "content")
            if len(content.encode("utf-8")) > MAX_FILE_BYTES:
                raise ValueError("file content exceeds size limit")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return {"path": str(path.relative_to(self._workspace)), "edited": True}
        if tool == "run_tests":
            return self._run_tests()
        return self._submit()

    def _run_tests(self) -> dict[str, Any]:
        assert self.task is not None and self._workspace is not None
        try:
            completed = subprocess.run(
                [sys.executable, "-m", *self.task.test_command],
                cwd=self._workspace,
                capture_output=True,
                text=True,
                timeout=self.test_timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            return {"passed": False, "timed_out": True, "output": str(exc)[:MAX_OUTPUT_CHARS], "score": 0.0}
        output = (completed.stdout + "\n" + completed.stderr)[-MAX_OUTPUT_CHARS:]
        return {"passed": completed.returncode == 0, "output": output, "score": 0.0}

    def _submit(self) -> dict[str, Any]:
        test_result = self._run_tests()
        score, breakdown = self.task.grader(test_result.get("output", ""))
        return {
            "submitted": True,
            "passed": test_result.get("passed", False),
            "score": score,
            "breakdown": breakdown,
            "test_output": test_result.get("output", ""),
            "done": score >= 1.0,
            "termination_reason": "task_completed" if score >= 1.0 else "submission_failed",
        }

    @staticmethod
    def _required_string(payload: dict[str, Any], key: str) -> str:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            raise ValueError(f"{key} must be a non-empty string")
        return value

    def _observation(self) -> Observation:
        assert self.task is not None
        files = sorted(str(path.relative_to(self._workspace)) for path in self._workspace.rglob("*") if path.is_file())
        return Observation(
            task_name=self.task.name,
            difficulty=self.task.difficulty,
            objective=self.task.objective,
            task_input={"files": files, "last_result": self._state.get("last_result")},
            submission_format='JSON tool action: {"tool": "list_files|read_file|edit_file|run_tests|submit", ...}',
            step=self._state["step"],
            max_steps=self.task.max_steps,
            remaining_steps=max(self.task.max_steps - self._state["step"], 0),
            done=self._state["done"],
            latest_score=self._state["latest_score"],
            history=list(self._state["history"]),
        )
