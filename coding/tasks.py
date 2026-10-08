from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class CodingTask:
    name: str
    difficulty: str
    objective: str
    files: dict[str, str]
    test_command: tuple[str, ...]
    grader: Callable[[str], tuple[float, dict[str, float]]]
    max_steps: int = 12


def _pytest_grader(output: str) -> tuple[float, dict[str, float]]:
    passed = " passed" in output or " passed in " in output
    score = 1.0 if passed and "failed" not in output.lower() and "error" not in output.lower() else 0.0
    return score, {"tests_passed": score}


CODING_TASKS = {
    "fix_addition": CodingTask(
        name="fix_addition",
        difficulty="easy",
        objective="Fix add_numbers so it returns the sum of two integers.",
        files={
            "solution.py": "def add_numbers(first: int, second: int) -> int:\n    return first - second\n",
            "test_solution.py": (
                "from solution import add_numbers\n\n"
                "def test_add_numbers():\n"
                "    assert add_numbers(2, 3) == 5\n"
                "    assert add_numbers(-2, 3) == 1\n"
            ),
        },
        test_command=("pytest", "-q", "test_solution.py"),
        grader=_pytest_grader,
    ),
    "fix_email_normalization": CodingTask(
        name="fix_email_normalization",
        difficulty="medium",
        objective="Normalize an email by trimming whitespace and lowercasing it.",
        files={
            "solution.py": (
                "def normalize_email(value: str) -> str:\n"
                "    return value\n"
            ),
            "test_solution.py": (
                "from solution import normalize_email\n\n"
                "def test_normalize_email():\n"
                "    assert normalize_email('  Alice@Example.COM ') == 'alice@example.com'\n"
            ),
        },
        test_command=("pytest", "-q", "test_solution.py"),
        grader=_pytest_grader,
    ),
    "fix_safe_division": CodingTask(
        name="fix_safe_division",
        difficulty="hard",
        objective="Return zero when dividing by zero; otherwise return the quotient.",
        files={
            "solution.py": (
                "def safe_divide(numerator: float, denominator: float) -> float:\n"
                "    return numerator / denominator\n"
            ),
            "test_solution.py": (
                "from solution import safe_divide\n\n"
                "def test_safe_divide():\n"
                "    assert safe_divide(6, 2) == 3\n"
                "    assert safe_divide(6, 0) == 0\n"
            ),
        },
        test_command=("pytest", "-q", "test_solution.py"),
        grader=_pytest_grader,
    ),
}


def get_coding_task(name: str) -> CodingTask:
    try:
        return CODING_TASKS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown coding task: {name}") from exc


def list_coding_tasks() -> list[str]:
    return list(CODING_TASKS)
