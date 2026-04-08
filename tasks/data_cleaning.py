from __future__ import annotations

from typing import Any, Dict

from environment.models import Action, Reward

from .base import BaseTask
from .utils import compare_rows, extract_json_payload, sort_rows


class DataCleaningTask(BaseTask):
    name = "data_cleaning"
    difficulty = "medium"
    objective = "Remove duplicate customer rows and normalize all fields into a clean canonical table."
    max_steps = 3

    def scenario(self) -> Dict[str, Any]:
        return {
            "instructions": [
                "Remove duplicate customer rows.",
                "Normalize names to title case.",
                "Lowercase emails.",
                "Convert dates to YYYY-MM-DD.",
                "Format phone numbers as XXX-XXX-XXXX.",
            ],
            "dirty_rows": [
                {"id": "001", "name": "  alice johnson ", "email": "ALICE@EXAMPLE.com ", "signup_date": "2026/04/01", "phone": "(555) 123-4567"},
                {"id": "002", "name": "Bob Smith", "email": "bob.smith@example.com", "signup_date": "01-04-2026", "phone": "555.987.6543"},
                {"id": "001", "name": "Alice Johnson", "email": "alice@example.com", "signup_date": "2026-04-01", "phone": "5551234567"},
            ],
            "expected_rows": [
                {"id": "001", "name": "Alice Johnson", "email": "alice@example.com", "signup_date": "2026-04-01", "phone": "555-123-4567"},
                {"id": "002", "name": "Bob Smith", "email": "bob.smith@example.com", "signup_date": "2026-04-01", "phone": "555-987-6543"},
            ],
        }

    def submission_format(self) -> str:
        return "Return JSON with a cleaned_rows array. Example: {\"cleaned_rows\": [{...}, {...}]}"

    def grade(self, action: Action, state: Dict[str, Any]) -> Reward:
        parsed = extract_json_payload(action.content)
        if isinstance(parsed, dict) and "cleaned_rows" in parsed:
            cleaned_rows = parsed["cleaned_rows"]
        elif isinstance(parsed, list):
            cleaned_rows = parsed
        else:
            cleaned_rows = []

        if not isinstance(cleaned_rows, list):
            cleaned_rows = []

        expected_rows = sort_rows(state["scenario"]["expected_rows"])
        actual_rows = sort_rows([row for row in cleaned_rows if isinstance(row, dict)])
        score, breakdown = compare_rows(expected_rows, actual_rows)

        duplicate_penalty = max(0, len(actual_rows) - len(expected_rows)) * 0.05
        missing_penalty = 0.0 if actual_rows else 0.2
        penalty = round(min(0.4, duplicate_penalty + missing_penalty), 3)
        value = round(score - penalty, 3)

        return Reward(
            value=value,
            score=round(score, 3),
            penalty=penalty,
            breakdown={
                "field_matches": round(score, 3),
                **{f"field_{key}": round(val, 3) for key, val in breakdown.items()},
            },
        )
