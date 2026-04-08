from __future__ import annotations

from typing import Any, Dict

from environment.models import Action, Reward

from .base import BaseTask
from .utils import extract_json_payload, normalize_text, word_overlap_score


class EmailClassificationTask(BaseTask):
    name = "email_classification"
    difficulty = "easy"
    objective = "Classify the email as spam or important and explain the decision using evidence from the message."
    max_steps = 3

    def scenario(self) -> Dict[str, Any]:
        return {
            "email": {
                "from": "billing-update@secure-payments.example",
                "subject": "Verify your account to avoid suspension",
                "body": (
                    "Your subscription will be canceled unless you confirm the attached form and provide card details. "
                    "Click the link immediately to restore access."
                ),
            },
            "allowed_labels": ["spam", "important"],
            "expected_label": "spam",
            "evidence_tokens": ["verify", "suspension", "card", "click", "immediately", "form"],
        }

    def submission_format(self) -> str:
        return "Return JSON with keys: label and reason. Example: {\"label\": \"spam\", \"reason\": \"...\"}"

    def grade(self, action: Action, state: Dict[str, Any]) -> Reward:
        parsed = extract_json_payload(action.content)
        payload = parsed if isinstance(parsed, dict) else {}
        raw_text = normalize_text(action.content)

        label = str(payload.get("label") or payload.get("classification") or "").strip().lower()
        if not label:
            if "spam" in raw_text and "important" not in raw_text:
                label = "spam"
            elif "important" in raw_text and "spam" not in raw_text:
                label = "important"

        expected_label = state["scenario"]["expected_label"]
        correct_label = 1.0 if label == expected_label else 0.0

        reason_text = str(payload.get("reason") or payload.get("explanation") or action.content)
        evidence_score = word_overlap_score(reason_text, state["scenario"]["evidence_tokens"])
        format_score = 1.0 if isinstance(parsed, dict) and ("label" in parsed or "classification" in parsed) else 0.0

        score = round(min(1.0, (0.7 * correct_label) + (0.2 * evidence_score) + (0.1 * format_score)), 3)
        penalty = 0.0
        if label and label != expected_label:
            penalty += 0.25
        if not label:
            penalty += 0.15

        value = round(score - penalty, 3)
        breakdown = {
            "label": correct_label,
            "evidence": round(evidence_score, 3),
            "format": round(format_score, 3),
        }
        return Reward(value=value, score=score, penalty=penalty, breakdown=breakdown)
