from __future__ import annotations

from typing import Any, Dict

from environment.models import Action, Reward

from .base import BaseTask
from .utils import contains_any, normalize_text


class CustomerSupportReplyTask(BaseTask):
    name = "customer_support_reply"
    difficulty = "hard"
    objective = "Write a polite customer support reply that acknowledges the issue, follows policy, and gives a clear next step."
    max_steps = 3

    def scenario(self) -> Dict[str, Any]:
        return {
            "customer_message": (
                "Hi, my order #4821 arrived with a cracked screen. I need a replacement or a refund. Please help."
            ),
            "policy": {
                "tone": "apologetic, professional, concise",
                "allowed_actions": ["replacement", "refund"],
                "required_follow_up": ["order number", "photo", "support review"],
                "timeframe": "within 30 days of delivery",
            },
        }

    def submission_format(self) -> str:
        return "Return a helpful support reply in plain text."

    def grade(self, action: Action, state: Dict[str, Any]) -> Reward:
        text = normalize_text(action.content)
        apology_score = 1.0 if contains_any(text, ["sorry", "apolog", "apologies", "we're sorry", "i'm sorry"]) else 0.0
        issue_score = 1.0 if contains_any(text, ["cracked", "damaged", "broken", "arrived damaged", "screen"]) else 0.0
        resolution_score = 1.0 if contains_any(text, ["replacement", "refund"]) else 0.0
        followup_score = 1.0 if contains_any(text, ["order", "photo", "support review", "within 30 days", "days of delivery"]) else 0.0
        professionalism_score = 1.0 if not contains_any(text, ["not my problem", "can't help", "your fault", "no refund guaranteed"]) else 0.0

        score = round(min(1.0, (0.2 * apology_score) + (0.2 * issue_score) + (0.3 * resolution_score) + (0.2 * followup_score) + (0.1 * professionalism_score)), 3)
        penalty = 0.0
        if contains_any(text, ["no refund guaranteed", "not my problem", "your fault"]):
            penalty += 0.2
        if len(text.split()) < 15:
            penalty += 0.05

        value = round(score - penalty, 3)
        return Reward(
            value=value,
            score=score,
            penalty=round(penalty, 3),
            breakdown={
                "apology": apology_score,
                "issue": issue_score,
                "resolution": resolution_score,
                "follow_up": followup_score,
                "professionalism": professionalism_score,
            },
        )
