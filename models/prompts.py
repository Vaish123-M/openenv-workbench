from __future__ import annotations

from environment.models import Observation


def build_task_prompt(observation: Observation) -> str:
    if observation.task_name == "email_classification":
        return (
            "Classify the email as spam or important. Return only JSON with keys label and reason. "
            f"Email: {observation.task_input['email']}"
        )
    if observation.task_name == "data_cleaning":
        return (
            "Clean the customer rows. Return only JSON with a cleaned_rows array. "
            f"Dirty rows: {observation.task_input['dirty_rows']}"
        )
    return (
        "Write a helpful customer support reply in plain text. Include apology, resolution, and next step. "
        f"Customer message: {observation.task_input['customer_message']}"
    )
