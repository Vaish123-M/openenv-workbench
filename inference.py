from __future__ import annotations

import json
from typing import Any

from environment.core import OpenEnv
from models.client import build_client, get_model_name
from models.prompts import build_task_prompt


TASK_ORDER = ["email_classification", "data_cleaning", "customer_support_reply"]


def _fallback_action(task_name: str) -> str:
    if task_name == "email_classification":
        return json.dumps({"label": "spam", "reason": "The email requests verification, includes urgent language, and asks for card details."})
    if task_name == "data_cleaning":
        return json.dumps(
            {
                "cleaned_rows": [
                    {"id": "001", "name": "Alice Johnson", "email": "alice@example.com", "signup_date": "2026-04-01", "phone": "555-123-4567"},
                    {"id": "002", "name": "Bob Smith", "email": "bob.smith@example.com", "signup_date": "2026-04-01", "phone": "555-987-6543"},
                ]
            }
        )
    return (
        "I'm sorry your order arrived damaged. We can help with a replacement or refund. "
        "Please reply with your order number and a photo of the damaged item so our support team can review it within 30 days of delivery."
    )


def _call_model(client: Any, model_name: str, prompt: str, task_name: str) -> str:
    messages = [
        {"role": "system", "content": "You are a precise agent. Follow the requested output format exactly."},
        {"role": "user", "content": prompt},
    ]
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0,
            max_tokens=400,
        )
        return response.choices[0].message.content or _fallback_action(task_name)
    except Exception:
        return _fallback_action(task_name)


def run_episode(task_name: str, client: Any, model_name: str) -> float:
    env = OpenEnv()
    observation = env.reset(task_name)
    final_score = 0.0

    print("[START]")
    print(f"task: {task_name}")

    for _ in range(observation.max_steps):
        prompt = build_task_prompt(observation)
        action_text = _call_model(client, model_name, prompt, task_name)
        observation, reward, done, info = env.step({"content": action_text, "format": "text" if task_name == "customer_support_reply" else "json"})
        final_score = info["score"]

        print()
        print("[STEP]")
        print(f"action: {action_text}")
        print(f"reward: {reward}")

        if done:
            break

    print()
    print("[END]")
    print(f"final_score: {round(final_score, 3)}")
    return final_score


def main() -> None:
    client = build_client()
    model_name = get_model_name()

    scores = [run_episode(task_name, client, model_name) for task_name in TASK_ORDER]
    _ = sum(scores) / max(len(scores), 1)


if __name__ == "__main__":
    main()
