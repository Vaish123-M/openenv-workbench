from __future__ import annotations

import os

from openai import OpenAI


def build_client() -> OpenAI:
    base_url = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
    token = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "local-dev-key"
    return OpenAI(base_url=base_url, api_key=token)


def get_model_name() -> str:
    return os.getenv("MODEL_NAME", "gpt-4o-mini")
