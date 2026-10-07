"""Shared chat-model access for the planning agent tools."""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI


ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(ENV_PATH)

# One model setting for the planning-advice pipeline: recommendation extraction,
# grounded search and budget allocation all import this value. Keep the environment
# override so deployments can pin a snapshot without editing application code.
MODEL_NAME = os.getenv("PLANNING_MODEL_NAME", "qwen3.7-plus").strip() or "qwen3.7-plus"
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# The OpenAI SDK defaults to a 10 minute timeout with automatic retries. A single
# stalled provider call would therefore hold a planning request open for minutes,
# which the reverse proxy answers with a 504 and the user sees as "cannot generate
# a plan". Every model client must bound its own wait instead.
REQUEST_TIMEOUT = 30.0
MAX_RETRIES = 1


def model_client() -> OpenAI | None:
    """Return a configured chat client, or None when no key is available."""
    load_dotenv(ENV_PATH)
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    return OpenAI(api_key=api_key, base_url=BASE_URL, timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES)
