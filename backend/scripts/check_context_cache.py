"""Verify DashScope implicit context-cache hits without printing credentials or content."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agents.tools.llm import BASE_URL, MODEL_NAME


def _usage(response) -> dict:
    usage = response.usage
    details = getattr(usage, "prompt_tokens_details", None)
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "cached_tokens": getattr(details, "cached_tokens", 0) if details is not None else 0,
    }


def main() -> None:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    key = os.getenv("DASHSCOPE_API_KEY")
    if not key:
        raise SystemExit("DASHSCOPE_API_KEY is not configured")
    client = OpenAI(api_key=key, base_url=BASE_URL, timeout=60.0, max_retries=0)
    common_prefix = "以下是只用于抽取测试的东京旅行素材。浅草寺、上野公园、东京塔。" * 420
    usages = []
    for suffix in ("仅回答数字一。", "仅回答数字二。"):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": common_prefix},
                {"role": "user", "content": suffix},
            ],
            extra_body={"enable_thinking": False},
            temperature=0,
            max_tokens=8,
        )
        usages.append(_usage(response))
    print(json.dumps({"model": MODEL_NAME, "requests": usages}, ensure_ascii=False))


if __name__ == "__main__":
    main()
