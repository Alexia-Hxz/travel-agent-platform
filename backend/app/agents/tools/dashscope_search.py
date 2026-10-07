"""Real-time search through the model's own web-search capability.

The project already pays for DashScope, and the configured planning model searches the live web when a
request sets `enable_search`. Measured on this machine: one such call answers a travel
question with current, factual material in about 20 seconds, in Chinese, without touching
the separate search subscription. Two things follow.

First, it can stand in for the search API for *text*. The search API's billing is per
request and a five day plan needs several lookups, which exhausted its quota in an
afternoon of testing; the model key has no such per-search ceiling.

Second, it can search *and* answer in the same call. Asking for a JSON array of real
places while the search is enabled returned twelve distinct, genuinely existing places in
one round trip, whereas the previous pipeline paid for a search plus a separate
extraction call for the same result.

What it does not provide is a source list: the compatible-mode response exposes only the
finished answer, with no result URLs. Anything that needs raw links or images therefore
still goes through `web_search`. Nothing here is ever shown to the traveller — the answer
text is used as extraction material, exactly like a search snippet was.
"""

import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from app.agents.tools.llm import BASE_URL, MODEL_NAME

logger = logging.getLogger(__name__)

ENV_PATH = Path(__file__).resolve().parents[3] / ".env"

# A searching model is slower than a plain one — the live lookups happen before the
# answer is written — so this is deliberately longer than the shared chat timeout.
SEARCH_TIMEOUT = 120.0
# The answer itself is a short list; the searches are what take the time.
SEARCH_MAX_TOKENS = 2000
# `enable_search` is passed out of band: it is a provider extension, not an OpenAI
# parameter, so it travels in `extra_body` and is rejected if placed at the top level.
SEARCH_BODY = {"enable_search": True, "enable_thinking": False}

# Sentences that deny a capability the request actually has. One live answer opened with
# "截至2024年7月，我无法实时联网检索或访问互联网" and then went on to give correct, detailed
# facts. The sentence is simply false — the search did run — and anything next to it is
# undermined by it, so it is removed before the text is used as material.
DISCLAIMER_MARKERS = (
    "无法实时联网", "无法访问互联网", "无法联网", "不能联网", "没有联网", "未联网",
    "无法获取实时", "无法获取最新", "知识截止", "知识库截止", "训练数据截止",
)
_NO_DISCLAIMER_RULE = (
    "本次请求已开启联网检索，你可以直接获取实时信息。"
    "不要声明自己无法联网，不要提及知识截止时间，也不要写任何免责声明或能力说明，"
    "直接给出检索到的事实。"
)
# Below this, an answer is not worth using as material — either it was a disclaimer with
# nothing behind it, or the search failed.
MIN_ANSWER_CHARS = 40

_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?；;])\s*")


def _strip_disclaimers(text: str) -> str:
    """Remove capability disclaimers from a live answer.

    Line by line rather than over the whole answer, because a disclaimer is usually the
    opening paragraph of a line that is *followed* by a table: a table's rows carry no
    sentence-ending punctuation at all, so splitting the whole text into sentences joined
    the disclaimer to the entire table and deleted both.
    """
    kept_lines: list[str] = []
    for line in text.splitlines():
        remaining = "".join(
            sentence
            for sentence in _SENTENCE_SPLIT.split(line)
            if not any(marker in sentence for marker in DISCLAIMER_MARKERS)
        ).strip()
        if remaining:
            kept_lines.append(remaining)
    return "\n".join(kept_lines).strip()


def search_available() -> bool:
    """Whether live-web search through the model is configured.

    `DISABLE_LIVE_WEB_SEARCH` turns it off even when a key is present. The test suite
    sets it: a real searching call takes about twenty seconds and reaches the network,
    which would make every test slow and order-dependent.
    """
    load_dotenv(ENV_PATH)
    if os.getenv("DISABLE_LIVE_WEB_SEARCH", "").strip().lower() in {"1", "true", "yes"}:
        return False
    return bool(os.getenv("DASHSCOPE_API_KEY"))


def search_client(timeout: float | None = None) -> OpenAI | None:
    """Return a client whose timeout leaves room for the live lookups.

    Planning material legitimately takes a long time -- a searching model runs its lookups
    before it writes its answer -- but a single per-place price line does not. ``timeout``
    lets that caller bound its own lookup instead of silently inheriting the planning bound,
    which is what every caller of ``search_snippets`` already assumed it was getting: they
    passed 8.0s and the value was dropped on the floor while the real limit stayed 120s.
    """
    load_dotenv(ENV_PATH)
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        return None
    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        timeout=timeout if timeout else SEARCH_TIMEOUT,
        max_retries=0,
    )


def _ask(
    prompt: str, max_tokens: int = SEARCH_MAX_TOKENS, timeout: float | None = None
) -> tuple[str, str | None] | None:
    """Send one searching request. Returns (content, finish_reason), or None.

    Never raises: a search that fails must degrade to "no material", because the caller
    already knows how to carry on without it.
    """
    client = search_client(timeout)
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            extra_body=SEARCH_BODY,
            temperature=0.1,
            max_tokens=max_tokens,
        )
    except Exception as error:  # noqa: BLE001 - any provider failure means "no material"
        logger.warning("live search call failed: %s", error)
        return None
    content = response.choices[0].message.content or ""
    finish_reason = response.choices[0].finish_reason
    if not content.strip():
        logger.warning("live search returned an empty answer (finish_reason=%s)", finish_reason)
        return None
    return content, finish_reason


def grounded_answer(
    query: str, min_chars: int = MIN_ANSWER_CHARS, timeout: float | None = None
) -> str | None:
    """Search the live web and return the answer text, or None.

    This is the drop-in replacement for a text search: the caller treats the returned
    text as it treated search snippets.

    ``timeout`` bounds one specific lookup. Left unset, the search keeps its own generous
    planning bound, which is right for category material and wrong for a single price line.

    ``min_chars`` exists because the 40-character floor is a rule about *planning*
    material, where a paragraph is expected. A one-line price answer ("东京国立博物馆常设展
    630 日元") is 15 characters and perfectly usable, and enforcing the paragraph floor on
    it silently discarded every per-place price lookup: measured 18 lookups, 18 discards,
    and every restaurant in a Tokyo plan falling back to the Chinese default price.
    """
    asked = _ask(
        f"请先联网检索，然后用中文回答下面的问题，尽量给出具体地名、店名和价格。\n"
        f"{_NO_DISCLAIMER_RULE}\n\n{query}",
        timeout=timeout,
    )
    if asked is None:
        return None
    cleaned = _strip_disclaimers(asked[0])
    if len(cleaned) < min_chars:
        logger.warning(
            "live search answer had no usable content after stripping disclaimers (%s chars < %s)",
            len(cleaned),
            min_chars,
        )
        return None
    return cleaned
