"""Web material for planning, from the model's own live search.

This module used to be a two-provider arrangement: the model's live search first, and a
metered search API (Tavily) as the fallback. **The fallback is gone.** It was measured
returning HTTP 432 for every query -- Chinese and Japanese alike -- so it was not a safety
net, only a slower path to the same empty answer, and the images it once supplied were the
*least* relevant of the three sources. Everything here now goes through the model's own
search, which the project already pays for, answers in Chinese with current prices, and has
no separate quota to exhaust.

Removing it also removed the last dependency on a key that only ever failed, and the
per-place price lookup (``search_snippets``) now reads the same live search instead of a
provider that returned nothing.
"""

from typing import Any

from app.agents.tools import dashscope_search

# A price answer is one short line. The planning floor (40 characters, in
# `dashscope_search.MIN_ANSWER_CHARS`) is a rule about paragraphs, and applying it here
# discarded every per-place price lookup in a measured Tokyo run.
PRICE_ANSWER_MIN_CHARS = 6


def _search_web_with_model(query: str, destination: str) -> tuple[list[dict[str, Any]], str] | None:
    """Search through the model's own live search, or None when that is unavailable.

    Returns one answer-shaped result instead of a list of pages: the compatible-mode
    response exposes only the finished answer, with no result URLs. That is enough, since
    the plan only ever used this material to extract place names and prices, and pictures
    come from the photo library (``place_images``).
    """
    if not dashscope_search.search_available():
        return None
    answer = dashscope_search.grounded_answer(
        f"{destination}：{query}\n\n"
        "请给出具体的景点、餐厅、酒店名称和价格等事实信息，名称一律用中文。"
    )
    if not answer:
        return None
    return (
        [
            {
                "source_type": "web_answer",
                # Never rendered as a recommendation: the provider markers in
                # `recommendations._is_descriptive_name` reject this heading by name.
                "title": "联网检索摘要",
                "url": None,
                "snippet": answer,
                "score": None,
                "image_url": None,
            }
        ],
        "completed",
    )


def search_web(query: str, destination: str, limit: int = 5) -> tuple[list[dict[str, Any]], str]:
    """Search the live web for trip material; never raises."""
    live = _search_web_with_model(query, destination)
    if live is not None:
        return live
    return [], "fallback: live search is not configured"


def search_snippets(query: str, limit: int = 3, timeout: float = 8.0) -> list[str]:
    """Return the live search's answer for one query as a snippet list; never raises.

    Used to look up what a specific recommended place actually costs, instead of guessing
    from a category default. ``limit`` is accepted and ignored: the live search returns one
    answer rather than a ranked list.

    ``timeout`` is honoured. It used to be accepted and dropped, on the reasoning that the
    live search "bounds its own request" -- it does, at 120s, which is right for planning
    material and absurd for one price line. A single stalled lookup could therefore hold a
    price phase open for two minutes. Callers pass a bound sized to the answer they want.

    A price answer is one line, so the paragraph-length floor used for planning material
    would throw away every one of them; the floor here is "there is a number in it", which
    is what the caller needs. Returns an empty list on any failure, so the caller falls
    back to a labelled default estimate rather than showing a wrong price.
    """
    if not dashscope_search.search_available():
        return []
    try:
        answer = dashscope_search.grounded_answer(
            f"{query}\n\n请给出具体价格数字和币种（例如「人均 1200 日元」「门票 600 日元」），"
            "只写事实，不要输出多余解释。",
            min_chars=PRICE_ANSWER_MIN_CHARS,
            timeout=timeout,
        )
    except Exception:
        # The lookup is best-effort by contract; a failure must not break a plan.
        return []
    text = str(answer or "").strip()
    if not any(character.isdigit() for character in text):
        # An answer with no number in it cannot contain a price, whatever else it says.
        return []
    return [text]
