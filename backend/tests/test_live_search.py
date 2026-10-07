"""The model's own live web search, used instead of a separate search subscription.

The project already pays for DashScope, and `qwen3.7-plus` searches the live web when a
request sets `enable_search`. These tests pin the behaviour that matters: it is preferred
when available, it is skipped when unavailable, its answer is not truncated the way a
short search snippet is, and nothing about it is ever shown to the traveller.
"""

from typing import Any

import pytest

from app.agents.tools import dashscope_search, recommendations, web_search
from app.agents.workflows import planner

LIVE_ANSWER = (
    "南京值得一去的景点包括："
    "中山陵（免费，需预约）、夫子庙-秦淮河风光带（免费）、南京博物院（免费，需预约）、"
    "总统府（门票35元）、玄武湖公园（免费）、雨花台烈士陵园（免费）、栖霞山（门票25元）、"
    "牛首山文化旅游区（门票98元）、阅江楼（门票40元）、中华门城堡（门票50元）、"
    "燕子矶（免费）、高淳老街（免费）。"
)


@pytest.fixture
def live_enabled(monkeypatch: pytest.MonkeyPatch):
    """Turn live search on, with the network call itself replaced.

    `conftest` disables live search for the whole suite; these tests are the ones that
    exercise it, so they re-enable it and answer the request locally.
    """
    monkeypatch.delenv("DISABLE_LIVE_WEB_SEARCH", raising=False)
    answers: list[str] = [LIVE_ANSWER]
    prompts: list[str] = []

    def _fake_ask(prompt: str, max_tokens: int = dashscope_search.SEARCH_MAX_TOKENS, timeout: float | None = None):
        prompts.append(prompt)
        return (answers.pop(0) if answers else ""), "stop"

    monkeypatch.setattr(dashscope_search, "_ask", _fake_ask)
    return {"prompts": prompts, "answers": answers}


def test_live_search_is_used_for_category_material(live_enabled, monkeypatch) -> None:
    """The live answer becomes the material, so no search-provider request is made."""
    called: list[str] = []
    monkeypatch.setattr(
        recommendations, "search_snippets", lambda query, limit=3, timeout=8.0: called.append(query) or []
    )

    material = recommendations._category_material("南京", "attraction")

    assert len(material) == 1
    assert "中山陵" in material[0]["snippet"]
    assert called == []
    # The question asks for real names in Chinese, which is what keeps one venue from
    # coming back once per language.
    assert "中文" in live_enabled["prompts"][0]
    assert "南京" in live_enabled["prompts"][0]


def test_a_long_live_answer_is_not_truncated_like_a_short_snippet(live_enabled) -> None:
    """The answer is the whole material, not one snippet among several.

    Cutting it at the snippet limit removed half the places from a listed answer.
    """
    long_answer = LIVE_ANSWER * 12
    assert len(long_answer) > recommendations.SNIPPET_CHARS
    live_enabled["answers"].clear()
    live_enabled["answers"].append(long_answer)

    material = recommendations._live_category_material("南京", "attraction")

    assert material is not None
    assert recommendations.LIVE_MATERIAL_CHARS > recommendations.SNIPPET_CHARS
    assert len(material[0]["snippet"]) > recommendations.SNIPPET_CHARS


def test_live_search_answers_the_breakfast_question_for_a_breakfast_batch(live_enabled) -> None:
    """A breakfast batch must ask about breakfast shops, not about restaurants."""
    recommendations._category_material("南京", "restaurant", "早餐 早点 小吃 人均")

    assert "早餐" in live_enabled["prompts"][0]


def test_the_live_path_is_skipped_when_it_is_unavailable(monkeypatch) -> None:
    """A disabled or unconfigured live search must not change the existing behaviour."""
    monkeypatch.setenv("DISABLE_LIVE_WEB_SEARCH", "1")
    answers: list[str] = []
    monkeypatch.setattr(dashscope_search, "_ask", lambda *a, **k: answers.append(1) or (LIVE_ANSWER, "stop"))
    monkeypatch.setattr(
        recommendations,
        "search_snippets",
        lambda query, limit=3, timeout=8.0: ["外滩适合傍晚漫步，夜景尤其值得一看。南京路步行街就在附近，可以一起安排。"],
    )

    material = recommendations._category_material("上海", "attraction")

    # Nothing was asked, and the snippet path supplied the material instead.
    assert answers == []
    assert material[0]["snippet"] == "外滩适合傍晚漫步，夜景尤其值得一看。南京路步行街就在附近，可以一起安排。"


def test_the_live_path_falls_back_when_the_answer_is_unusable(monkeypatch) -> None:
    """An empty or failed live answer must not leave the category with no material."""
    monkeypatch.delenv("DISABLE_LIVE_WEB_SEARCH", raising=False)
    monkeypatch.setattr(dashscope_search, "_ask", lambda prompt, max_tokens=0, timeout=None: ("", "stop"))
    monkeypatch.setattr(
        recommendations,
        "search_snippets",
        lambda query, limit=3, timeout=8.0: ["外滩适合傍晚漫步，夜景尤其值得一看。南京路步行街就在附近，可以一起安排。"],
    )

    material = recommendations._category_material("上海", "attraction")

    assert material[0]["snippet"] == "外滩适合傍晚漫步，夜景尤其值得一看。南京路步行街就在附近，可以一起安排。"


def test_search_available_respects_the_off_switch(monkeypatch) -> None:
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.delenv("DISABLE_LIVE_WEB_SEARCH", raising=False)
    assert dashscope_search.search_available() is True

    monkeypatch.setenv("DISABLE_LIVE_WEB_SEARCH", "1")
    assert dashscope_search.search_available() is False


def test_a_failed_live_call_is_logged_and_returns_nothing(monkeypatch) -> None:
    """A provider failure must degrade to "no material" rather than raising."""
    monkeypatch.delenv("DISABLE_LIVE_WEB_SEARCH", raising=False)

    class _Boom:
        class chat:  # noqa: N801 - stands in for the SDK's nested namespace
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs: Any):
                    raise RuntimeError("provider exploded")

    monkeypatch.setattr(dashscope_search, "search_client", lambda timeout=None: _Boom())

    assert dashscope_search._ask("南京景点") is None


def test_an_empty_live_answer_is_treated_as_no_material(monkeypatch) -> None:
    monkeypatch.delenv("DISABLE_LIVE_WEB_SEARCH", raising=False)

    class _Empty:
        class chat:  # noqa: N801
            class completions:  # noqa: N801
                @staticmethod
                def create(**kwargs: Any):
                    class _Message:
                        content = "   "

                    class _Choice:
                        message = _Message()
                        finish_reason = "stop"

                    class _Response:
                        choices = [_Choice()]

                    return _Response()

    monkeypatch.setattr(dashscope_search, "search_client", lambda timeout=None: _Empty())

    assert dashscope_search._ask("南京景点") is None


def test_live_search_disables_thinking_without_disabling_search(monkeypatch) -> None:
    seen: list[dict[str, Any]] = []

    class _Completions:
        @staticmethod
        def create(**kwargs: Any):
            seen.append(kwargs)
            message = type("_Message", (), {"content": LIVE_ANSWER})()
            choice = type("_Choice", (), {"message": message, "finish_reason": "stop"})()
            return type("_Response", (), {"choices": [choice]})()

    client = type("_Client", (), {"chat": type("_Chat", (), {"completions": _Completions()})()})()
    monkeypatch.setattr(dashscope_search, "search_client", lambda timeout=None: client)

    assert dashscope_search._ask("南京景点") is not None
    assert seen[0]["extra_body"] == {"enable_search": True, "enable_thinking": False}


def test_a_lookup_can_bound_its_own_search(monkeypatch) -> None:
    """A caller's timeout must reach the client instead of being dropped.

    The live search bounds itself at 120s, which is right for planning material and absurd
    for the one-line answer a price lookup wants. Callers passed 8.0s and the value went
    nowhere, so a single stalled lookup could hold a price phase open for two minutes.
    """
    captured: list[float | None] = []

    monkeypatch.setattr(dashscope_search, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    class _FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.append(kwargs["timeout"])

    monkeypatch.setattr(dashscope_search, "OpenAI", _FakeOpenAI)

    dashscope_search.search_client(7.5)
    dashscope_search.search_client()

    # An unbounded caller keeps the planning bound; nobody is left without one.
    assert captured == [7.5, dashscope_search.SEARCH_TIMEOUT]


def test_search_snippets_passes_its_timeout_to_the_live_search(monkeypatch) -> None:
    monkeypatch.delenv("DISABLE_LIVE_WEB_SEARCH", raising=False)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    seen: list[float | None] = []

    def _grounded(query: str, min_chars: int = 0, timeout: float | None = None) -> str:
        seen.append(timeout)
        return "人均 120 元"

    monkeypatch.setattr(dashscope_search, "grounded_answer", _grounded)

    assert web_search.search_snippets("成都 陈麻婆豆腐 人均", timeout=12.0) == ["人均 120 元"]
    assert seen == [12.0]


def test_search_web_prefers_the_live_answer(live_enabled, monkeypatch) -> None:
    """The plan's own search uses the live answer, and there is no other provider to call."""
    assert not hasattr(web_search, "httpx")

    results, status = web_search.search_web("推荐景点和美食", "南京")

    assert status == "completed"
    assert len(results) == 1
    assert results[0]["source_type"] == "web_answer"
    assert "中山陵" in results[0]["snippet"]
    # No source links: the compatible-mode response exposes only the finished answer, and
    # raw sources must never reach the traveller anyway.
    assert results[0]["url"] is None


def test_the_live_answer_heading_is_never_shown_as_a_place(live_enabled) -> None:
    """The answer's own title must not survive as a recommendation card name."""
    results, _status = web_search.search_web("推荐景点", "南京")

    assert recommendations._is_descriptive_name(results[0]["title"]) is True


def test_search_web_reports_a_fallback_when_live_search_is_off(monkeypatch) -> None:
    """With live search off there is no provider to fall back to, and the status says so.

    The metered search API was removed: it answered HTTP 432 to every query, so the
    "fallback" it provided was an empty list with a delay attached.
    """
    monkeypatch.setenv("DISABLE_LIVE_WEB_SEARCH", "1")
    assert not hasattr(web_search, "httpx")

    results, status = web_search.search_web("推荐景点", "南京")

    assert results == []
    assert status == "fallback: live search is not configured"


def test_transport_advice_accepts_the_live_answer() -> None:
    """The transit recommendation reads whichever written answer the search produced."""
    state = {
        "route_context": {"hotel_name": "某酒店", "hotel_address": "某地址", "destination": "南京南站"},
        "web_results": [{"source_type": "web_answer", "snippet": "地铁 3 号线约 25 分钟。"}],
    }

    advice = planner._transport_advice(state)

    assert advice is not None
    assert advice["recommendation"] == "地铁 3 号线约 25 分钟。"


def test_a_trip_question_is_asked_as_a_question_not_keywords() -> None:
    """A searching model retrieves better for a question than for a keyword list."""
    assert "景点" in recommendations._live_search_question("attraction", None)
    assert "早餐" in recommendations._live_search_question("restaurant", "早餐 早点 小吃 人均")


def test_a_capability_disclaimer_is_stripped_from_the_answer(monkeypatch) -> None:
    """One real answer opened by denying a capability it actually had.

    The sentence was false — the search had run — and any fact printed next to it looks
    untrustworthy, so it must not reach the material.
    """
    monkeypatch.delenv("DISABLE_LIVE_WEB_SEARCH", raising=False)
    answer = (
        "经核查，截至2024年7月，我无法实时联网检索或访问互联网。"
        "根据我的知识截止时间，现为您规划一份方案：\n"
        "| 酒店 | 价格 | 位置 |\n"
        "| 南京中心诺富特酒店 | 1880元 | 新街口地铁站旁 |\n"
        "| 南京涵碧楼酒店 | 2600元 | 临江而建 |"
    )
    monkeypatch.setattr(dashscope_search, "_ask", lambda prompt, max_tokens=0, timeout=None: (answer, "stop"))

    cleaned = dashscope_search.grounded_answer("南京住宿")

    assert cleaned is not None
    assert "无法实时联网" not in cleaned
    assert "知识截止" not in cleaned
    # The facts that came after the disclaimer survive, table rows included.
    assert "南京中心诺富特酒店" in cleaned
    assert "1880元" in cleaned


def test_the_prompt_tells_the_model_not_to_disclaim(monkeypatch) -> None:
    """The instruction is cheaper than repairing the answer afterwards."""
    monkeypatch.delenv("DISABLE_LIVE_WEB_SEARCH", raising=False)
    prompts: list[str] = []
    monkeypatch.setattr(
        dashscope_search,
        "_ask",
        lambda prompt, max_tokens=0, timeout=None: prompts.append(prompt) or (LIVE_ANSWER, "stop"),
    )

    dashscope_search.grounded_answer("南京景点")

    assert "不要声明自己无法联网" in prompts[0]


def test_an_answer_that_is_only_a_disclaimer_is_rejected(monkeypatch) -> None:
    """With nothing behind the disclaimer there is no material, so the caller falls back."""
    monkeypatch.delenv("DISABLE_LIVE_WEB_SEARCH", raising=False)
    monkeypatch.setattr(
        dashscope_search,
        "_ask",
        lambda prompt, max_tokens=0, timeout=None: ("我无法实时联网检索，请以官方信息为准。", "stop"),
    )

    assert dashscope_search.grounded_answer("南京景点") is None
