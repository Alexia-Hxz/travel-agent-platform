"""Every model client must bound how long it can block a request.

The OpenAI SDK defaults to a 10 minute timeout with automatic retries. A single
stalled provider call would therefore hold a planning request open for minutes,
which the reverse proxy answers with a 504 and the user reads as "cannot generate
a plan". These tests pin the guard that prevents that.
"""

import inspect

from app.agents.tools import llm, plan_summary
from app.agents.workflows import adjuster
from app.services import vision_service


def _openai_calls_in(module) -> list[str]:
    return [
        line.strip()
        for line in inspect.getsource(module).splitlines()
        if "OpenAI(" in line
    ]


def test_llm_client_sets_a_timeout_and_bounded_retries(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeOpenAI:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

    monkeypatch.setattr(llm, "OpenAI", _FakeOpenAI)
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")

    client = llm.model_client()

    assert client is not None
    assert captured["timeout"] == llm.REQUEST_TIMEOUT
    assert captured["max_retries"] == llm.MAX_RETRIES
    assert llm.REQUEST_TIMEOUT > 0


def test_llm_client_returns_none_without_a_key(monkeypatch) -> None:
    # model_client re-reads .env from disk, so the loader has to be neutralized too.
    monkeypatch.setattr(llm, "load_dotenv", lambda *args, **kwargs: None)
    monkeypatch.delenv("DASHSCOPE_API_KEY", raising=False)

    assert llm.model_client() is None


def test_every_project_module_bounds_its_openai_client() -> None:
    """Guard against a future client being added without a timeout."""
    for module in (llm, plan_summary, adjuster, vision_service):
        source = inspect.getsource(module)
        assert "timeout=" in source, f"{module.__name__} builds an OpenAI client without a timeout"
        assert "max_retries=" in source, f"{module.__name__} builds an OpenAI client without bounded retries"


def test_vision_timeout_is_looser_than_a_text_call() -> None:
    # Image analysis is slower, but still bounded.
    assert vision_service.VISION_REQUEST_TIMEOUT > llm.REQUEST_TIMEOUT
