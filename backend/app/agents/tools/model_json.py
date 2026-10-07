"""Reading a JSON object out of a chat completion.

Models wrap JSON in prose or a Markdown fence often enough that the parsing is worth having in one
place. Two tools need it now (recommendation prices and the budget split), and a second copy of the
same six regex-and-slice lines is how the two drift apart.
"""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(content: Any) -> dict[str, Any] | None:
    """The first JSON object in ``content``, or None when there is not one.

    Tolerates a ```json fence and surrounding explanation. Never raises.
    """
    text = str(content or "").strip()
    if not text:
        return None
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None
