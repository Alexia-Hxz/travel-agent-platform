import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from app.agents.tools.llm import MAX_RETRIES
from app.multimodal.schemas import VisionPreview, normalize_vision_result

# Image analysis needs more room than a text call, but it must still be bounded.
VISION_REQUEST_TIMEOUT = 60.0


class VisionConfigurationError(Exception):
    pass


class VisionProviderError(Exception):
    pass


@dataclass(frozen=True)
class VisionAnalysis:
    document_type: str
    summary: str
    place: dict[str, Any] | None
    itinerary: dict[str, Any] | None
    reference_number: str | None
    raw_text: str | None


VISION_PROMPT = """You extract travel details from one image. Return only JSON with these keys:
document_type (attraction_photo, hotel_booking, transport_ticket, menu, travel_guide, other),
summary, place, itinerary, reference_number, raw_text.
place must be null or {name,type,address,latitude,longitude,image_url,rating,price_level,opening_hours,source_url};
type must be attraction, hotel, restaurant, airport, or station. Use null for unknown fields.
itinerary must be null or {date,start_time,end_time,note}; dates use YYYY-MM-DD and times use HH:MM.
Do not guess facts that are absent from the image. Keep raw_text short and only include legible travel-relevant text."""


def _client_from_environment() -> OpenAI:
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise VisionConfigurationError("Multimodal analysis is not configured.")
    return OpenAI(
        api_key=api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        # Image analysis is slower than a text call, but it still must be bounded.
        timeout=VISION_REQUEST_TIMEOUT,
        max_retries=MAX_RETRIES,
    )


def decode_model_json(content: str) -> dict[str, Any]:
    """Decode strict JSON plus the fenced JSON format often returned by chat models."""
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("The vision model response must be a JSON object.")
    return parsed


def analyze_file(file_path: Path, content_type: str) -> VisionAnalysis:
    if not content_type.startswith("image/"):
        raise VisionProviderError("Only image files can be analyzed by the multimodal agent.")
    try:
        encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
        response = _client_from_environment().chat.completions.create(
            model="qwen3-vl-flash",
            messages=[
                {"role": "system", "content": "You are a careful travel-document extraction assistant."},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:{content_type};base64,{encoded}"}},
                    ],
                },
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        content = response.choices[0].message.content or "{}"
        preview: VisionPreview = normalize_vision_result(decode_model_json(content))
        return VisionAnalysis(**preview.model_dump(mode="json"))
    except (OSError, json.JSONDecodeError, ValueError, KeyError) as error:
        raise VisionProviderError("The vision model returned an invalid analysis result.") from error
    except Exception as error:
        raise VisionProviderError("Multimodal analysis failed. Please retry later.") from error
