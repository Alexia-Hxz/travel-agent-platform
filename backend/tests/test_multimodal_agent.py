from io import BytesIO
from uuid import uuid4

from fastapi.testclient import TestClient

from app.main import app


def _trip_payload() -> dict[str, object]:
    return {
        "title": f"vision test {uuid4().hex[:8]}",
        "destination": "Hangzhou",
        "start_date": "2026-10-01",
        "end_date": "2026-10-03",
        "budget": 3200,
    }


def test_normalizes_visual_model_json_into_place_and_itinerary_suggestions() -> None:
    from app.multimodal.schemas import normalize_vision_result

    result = normalize_vision_result(
        {
            "document_type": "hotel_booking",
            "summary": "West Lake Hotel booking",
            "place": {"name": "West Lake Hotel", "type": "hotel", "address": "88 Lakeside Road"},
            "itinerary": {"date": "2026-10-02", "start_time": "14:00", "end_time": "15:00", "note": "Check in"},
            "reference_number": "AB123456",
        }
    )

    assert result.document_type == "hotel_booking"
    assert result.place.name == "West Lake Hotel"
    assert result.place.type == "hotel"
    assert result.itinerary.date.isoformat() == "2026-10-02"
    assert result.reference_number == "AB123456"


def test_normalizes_partial_model_output_without_discarding_the_summary() -> None:
    from app.multimodal.schemas import normalize_vision_result

    result = normalize_vision_result(
        {
            "document_type": "hotel_booking",
            "summary": "识别到酒店预订截图",
            "place": {"name": "初心启程酒店", "type": "hotel"},
            "itinerary": None,
        }
    )

    assert result.summary == "识别到酒店预订截图"
    assert result.place is None


def test_decodes_json_wrapped_in_a_markdown_code_fence() -> None:
    from app.services.vision_service import decode_model_json

    assert decode_model_json("```json\n{\"summary\": \"ok\"}\n```") == {"summary": "ok"}


def test_normalizes_model_price_text_without_failing_the_preview() -> None:
    from app.multimodal.schemas import normalize_vision_result

    result = normalize_vision_result({
        "document_type": "hotel_booking",
        "summary": "酒店截图",
        "place": {"name": "测试酒店", "type": "hotel", "address": "测试路", "price_level": "¥129起"},
    })
    assert result.place is not None
    assert result.place.price_level is None


def test_normalizes_chinese_place_type_and_discards_an_invalid_itinerary() -> None:
    from app.multimodal.schemas import normalize_vision_result

    result = normalize_vision_result(
        {
            "document_type": "酒店预订",
            "summary": "识别到酒店订单",
            "place": {"name": "初心启程酒店", "type": "酒店", "address": "浦东新区界龙大道266号"},
            "itinerary": {
                "date": "2023-09-07",
                "start_time": "00:00",
                "end_time": "2023-09-08",
                "note": "一晚",
            },
        }
    )

    assert result.document_type == "hotel_booking"
    assert result.place is not None
    assert result.place.type == "hotel"
    assert result.itinerary is None


def test_vision_preview_and_confirm_save_user_approved_data(monkeypatch) -> None:
    from app.services import vision_service

    monkeypatch.setattr(
        vision_service,
        "analyze_file",
        lambda *_args, **_kwargs: vision_service.VisionAnalysis(
            document_type="attraction_photo",
            summary="A museum visit",
            place={"name": "West Lake Museum", "type": "attraction", "address": "Museum Road"},
            itinerary={"date": "2026-10-02", "start_time": "09:00", "end_time": "11:00", "note": "Morning visit"},
            reference_number=None,
            raw_text="Museum Road",
        ),
    )

    with TestClient(app) as client:
        trip = client.post("/api/v1/trips", json=_trip_payload()).json()
        uploaded = client.post(
            f"/api/v1/trips/{trip['id']}/files",
            files={"file": ("museum.jpg", BytesIO(b"image"), "image/jpeg")},
        ).json()

        preview = client.post(f"/api/v1/trips/{trip['id']}/files/{uploaded['id']}/vision/preview")
        assert preview.status_code == 200
        assert preview.json()["place"]["name"] == "West Lake Museum"

        confirmation = client.post(
            f"/api/v1/trips/{trip['id']}/files/{uploaded['id']}/vision/confirm",
            json={"save_place": True, "save_itinerary": True, "place": preview.json()["place"], "itinerary": preview.json()["itinerary"]},
        )
        assert confirmation.status_code == 201
        body = confirmation.json()
        assert body["place"]["name"] == "West Lake Museum"
        assert body["itinerary"]["place_id"] == body["place"]["id"]


def test_vision_confirmation_does_not_save_place_when_itinerary_date_is_outside_trip() -> None:
    with TestClient(app) as client:
        trip = client.post("/api/v1/trips", json=_trip_payload()).json()
        uploaded = client.post(
            f"/api/v1/trips/{trip['id']}/files",
            files={"file": ("hotel.jpg", BytesIO(b"image"), "image/jpeg")},
        ).json()

        response = client.post(
            f"/api/v1/trips/{trip['id']}/files/{uploaded['id']}/vision/confirm",
            json={
                "save_place": True,
                "save_itinerary": True,
                "place": {"name": "Out of range hotel", "type": "hotel", "address": "Test address"},
                "itinerary": {
                    "date": "2026-10-10",
                    "start_time": "14:00",
                    "end_time": "15:00",
                    "note": "Check in",
                },
            },
        )

        assert response.status_code == 422
        assert client.get(f"/api/v1/trips/{trip['id']}/places").json() == []


def test_vision_preview_accepts_octet_stream_for_image_extension(monkeypatch) -> None:
    from app.services import vision_service

    monkeypatch.setattr(
        vision_service,
        "analyze_file",
        lambda path, content_type: vision_service.VisionAnalysis(
            document_type="attraction_photo", summary="ok", place=None,
            itinerary=None, reference_number=None, raw_text=None,
        ),
    )
    with TestClient(app) as client:
        trip = client.post("/api/v1/trips", json=_trip_payload()).json()
        uploaded = client.post(
            f"/api/v1/trips/{trip['id']}/files",
            files={"file": ("photo.jpg", BytesIO(b"image"), "application/octet-stream")},
        ).json()
        response = client.post(f"/api/v1/trips/{trip['id']}/files/{uploaded['id']}/vision/preview")
        assert response.status_code == 200


def test_vision_preview_keeps_a_hotel_but_discards_an_itinerary_outside_trip_dates(monkeypatch) -> None:
    from app.services import vision_service

    monkeypatch.setattr(
        vision_service,
        "analyze_file",
        lambda *_args, **_kwargs: vision_service.VisionAnalysis(
            document_type="hotel_booking",
            summary="识别到酒店订单",
            place={"name": "初心启程酒店", "type": "hotel", "address": "浦东新区界龙大道266号"},
            itinerary={"date": "2023-09-07", "start_time": "00:00", "end_time": "23:59", "note": "一晚"},
            reference_number=None,
            raw_text="酒店订单",
        ),
    )
    with TestClient(app) as client:
        trip = client.post("/api/v1/trips", json=_trip_payload()).json()
        uploaded = client.post(
            f"/api/v1/trips/{trip['id']}/files",
            files={"file": ("hotel.jpg", BytesIO(b"image"), "image/jpeg")},
        ).json()

        response = client.post(f"/api/v1/trips/{trip['id']}/files/{uploaded['id']}/vision/preview")

    assert response.status_code == 200
    assert response.json()["place"]["name"] == "初心启程酒店"
    assert response.json()["itinerary"] is None
