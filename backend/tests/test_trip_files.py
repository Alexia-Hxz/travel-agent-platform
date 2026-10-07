from io import BytesIO
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.main import app


def _trip_payload() -> dict[str, object]:
    suffix = uuid4().hex[:8]
    return {
        "title": f"file upload test {suffix}",
        "destination": "Hangzhou",
        "start_date": "2026-10-01",
        "end_date": "2026-10-03",
        "budget": 3200,
    }


def _create_trip(client: TestClient) -> int:
    return client.post("/api/v1/trips", json=_trip_payload()).json()["id"]


@pytest.mark.parametrize(
    ("filename", "content_type"),
    [
        ("ticket.jpg", "image/jpeg"),
        ("ticket.JPEG", "image/jpeg"),
        ("ticket.png", "image/png"),
        ("ticket.webp", "image/webp"),
        ("ticket.pdf", "application/pdf"),
    ],
)
def test_uploads_each_allowed_file_type_and_persists_metadata(
    filename: str, content_type: str
) -> None:
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        response = client.post(
            f"/api/v1/trips/{trip_id}/files",
            files={"file": (filename, BytesIO(b"test file"), content_type)},
        )

        assert response.status_code == 201
        uploaded = response.json()
        assert uploaded["trip_id"] == trip_id
        assert uploaded["original_filename"] == filename
        assert uploaded["stored_filename"].endswith(Path(filename).suffix.lower())
        assert uploaded["content_type"] == content_type
        assert uploaded["file_size"] == 9
        assert uploaded["file_path"].startswith(f"data/uploads/{trip_id}/")
        assert uploaded["created_at"]

        assert client.delete(f"/api/v1/trips/{trip_id}/files/{uploaded['id']}").status_code == 204


def test_rejects_an_unsupported_file_type() -> None:
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        response = client.post(
            f"/api/v1/trips/{trip_id}/files",
            files={"file": ("notes.txt", BytesIO(b"not supported"), "text/plain")},
        )

    assert response.status_code == 400
    assert "supported" in response.json()["detail"].lower()


def test_accepts_a_jpeg_when_multipart_filename_loses_its_extension() -> None:
    """Some clients mangle non-ASCII filenames; the MIME type still identifies the JPEG."""
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        response = client.post(
            f"/api/v1/trips/{trip_id}/files",
            files={"file": ("____________", BytesIO(b"fake-jpeg"), "image/jpeg")},
        )

    assert response.status_code == 201
    assert response.json()["stored_filename"].endswith(".jpg")


def test_rejects_a_file_larger_than_the_limit() -> None:
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        response = client.post(
            f"/api/v1/trips/{trip_id}/files",
            files={"file": ("large.pdf", BytesIO(b"x" * (10 * 1024 * 1024 + 1)), "application/pdf")},
        )

    assert response.status_code == 400
    assert "10 MiB" in response.json()["detail"]


def test_lists_files_then_retrieves_and_downloads_one_file() -> None:
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        uploaded = client.post(
            f"/api/v1/trips/{trip_id}/files",
            files={"file": ("booking.pdf", BytesIO(b"pdf-content"), "application/pdf")},
        ).json()

        listed = client.get(f"/api/v1/trips/{trip_id}/files")
        retrieved = client.get(f"/api/v1/trips/{trip_id}/files/{uploaded['id']}")
        downloaded = client.get(f"/api/v1/trips/{trip_id}/files/{uploaded['id']}/download")

        assert listed.status_code == 200
        assert [item["id"] for item in listed.json()] == [uploaded["id"]]
        assert retrieved.status_code == 200
        assert retrieved.json()["stored_filename"] == uploaded["stored_filename"]
        assert downloaded.status_code == 200
        assert downloaded.content == b"pdf-content"
        assert "booking.pdf" in downloaded.headers["content-disposition"]

        assert client.delete(f"/api/v1/trips/{trip_id}/files/{uploaded['id']}").status_code == 204
        assert client.get(f"/api/v1/trips/{trip_id}/files").json() == []


def test_rejects_upload_to_a_nonexistent_trip() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/trips/999999/files",
            files={"file": ("ticket.pdf", BytesIO(b"file"), "application/pdf")},
        )

    assert response.status_code == 404


def test_parses_an_uploaded_file_and_returns_structured_travel_details(monkeypatch: pytest.MonkeyPatch) -> None:
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        uploaded = client.post(
            f"/api/v1/trips/{trip_id}/files",
            files={"file": ("booking.png", BytesIO(b"image"), "image/png")},
        ).json()

        monkeypatch.setattr(
            "app.services.file_parser.extract_text",
            lambda _: "Hotel: West Lake Hotel\nDate: 2026-10-02\nTime: 14:30\nAddress: 88 Lakeside Road\nOrder No: AB123456",
        )
        response = client.post(f"/api/v1/trips/{trip_id}/files/{uploaded['id']}/parse")
        status_response = client.get(f"/api/v1/trips/{trip_id}/files/{uploaded['id']}/parse")
        result_response = client.get(f"/api/v1/trips/{trip_id}/files/{uploaded['id']}/parse/result")

    assert response.status_code == 200
    assert response.json()["parse_status"] == "completed"
    assert "West Lake Hotel" in response.json()["parsed_text"]
    assert response.json()["parsed_result"]["hotel_names"] == ["West Lake Hotel"]
    assert response.json()["parsed_result"]["dates"] == ["2026-10-02"]
    assert response.json()["parsed_result"]["times"] == ["14:30"]
    assert response.json()["parsed_result"]["addresses"] == ["88 Lakeside Road"]
    assert response.json()["parsed_result"]["reference_numbers"] == ["AB123456"]
    assert status_response.json()["parse_status"] == "completed"
    assert result_response.json()["parsed_result"]["hotel_names"] == ["West Lake Hotel"]


def test_rejects_parse_for_non_image_or_pdf_file() -> None:
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        uploaded = client.post(
            f"/api/v1/trips/{trip_id}/files",
            files={"file": ("ticket.pdf", BytesIO(b"pdf"), "application/pdf")},
        ).json()

        response = client.post(f"/api/v1/trips/{trip_id}/files/{uploaded['id']}/parse")

    # A malformed PDF is a parsing failure, not an unsupported upload type.
    assert response.status_code in {400, 422, 503}


def test_deleting_a_trip_removes_its_file_records_and_local_files() -> None:
    with TestClient(app) as client:
        trip_id = _create_trip(client)
        uploaded = client.post(
            f"/api/v1/trips/{trip_id}/files",
            files={"file": ("receipt.png", BytesIO(b"receipt"), "image/png")},
        ).json()
        stored_path = Path(__file__).resolve().parents[2] / uploaded["file_path"]

        assert stored_path.is_file()
        assert client.delete(f"/api/v1/trips/{trip_id}").status_code == 204
        assert not stored_path.exists()
        assert not stored_path.parent.exists()
        assert client.get(f"/api/v1/trips/{trip_id}/files").status_code == 404
