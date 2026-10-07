import re
import os
from datetime import datetime
from pathlib import Path

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.trip_file import TripFile
from app.services.file_service import resolve_file_path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class ParseFailure(Exception):
    pass


def extract_structured_result(text: str) -> dict[str, list[str]]:
    result = {key: [] for key in ("hotel_names", "attraction_names", "transport_hubs", "dates", "times", "addresses", "reference_numbers")}
    labels = {"hotel_names": r"hotel(?: name)?|酒店(?:名称)?", "attraction_names": r"attraction(?: name)?|景点(?:名称)?", "transport_hubs": r"airport|station|机场|车站", "addresses": r"address|地址", "reference_numbers": r"order(?: no| number)?|ticket(?: no| number)?|订单号|票号"}
    for key, label in labels.items():
        result[key] = [v.strip() for v in re.findall(rf"(?im)^\s*(?:{label})\s*[:#]?\s*(.+?)\s*$", text)]
    result["dates"] = list(dict.fromkeys(re.findall(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", text)))
    result["times"] = list(dict.fromkeys(re.findall(r"\b(?:[01]?\d|2[0-3]):[0-5]\d\b", text)))
    return result


def extract_text(file_path: Path) -> str:
    if file_path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
            return "\n".join(page.extract_text() or "" for page in PdfReader(str(file_path)).pages)
        except ImportError as error:
            raise ParseFailure("PDF parsing support is not installed.") from error
        except Exception as error:
            raise ParseFailure(f"PDF text extraction failed: {error}") from error
    if file_path.suffix.lower() in IMAGE_EXTENSIONS:
        try:
            project_data = Path(__file__).resolve().parents[3] / "data"
            os.environ.setdefault("PADDLE_HOME", str(project_data / "paddle"))
            os.environ.setdefault("PADDLE_OCR_BASE_DIR", str(project_data / "paddleocr"))
            # Load SciPy before PaddlePaddle's native runtime to avoid a zlib
            # symbol collision between their bundled binary dependencies.
            import scipy  # noqa: F401
            from paddleocr import PaddleOCR
            # Paddle's oneDNN fused convolution can fail on some Windows CPU builds.
            model_root = project_data / "paddleocr" / "whl"
            ocr = PaddleOCR(
                use_angle_cls=True,
                lang="en",
                det_model_dir=str(model_root / "det" / "en" / "en_PP-OCRv3_det_infer"),
                rec_model_dir=str(model_root / "rec" / "en" / "en_PP-OCRv4_rec_infer"),
                cls_model_dir=str(model_root / "cls" / "ch_ppocr_mobile_v2.0_cls_infer"),
                enable_mkldnn=False,
                ir_optim=False,
            )
            return "\n".join(line[1][0] for page in ocr.ocr(str(file_path), cls=True) for line in (page or []))
        except ImportError as error:
            raise ParseFailure("PaddleOCR is not installed. Install the OCR runtime before parsing images.") from error
        except Exception as error:
            raise ParseFailure(
                "OCR failed while loading or running the recognition model. "
                "Please retry after restarting the backend."
            ) from error
    raise ParseFailure("Only JPG, JPEG, PNG, WEBP, and PDF files can be parsed.")


def parse_trip_file(session: Session, trip_file: TripFile) -> TripFile:
    if trip_file.file_path.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".pdf")) is False:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="This file type cannot be parsed.")
    trip_file.parse_status = "processing"
    session.commit()
    try:
        trip_file.parsed_text = extract_text(resolve_file_path(trip_file))
        trip_file.parsed_result = extract_structured_result(trip_file.parsed_text)
        trip_file.parse_status = "completed"
        trip_file.parse_error = None
        trip_file.parsed_at = datetime.utcnow()
    except ParseFailure as error:
        trip_file.parse_status = "failed"
        trip_file.parse_error = str(error)
        session.commit()
        raise HTTPException(status_code=422, detail=trip_file.parse_error) from error
    session.commit(); session.refresh(trip_file)
    return trip_file
