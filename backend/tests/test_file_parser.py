import os
import sys
from pathlib import Path
from types import SimpleNamespace

from app.services.file_parser import extract_text


def test_ocr_runtime_dependencies_can_import() -> None:
    import paddle
    import paddleocr

    assert paddle.__version__ == "2.6.2"
    assert paddleocr.__version__ == "2.9.1"


def test_image_ocr_uses_project_local_model_cache(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakePaddleOCR:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def ocr(self, *_args, **_kwargs):
            return []

    monkeypatch.delenv("PADDLE_OCR_BASE_DIR", raising=False)
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PaddleOCR=FakePaddleOCR))

    assert extract_text(Path("receipt.jpg")) == ""
    assert Path(os.environ["PADDLE_OCR_BASE_DIR"]).name == "paddleocr"
    assert captured["enable_mkldnn"] is False
    assert captured["ir_optim"] is False
    assert Path(captured["det_model_dir"]).name == "en_PP-OCRv3_det_infer"
    assert Path(captured["rec_model_dir"]).name == "en_PP-OCRv4_rec_infer"
    assert Path(captured["cls_model_dir"]).name == "ch_ppocr_mobile_v2.0_cls_infer"
