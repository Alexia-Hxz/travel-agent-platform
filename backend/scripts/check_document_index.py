"""End-to-end check of the document index against the REAL embedding provider.

The pytest suite stubs embeddings (see ``tests/conftest.py::_offline_embeddings``) so it
stays offline, fast and deterministic. That means the suite can prove the storage, the
trip filter and the indexing lifecycle, but never that the real ``text-embedding-v4`` path
works. This script covers exactly that gap, and it is the check to run after changing the
embedding model, the chunking, or the collection settings.

It uses a throwaway database and vector directory, so it never touches your own trips.

Usage (from ``backend/``, with DASHSCOPE_API_KEY set in ``backend/.env``):

    .venv-ocr\\Scripts\\python.exe scripts\\check_document_index.py

Exit code 0 means every check passed. This does call the network and does consume a small
amount of embedding quota.
"""

import os
import shutil
import sys
import tempfile
from datetime import date
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_ROOT))

TEMP_DIR = tempfile.mkdtemp(prefix="document-index-check-")
os.environ["TRAVEL_AGENT_DB_PATH"] = os.path.join(TEMP_DIR, "check.db")
os.environ["TRAVEL_AGENT_CHROMA_PATH"] = os.path.join(TEMP_DIR, "chroma")
os.environ["DISABLE_LIVE_WEB_SEARCH"] = "1"

from app.db.session import Base, SessionLocal, engine  # noqa: E402
from app.models import Trip, TripFile  # noqa: E402,F401
from app.services import vector_service  # noqa: E402


FAILURES: list[str] = []

ORDER = """杭州西湖宾馆 预订确认单
订单号: AB123456
入住日期: 2026-10-02
离店日期: 2026-10-05
房型: 豪华大床房
地址: 杭州市西湖区湖滨路88号
电话: 0571-88888888
含早餐 2 份
入住人: 张伟
备注: 需要无烟房，延迟退房至 14:00
"""

TICKET = """东京成田机场 电子客票行程单
票号: JP998877
出发日期: 2026-11-11 09:35
航班: NH920
舱位: 经济舱
乘客: 张伟
"""


def step(number: int, message: str) -> None:
    print(f"[{number:02d}] {message}", flush=True)


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def _file(trip_id: int, name: str, text: str, stored: str) -> TripFile:
    return TripFile(
        trip_id=trip_id,
        original_filename=name,
        stored_filename=stored,
        content_type="application/pdf",
        file_size=len(text.encode("utf-8")),
        file_path=f"data/uploads/{stored}",
        parse_status="completed",
        parsed_text=text,
    )


def run_checks(session) -> None:
    trip_a = Trip(title="index check A", destination="杭州", start_date=date(2026, 10, 1), end_date=date(2026, 10, 5), budget=8000)
    trip_b = Trip(title="index check B", destination="东京", start_date=date(2026, 11, 10), end_date=date(2026, 11, 13), budget=9000)
    session.add_all([trip_a, trip_b])
    session.commit()
    session.refresh(trip_a)
    session.refresh(trip_b)

    file_a = _file(trip_a.id, "杭州酒店订单.pdf", ORDER, "check-a.pdf")
    file_b = _file(trip_b.id, "东京机票.pdf", TICKET, "check-b.pdf")
    session.add_all([file_a, file_b])
    session.commit()
    session.refresh(file_a)
    session.refresh(file_b)

    outcome_a = vector_service.index_trip_file(session, file_a)
    step(2, f"index the hotel order -> {outcome_a.status}, {outcome_a.chunk_count} chunk(s), reason={outcome_a.reason}")
    check(outcome_a.status == "indexed", f"the hotel order was not indexed: {outcome_a.reason}")

    outcome_b = vector_service.index_trip_file(session, file_b)
    step(3, f"index the flight ticket -> {outcome_b.status}, {outcome_b.chunk_count} chunk(s), reason={outcome_b.reason}")
    check(outcome_b.status == "indexed", f"the flight ticket was not indexed: {outcome_b.reason}")

    question = "我订的酒店什么时候入住"
    found = vector_service.search_trip_documents(trip_a.id, question)
    step(4, f"ask {question!r} -> status={found.status}, {len(found.chunks)} hit(s)")
    for chunk in found.chunks:
        print(f"       similarity={chunk.similarity:.4f} file_id={chunk.file_id} text={chunk.text.splitlines()[0][:26]!r}", flush=True)
    check(found.status == "ok", f"the search did not succeed: {found.status} / {found.reason}")
    if found.chunks:
        check(found.chunks[0].file_id == file_a.id, f"top-1 came from file {found.chunks[0].file_id}, expected {file_a.id}")
        check("2026-10-02" in found.chunks[0].text, f"top-1 does not carry the check-in date: {found.chunks[0].text!r}")

    # Trip isolation: Chroma has no partitions, so only the filter keeps these apart.
    cross = vector_service.search_trip_documents(trip_a.id, "东京 航班 出发日期 舱位 乘客")
    ids = sorted({chunk.file_id for chunk in cross.chunks})
    step(5, f"ask trip A about trip B's ticket -> file_ids={ids}")
    check(file_b.id not in ids, "trip B's ticket leaked into trip A's search results")

    # A colloquial paraphrase must outscore a question about something else entirely.
    paraphrase = vector_service.search_trip_documents(trip_a.id, "住哪儿 几号能 check in")
    unrelated = vector_service.search_trip_documents(trip_a.id, "请问海底捞几点关门")
    hot = paraphrase.chunks[0].similarity if paraphrase.chunks else None
    cold = unrelated.chunks[0].similarity if unrelated.chunks else None
    step(6, f"paraphrase similarity={hot} vs unrelated similarity={cold}")
    check(hot is not None and cold is not None and hot > cold, "a paraphrase did not outscore an unrelated question")

    removed = vector_service.delete_file_chunks(file_a.id)
    left_a = vector_service.knowledge_status(trip_a.id)["indexed_chunks"]
    step(7, f"delete file A -> removed {removed} chunk(s), {left_a} left in trip A")
    check(left_a == 0, "chunks survived the file delete")

    left_b = vector_service.knowledge_status(trip_b.id)["indexed_chunks"]
    step(8, f"trip B untouched -> {left_b} chunk(s)")
    check(left_b > 0, "deleting trip A's chunks also removed trip B's")

    # Re-indexing the same file must not leave two generations of its text behind.
    first = vector_service.index_trip_file(session, file_b).chunk_count
    second = vector_service.index_trip_file(session, file_b).chunk_count
    counted = vector_service.knowledge_status(trip_b.id)["indexed_chunks"]
    step(9, f"re-index twice -> {first}, {second}, stored={counted}")
    check(first == second == counted, "re-indexing left duplicate chunks behind")

    # Too little text must be refused with a recorded reason, not indexed as a promise.
    blank = _file(trip_a.id, "模糊截图.png", "AB", "check-c.png")
    session.add(blank)
    session.commit()
    session.refresh(blank)
    skipped = vector_service.index_trip_file(session, blank)
    step(10, f"index an unreadable upload -> {skipped.status}, reason={skipped.reason}")
    check(skipped.status == "skipped", "an unreadable upload was not refused")
    check(blank.index_error is not None, "the refusal reason was not recorded on the file row")


def main() -> int:
    Base.metadata.create_all(bind=engine)

    available, error = vector_service.warm_up()
    step(1, f"open the collection -> available={available} error={error}")
    if not available:
        check(False, f"the vector store could not be opened: {error}")
        return report()

    with SessionLocal() as session:
        run_checks(session)
    return report()


def report() -> int:
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    if FAILURES:
        print("\nFAILED:", flush=True)
        for item in FAILURES:
            print(f"  - {item}", flush=True)
        return 1
    print("\nAll document-index checks passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
