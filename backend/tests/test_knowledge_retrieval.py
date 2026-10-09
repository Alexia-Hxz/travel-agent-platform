"""Semantic search over the traveller's uploaded documents.

The embedding provider is stubbed for the whole suite (see conftest), so these tests are
about chunking, the trip filter, the indexing lifecycle and the reported statuses. The
provider's own request shaping is covered in test_embedding_service.
"""

from datetime import date
from uuid import uuid4

from fastapi.testclient import TestClient

from app.db.session import SessionLocal
from app.main import app
from app.models.trip import Trip
from app.models.trip_file import TripFile
from app.services import knowledge_context, vector_service


HOTEL_HANGZHOU = """杭州西湖宾馆 预订确认单
订单号: AB123456
入住日期: 2026-10-02
离店日期: 2026-10-05
房型: 豪华大床房
地址: 杭州市西湖区湖滨路88号
电话: 0571-88888888
"""

FLIGHT_TOKYO = """东京成田机场 机票行程单
票号: JP998877
出发日期: 2026-11-11
航班: NH920
"""


def _create_trip(session) -> Trip:
    trip = Trip(
        title=f"索引测试 {uuid4().hex[:8]}",
        destination="杭州",
        start_date=date(2026, 10, 1),
        end_date=date(2026, 10, 5),
        budget=6000,
    )
    session.add(trip)
    session.commit()
    session.refresh(trip)
    return trip


def _create_file(session, trip_id: int, name: str, text: str) -> TripFile:
    """A file row as it looks once parsing has succeeded."""
    trip_file = TripFile(
        trip_id=trip_id,
        original_filename=name,
        stored_filename=f"{uuid4().hex}.pdf",
        content_type="application/pdf",
        file_size=len(text.encode("utf-8")),
        file_path=f"data/uploads/{trip_id}/{uuid4().hex}.pdf",
        parse_status="completed",
        parsed_text=text,
    )
    session.add(trip_file)
    session.commit()
    session.refresh(trip_file)
    return trip_file


def _api_trip(client: TestClient) -> int:
    response = client.post(
        "/api/v1/trips",
        json={
            "title": f"知识检索测试 {uuid4().hex[:8]}",
            "destination": "杭州",
            "start_date": "2026-10-01",
            "end_date": "2026-10-05",
            "budget": 6000,
        },
    )
    assert response.status_code == 201, response.text
    return response.json()["id"]


# --------------------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------------------


def test_split_document_keeps_every_line_whole() -> None:
    """A chunk must not begin or end mid-line.

    Orders are line-oriented, and cutting inside a line is what separates a label from
    its value ("入住日期" from "2026-10-02") so neither half can answer anything.
    """
    pieces = vector_service.split_document(HOTEL_HANGZHOU)
    assert pieces
    for line in HOTEL_HANGZHOU.strip().splitlines():
        assert any(line.strip() in piece.text.split("\n") for piece in pieces), line


def test_split_document_overlaps_across_a_split_line() -> None:
    pieces = vector_service.split_document("".join(str(index % 10) for index in range(1200)))
    assert len(pieces) > 1
    overlap = vector_service.CHUNK_OVERLAP_CHARS
    assert pieces[0].text[-overlap:] == pieces[1].text[:overlap]


def test_split_document_ignores_blank_text() -> None:
    assert vector_service.split_document("   \n\n  \n") == []


# --------------------------------------------------------------------------------------
# Indexing and retrieval
# --------------------------------------------------------------------------------------


def test_indexing_makes_a_document_searchable() -> None:
    with SessionLocal() as session:
        trip = _create_trip(session)
        trip_file = _create_file(session, trip.id, "杭州酒店订单.pdf", HOTEL_HANGZHOU)

        outcome = vector_service.index_trip_file(session, trip_file)

        assert outcome.status == "indexed"
        assert outcome.chunk_count >= 1
        assert trip_file.index_status == "indexed"
        assert trip_file.indexed_chunks == outcome.chunk_count
        assert trip_file.indexed_at is not None

        found = vector_service.search_trip_documents(trip.id, "杭州酒店入住日期是哪天")
        assert found.status == "ok"
        assert found.chunks[0].source_label == "杭州酒店订单.pdf"
        assert any("2026-10-02" in chunk.text for chunk in found.chunks)


def test_search_never_looks_outside_the_trip() -> None:
    """Byte-identical documents in two trips: only the filter separates them.

    Chroma has no partitions, so this is the property that has to be proven rather than
    assumed -- a missing filter would silently answer one traveller with another
    traveller's bookings.
    """
    with SessionLocal() as session:
        trip_a = _create_trip(session)
        trip_b = _create_trip(session)
        file_a = _create_file(session, trip_a.id, "订单.pdf", HOTEL_HANGZHOU)
        file_b = _create_file(session, trip_b.id, "订单.pdf", HOTEL_HANGZHOU)
        vector_service.index_trip_file(session, file_a)
        vector_service.index_trip_file(session, file_b)

        found = vector_service.search_trip_documents(trip_a.id, "杭州酒店入住日期")

        assert found.chunks
        assert {chunk.file_id for chunk in found.chunks} == {file_a.id}
        assert file_b.id not in {chunk.file_id for chunk in found.chunks}


def test_short_parsed_text_is_skipped_and_the_reason_is_recorded() -> None:
    """A scan that OCR could not read must not enter the index as an empty promise."""
    with SessionLocal() as session:
        trip = _create_trip(session)
        trip_file = _create_file(session, trip.id, "模糊截图.png", "AB")

        outcome = vector_service.index_trip_file(session, trip_file)

        assert outcome.status == "skipped"
        assert "2 字" in (outcome.reason or "")
        assert trip_file.index_status == "skipped"
        assert trip_file.index_error
        # And the trip is reported as having nothing to search, rather than as a failure.
        assert vector_service.search_trip_documents(trip.id, "任何问题").status == "empty"


def test_reindexing_replaces_the_previous_chunks() -> None:
    with SessionLocal() as session:
        trip = _create_trip(session)
        trip_file = _create_file(session, trip.id, "订单.pdf", HOTEL_HANGZHOU)
        vector_service.index_trip_file(session, trip_file)

        trip_file.parsed_text = f"{HOTEL_HANGZHOU}\n{FLIGHT_TOKYO}"
        session.commit()
        second = vector_service.index_trip_file(session, trip_file)

        status = vector_service.knowledge_status(trip.id)
        assert status["indexed_chunks"] == second.chunk_count
        assert status["indexed_files"] == 1
        assert vector_service.search_trip_documents(trip.id, "东京航班出发日期").chunks


def test_deleting_a_file_removes_its_chunks() -> None:
    with SessionLocal() as session:
        trip = _create_trip(session)
        trip_file = _create_file(session, trip.id, "订单.pdf", HOTEL_HANGZHOU)
        vector_service.index_trip_file(session, trip_file)
        assert vector_service.knowledge_status(trip.id)["indexed_chunks"] > 0

        removed = vector_service.delete_file_chunks(trip_file.id)

        assert removed > 0
        assert vector_service.knowledge_status(trip.id)["indexed_chunks"] == 0


def test_deleting_a_trip_removes_all_of_its_chunks() -> None:
    with SessionLocal() as session:
        trip = _create_trip(session)
        first = _create_file(session, trip.id, "订单.pdf", HOTEL_HANGZHOU)
        second = _create_file(session, trip.id, "机票.pdf", FLIGHT_TOKYO)
        vector_service.index_trip_file(session, first)
        vector_service.index_trip_file(session, second)

        vector_service.delete_trip_chunks(trip.id)

        assert vector_service.knowledge_status(trip.id)["indexed_chunks"] == 0


# --------------------------------------------------------------------------------------
# What the agent graphs are handed
# --------------------------------------------------------------------------------------


def test_document_search_event_explains_an_empty_index() -> None:
    """An empty list must come with a reason, or it reads as "you have no such document"."""
    with SessionLocal() as session:
        trip = _create_trip(session)
        documents, event = knowledge_context.retrieve_traveler_documents(trip.id, "有资料吗")

        assert documents == []
        assert event["tool"] == "document_search"
        assert event["status"] == "skipped"
        assert "还没有已建立索引的资料" in event["message"]


def test_document_search_event_names_the_documents_it_used() -> None:
    with SessionLocal() as session:
        trip = _create_trip(session)
        trip_file = _create_file(session, trip.id, "杭州酒店订单.pdf", HOTEL_HANGZHOU)
        vector_service.index_trip_file(session, trip_file)

        documents, event = knowledge_context.retrieve_traveler_documents(trip.id, "杭州酒店入住日期")

        assert event["status"] == "completed"
        assert "杭州酒店订单.pdf" in event["message"]
        assert documents[0]["source"] == "杭州酒店订单.pdf"
        assert "2026-10-02" in documents[0]["excerpt"]


# --------------------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------------------


def test_knowledge_endpoints_rebuild_search_and_report() -> None:
    with TestClient(app) as client:
        trip_id = _api_trip(client)
        with SessionLocal() as session:
            _create_file(session, trip_id, "杭州酒店订单.pdf", HOTEL_HANGZHOU)

        # A file parsed before this feature existed has no chunks; rebuild is the repair.
        rebuilt = client.post(f"/api/v1/trips/{trip_id}/knowledge/rebuild")
        assert rebuilt.status_code == 200, rebuilt.text
        assert rebuilt.json()["indexed_files"] == 1
        assert rebuilt.json()["indexed_chunks"] >= 1

        status = client.get(f"/api/v1/trips/{trip_id}/knowledge/status")
        assert status.status_code == 200
        assert status.json()["available"] is True
        assert status.json()["indexed_chunks"] == rebuilt.json()["indexed_chunks"]
        assert status.json()["backend"] == "chromadb"

        found = client.get(
            f"/api/v1/trips/{trip_id}/knowledge/search", params={"q": "杭州酒店入住日期"}
        )
        assert found.status_code == 200, found.text
        payload = found.json()
        assert payload["status"] == "ok"
        assert payload["chunks"][0]["source_label"] == "杭州酒店订单.pdf"
        assert any("2026-10-02" in chunk["text"] for chunk in payload["chunks"])


def test_knowledge_search_distinguishes_an_empty_index_from_a_failure() -> None:
    with TestClient(app) as client:
        trip_id = _api_trip(client)
        response = client.get(
            f"/api/v1/trips/{trip_id}/knowledge/search", params={"q": "有资料吗"}
        )

        assert response.status_code == 200
        assert response.json()["status"] == "empty"
        assert response.json()["chunks"] == []
        assert response.json()["reason"]


def test_knowledge_endpoints_are_scoped_to_the_owned_trip() -> None:
    """The router sits behind the same ownership guard as the other trip routers."""
    with TestClient(app) as client:
        response = client.get("/api/v1/trips/99999999/knowledge/status")
        assert response.status_code == 404


# --------------------------------------------------------------------------------------
# Retrieval quality
# --------------------------------------------------------------------------------------

LABELLED_DOCUMENTS = {
    "杭州酒店订单.pdf": "杭州西湖宾馆 预订确认单 入住日期 2026-10-02 离店日期 2026-10-05 地址 杭州市西湖区湖滨路88号 订单号 AB123456",
    "东京机票行程单.pdf": "东京成田机场 机票行程单 出发日期 2026-11-11 航班 NH920 票号 JP998877 座位 32A",
    "大阪餐厅预约.pdf": "大阪道顿堀 蟹道乐 餐厅预约 就餐时间 2026-11-12 18:30 人数 4 预约号 OS7788",
    "京都寺庙门票.pdf": "京都清水寺 门票 参观日期 2026-11-13 两张 票号 KY3344 单价 400 日元",
    "北海道租车单.pdf": "北海道札幌 租车 取车日期 2026-11-14 车型 普锐斯 订单号 HK5566 门店 薄野营业所",
    "冲绳潜水预约.pdf": "冲绳 青之洞窟 潜水体验 预约日期 2026-11-15 教练 小林 预约号 OK1122",
    "首尔民宿确认.pdf": "首尔弘大 民宿 入住 2026-12-01 退房 2026-12-04 房东 金先生 确认号 SE2211",
    "曼谷包车单.pdf": "曼谷 素万那普机场 包车 接机时间 2026-12-05 车型 商务车 司机 阿南 订单号 BK8833",
    "新加坡环球影城门票.pdf": "新加坡 环球影城 门票 使用日期 2026-12-06 成人 2 张 订单号 SG4455",
    "吉隆坡双子塔观景台.pdf": "吉隆坡 双子塔 观景台 参观时段 2026-12-07 19:00 人数 3 预订号 KL6677",
    "河内下龙湾邮轮.pdf": "河内 下龙湾 两日游邮轮 出发日期 2026-12-08 舱位 豪华 订单号 HN8899",
    "上海迪士尼年卡.pdf": "上海 迪士尼 年卡 激活日期 2026-12-09 持卡人 张伟 卡号 SH1010",
}

LABELLED_QUERIES = {
    "杭州的酒店是哪天入住": "杭州酒店订单.pdf",
    "去东京的航班号和出发日期": "东京机票行程单.pdf",
    "大阪那家蟹道乐订了几个人": "大阪餐厅预约.pdf",
    "京都清水寺的门票买了几张": "京都寺庙门票.pdf",
    "札幌租车的取车日期和车型": "北海道租车单.pdf",
    "冲绳潜水体验的教练是谁": "冲绳潜水预约.pdf",
    "首尔的民宿住到几号，房东是谁": "首尔民宿确认.pdf",
    "曼谷接机的司机和车型": "曼谷包车单.pdf",
    "新加坡环球影城门票买了几张": "新加坡环球影城门票.pdf",
    "吉隆坡双子塔观景台几点参观": "吉隆坡双子塔观景台.pdf",
    "下龙湾邮轮哪天出发，什么舱位": "河内下龙湾邮轮.pdf",
    "上海迪士尼年卡是谁的": "上海迪士尼年卡.pdf",
}


def test_recall_at_five_on_a_labelled_question_set() -> None:
    """The measurement that stops this becoming an index nobody can judge.

    Twelve documents and twelve questions, each answerable only by one of them. A
    retrieval layer that is wired up but returns nothing, or returns the wrong document,
    scores below one here -- which is exactly what the earlier vector attempt shipped
    without anyone being able to tell.
    """
    with SessionLocal() as session:
        trip_id = _create_trip(session).id
        for name, text in LABELLED_DOCUMENTS.items():
            vector_service.index_trip_file(session, _create_file(session, trip_id, name, text))

        hits = 0
        top_one = 0
        misses: list[str] = []
        for question, expected in LABELLED_QUERIES.items():
            outcome = vector_service.search_trip_documents(trip_id, question, limit=5)
            sources = [chunk.source_label for chunk in outcome.chunks]
            if expected in sources:
                hits += 1
            else:
                misses.append(f"{question} -> {sources}")
            if sources and sources[0] == expected:
                top_one += 1

        assert hits == len(LABELLED_QUERIES), f"Recall@5 missed: {misses}"
        assert top_one == len(LABELLED_QUERIES), f"top-1 accuracy was {top_one}/{len(LABELLED_QUERIES)}"
