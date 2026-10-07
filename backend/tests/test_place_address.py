"""Addresses: the one thing that makes an itinerary usable, and the reason it was not.

The extraction now asks for an address; this covers what it still misses, with the three
promises the module makes -- it is paced and budgeted, it never invents an answer, and it writes
the address in a form a Chinese reader can follow.
"""
import pytest

from app.agents.tools import place_address


@pytest.fixture(autouse=True)
def _lift_the_offline_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests are about the lookup itself, so the suite's no-network switch is lifted.

    The module refuses to call out whenever the suite's throwaway-database variable is set --
    that is what keeps every other test offline. The one test that checks that rule sets the
    variable itself.
    """
    monkeypatch.delenv("TRAVEL_AGENT_DB_PATH", raising=False)
    # The lookup is off by default because this host cannot reach the geocoder; these tests are
    # about what happens when it can.
    monkeypatch.setattr(place_address, "LOOKUPS_ENABLED", True)


def test_japanese_administrative_names_are_written_the_way_a_chinese_reader_reads_them() -> None:
    assert place_address.to_chinese("東京都台東区上野1-1-1") == "东京都台东区上野1-1-1"
    assert place_address.to_chinese("大阪府大阪市中央区") == "大阪府大阪市中央区"
    # A street name is left alone: it has no Chinese form, and a guess would mislead.
    assert "上野" in place_address.to_chinese("東京都台東区上野")


def test_only_a_missing_address_is_looked_up() -> None:
    assert place_address.needs_address("") is True
    assert place_address.needs_address("日本东京 · 地址待确认") is True
    assert place_address.needs_address("日本东京", "日本东京") is True
    assert place_address.needs_address("东京都台东区上野1-1-1", "日本东京") is False


def test_a_looked_up_address_replaces_the_placeholder_and_brings_coordinates() -> None:
    calls: list[tuple[str, str]] = []

    def fake_lookup(name: str, destination: str):
        calls.append((name, destination))
        return {
            "address": "东京都台东区上野1-1-1",
            "latitude": "35.7126",
            "longitude": "139.7770",
            "source": "openstreetmap",
        }

    places = [
        {"name": "万珍楼", "type": "restaurant", "address": "日本东京 · 地址待确认"},
        {"name": "已定地点", "type": "attraction", "address": "东京都新宿区西新宿2-8-1"},
    ]

    filled = place_address.fill_place_addresses(places, "日本东京", lookup=fake_lookup)

    assert filled[0]["address"] == "东京都台东区上野1-1-1"
    assert filled[0]["address_source"] == "openstreetmap"
    assert filled[0]["latitude"] == 35.7126
    # A place that already had a real address is not searched for or overwritten.
    assert calls == [("万珍楼", "日本东京")]
    assert filled[1]["address"] == "东京都新宿区西新宿2-8-1"


def test_a_lookup_that_finds_nothing_leaves_the_text_alone() -> None:
    places = [{"name": "无名小馆", "type": "restaurant", "address": "日本东京 · 地址待确认"}]

    filled = place_address.fill_place_addresses(places, "日本东京", lookup=lambda name, destination: None)

    assert filled[0]["address"] == "日本东京 · 地址待确认"
    assert filled[0].get("address_source") is None


def test_the_budget_stops_the_pass_instead_of_holding_up_the_plan() -> None:
    looked: list[str] = []

    def fake_lookup(name: str, destination: str):
        looked.append(name)
        return {"address": "东京都台东区上野1-1-1", "latitude": None, "longitude": None}

    places = [
        {"name": f"地点{index}", "type": "attraction", "address": "日本东京 · 地址待确认"}
        for index in range(6)
    ]

    # A budget of zero means no lookups at all: the plan keeps whatever text it had.
    filled = place_address.fill_place_addresses(places, "日本东京", budget=0.0, lookup=fake_lookup)

    assert looked == []
    assert all(place["address"] == "日本东京 · 地址待确认" for place in filled)


def test_the_test_suite_never_calls_out() -> None:
    """The suite points the database at a throwaway file; that switch also keeps this offline."""
    import os

    os.environ["TRAVEL_AGENT_DB_PATH"] = "throwaway.db"
    try:
        called: list[str] = []
        place_address.fill_place_addresses(
            [{"name": "万珍楼", "type": "restaurant", "address": ""}],
            "日本东京",
            lookup=lambda name, destination: called.append(name),
        )
        assert called == []
    finally:
        os.environ.pop("TRAVEL_AGENT_DB_PATH", None)
