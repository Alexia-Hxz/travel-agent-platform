import sqlite3
import time

from app.agents.tools import recommendations


def test_category_material_is_persisted_by_destination_category_and_hint(monkeypatch, tmp_path):
    cache = tmp_path / "material.sqlite3"
    monkeypatch.setenv("CATEGORY_MATERIAL_CACHE_PATH", str(cache))
    monkeypatch.delenv("CATEGORY_MATERIAL_CACHE_FORCE_REFRESH", raising=False)
    calls = []

    def live(destination, category, hint=None):
        calls.append((destination, category, hint))
        return [{"title": "东京餐厅", "snippet": "寿司大是一家真实存在的餐厅，参考人均价格见店内。"}]

    monkeypatch.setattr(recommendations, "_live_category_material", live)

    first = recommendations._category_material("东京", "restaurant", "美食 推荐")
    second = recommendations._category_material("东京", "restaurant", "美食 推荐")

    assert second == first
    assert calls == [("东京", "restaurant", "美食 推荐")]
    assert cache.is_file()


def test_category_material_cache_ttl_and_force_refresh(monkeypatch, tmp_path):
    cache = tmp_path / "material.sqlite3"
    monkeypatch.setenv("CATEGORY_MATERIAL_CACHE_PATH", str(cache))
    monkeypatch.setenv("CATEGORY_MATERIAL_CACHE_TTL_HOURS", "1")
    answers = iter(
        [
            [{"title": "旧", "snippet": "这是第一份足够长的联网检索原始素材。"}],
            [{"title": "新", "snippet": "这是第二份足够长的联网检索原始素材。"}],
        ]
    )
    monkeypatch.setattr(recommendations, "_live_category_material", lambda *args: next(answers))

    recommendations._category_material("东京", "hotel", "酒店")
    with sqlite3.connect(cache) as connection:
        connection.execute("UPDATE category_material SET fetched_at = ?", (time.time() - 7200,))
    refreshed = recommendations._category_material("东京", "hotel", "酒店")
    assert refreshed[0]["title"] == "新"

    monkeypatch.setenv("CATEGORY_MATERIAL_CACHE_FORCE_REFRESH", "1")
    monkeypatch.setattr(
        recommendations,
        "_live_category_material",
        lambda *args: [{"title": "强制", "snippet": "强制刷新也会重新获得一份足够长的素材。"}],
    )
    forced = recommendations._category_material("东京", "hotel", "酒店")
    assert forced[0]["title"] == "强制"


def test_hints_that_ask_the_same_question_share_one_search(monkeypatch, tmp_path):
    """The price-tier backfill must not pay for the search the main group already ran.

    `_live_search_question` sends the plain category question for the tier wording, so
    "便宜 实惠 当地人常去 平价 人均低" and "米其林 名店…" ask, word for word, what the main
    restaurant group asked. Keying the cache on the hint made each of them buy that same
    answer again, and in a measured four-day plan those two searches sat on the critical
    path. The tier filter runs afterwards, on the extracted places, so nothing about the
    material changes.
    """
    cache = tmp_path / "material.sqlite3"
    monkeypatch.setenv("CATEGORY_MATERIAL_CACHE_PATH", str(cache))
    monkeypatch.delenv("CATEGORY_MATERIAL_CACHE_FORCE_REFRESH", raising=False)
    searched = []

    def live(destination, category, hint=None):
        searched.append(hint)
        return [{"title": "成都餐厅", "snippet": "马旺子川小馆是一家真实存在的餐厅，人均约 150 元。"}]

    monkeypatch.setattr(recommendations, "_live_category_material", live)

    main = recommendations._category_material("成都", "restaurant", "美食 餐厅 小吃 推荐 人均")
    cheap = recommendations._category_material("成都", "restaurant", "便宜 实惠 当地人常去 平价 人均低")
    splurge = recommendations._category_material("成都", "restaurant", "米其林 名店 高级 招牌 值得专程 好吃 排队")

    assert cheap == main
    assert splurge == main
    assert searched == ["美食 餐厅 小吃 推荐 人均"]

    # Breakfast is a genuinely different question, so it still searches for itself.
    breakfast = recommendations._category_material("成都", "restaurant", "早餐 早点 小吃 人均")
    assert len(searched) == 2
    assert breakfast is not None


def test_clear_category_material_cache_switch_is_scoped_to_this_cache(monkeypatch, tmp_path):
    cache = tmp_path / "material.sqlite3"
    monkeypatch.setenv("CATEGORY_MATERIAL_CACHE_PATH", str(cache))
    recommendations._write_category_material_cache(
        "东京", "attraction", "景点", [{"title": "景点", "snippet": "一段足够长的景点原始素材内容。"}]
    )
    recommendations.clear_category_material_cache()

    with sqlite3.connect(cache) as connection:
        count = connection.execute("SELECT COUNT(*) FROM category_material").fetchone()[0]
    assert count == 0
