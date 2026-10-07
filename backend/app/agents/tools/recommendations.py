"""Turn grounded search material into the places a plan actually recommends.

Search results are whole pages, so their titles are article headlines rather than
place names. This module extracts the concrete hotels, restaurants and attractions
mentioned in that material, so every recommendation can carry its own picture and
reason, and so the budget never charges a headline as if it were a booking.

Extraction prefers the chat model and falls back to keyword rules when no model key
is configured, so the plan still produces recommendations offline.
"""

import json
import logging
import os
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait
from contextlib import closing
from pathlib import Path
from typing import Any, Callable, Iterable

from app.agents.tools import currency, dashscope_search, meal_prices
from app.agents.tools.llm import MODEL_NAME, model_client
from app.agents.tools.model_json import parse_json_object
from app.agents.tools.place_images import LOOKUP_WORKERS, MAX_IMAGE_LOOKUPS, normalize_category
from app.agents.tools.web_search import search_snippets


logger = logging.getLogger(__name__)


CATEGORY_ORDER = ("hotel", "restaurant", "attraction")
DEFAULT_PRICE_LEVELS = {"hotel": 3, "restaurant": 2, "attraction": 1}
WEB_MATERIAL_LIMIT = 5
SNIPPET_CHARS = 700
# A live search answer is the whole material for one category, not one snippet among
# several, and it lists a dozen places. Cutting it at SNIPPET_CHARS dropped half of them.
LIVE_MATERIAL_CHARS = 4000
# How many places the live search is asked for. It answered twelve reliably in about
# twenty seconds; the per-request ceiling that applies to extraction does not apply here,
# because this text is not one structured answer but material for the next step.
LIVE_MATERIAL_ITEMS = 12
NAME_CHARS = 80
MIN_MATERIAL_CHARS = 20

_CATEGORY_KEYWORDS = (
    ("hotel", ("酒店", "宾馆", "民宿", "客栈", "住宿", "hotel", "hostel")),
    ("restaurant", ("餐厅", "美食", "小吃", "餐馆", "饭店", "咖啡", "茶馆", "面馆", "生煎", "菜馆", "restaurant", "food", "cafe")),
    ("attraction", ("景点", "景区", "公园", "博物馆", "古镇", "乐园", "美术馆", "步行街", "外滩", "塔", "寺", "attraction", "museum", "park", "garden")),
)

_SYSTEM_PROMPT = (
    "你是一名旅行规划助理，负责从给定的检索资料中抽取真实存在的具体地点。"
    "只输出 JSON 数组，不要输出任何解释或代码块标记。"
    '每个元素形如：{"name": "地点名称", "category": "hotel|restaurant|attraction", '
    '"reason": "一句中文推荐理由", "price_hint": "资料中的价格原文，未知则填 null", '
    '"unit_price": 单个计价单位的参考价（整数元，未知则填 null）, '
    '"price_unit": "night|person|person_meal|ticket", '
    '"image_query": "用于在图库中检索这张卡片配图的英文关键词（2-4 个英文单词）", '
    '"meal_slot": "breakfast|lunch|dinner|any（仅 restaurant 需要，其他填 null）", '
    '"latitude": 这个地点的**大致**纬度（数字，保留 3~4 位小数；不确定就填 null）, '
    '"longitude": 这个地点的**大致**经度（数字；不确定就填 null）}。'
    "name 必须是资料正文中出现的具体场所名称（酒店名、餐厅名、景点名）。"
    # Without this the same venue came back in Chinese and in English on different days,
    # and the name comparison could not tell they were one place.
    "名称一律使用资料中的中文原名，不要翻译成英文或其他语言。"
    "严禁把网页标题、攻略文章标题、栏目名或平台名当作 name。"
    "**严禁输出描述性名称或类别名，例如「夫子庙地铁站附近酒店」「南京住宿推荐」"
    "「老门东周边餐饮」——这些都不是可以预订或到访的具体场所。"
    "如果资料里没有符合预算的真实店名，就照实输出资料中真实存在的店名并如实标价，"
    "不要用描述性名称凑数。**"
    "category 只能是 hotel、restaurant、attraction 之一。"
    "unit_price 的计价单位按分类固定：hotel 填每晚房价（price_unit=night）、"
    "restaurant 填每人每餐人均（price_unit=person_meal）、attraction 填每人门票（price_unit=ticket）。"
    "unit_price 只能取自资料中出现的价格，资料没写就填 null，禁止凭常识编造价格。"
    "unit_price 必须是**人民币整数**：资料写的是外币时，按上面给出的汇率换算成人民币再填；"
    # The English keyword exists because the image library's index is English: a Chinese
    # name searched as-is returned a photo of Osaka Castle for three different Tokyo
    # museums. See `place_images`.
    "image_query 用英文写：景点写这个景点的英文名（如 Sensoji Temple、teamLab Planets），"
    "餐厅写这道菜的英文名或英文类别（如 tonkotsu ramen、sushi），不要写店名，"
    "酒店写 \"城市名 hotel\"。资料里没有英文名时，用最接近的英文类别词。"
    "不得编造资料中不存在的地点或价格；资料不足时宁可少输出。不要输出重复地点。"
    "**餐厅要覆盖一整天：至少给出 3 家，其中至少 1 家是早餐（meal_slot=breakfast，"
    "如包子、汤包、豆浆、早茶、面馆等），其余标注 lunch 或 dinner。"
    "同一家店不要重复输出。**"
)

PRICE_UNITS = {"night", "person", "person_meal", "ticket"}
DEFAULT_PRICE_UNIT = {"hotel": "night", "restaurant": "person_meal", "attraction": "ticket"}
# Used only when the material carries no price at all, so a card can still show a
# rough figure. The explanation always says the number is a mid-range default.
DEFAULT_UNIT_PRICES = {"hotel": 400, "restaurant": 80, "attraction": 100}
# Sanity bounds per unit, in CNY. Search material often contains phone numbers,
# star ratings or review counts; without a bound a number like a hotel's phone
# number becomes a "price" and the trip total explodes into the millions.
PRICE_BOUNDS = {
    "hotel": (50, 20000),
    "restaurant": (10, 2000),
    "attraction": (1, 3000),
}
# Above these CNY figures a foreign price is far more likely to be the local currency than
# a real CNY amount: nobody recommends a 3,000 元 museum ticket or a 1,480 元 bowl of ramen,
# while 3,000 日元 and 1,480 日元 are ordinary. Applied only to non-CNY destinations, so a
# 1,500 元 banquet in Shanghai is left alone.
FOREIGN_LOCAL_THRESHOLDS = {
    "restaurant": 600,
    "attraction": 400,
}

# The single mixed extraction sometimes returns only one category (a user asking
# for hotels, food and sights got just one sight). These focused prompts backfill
# whichever category came back empty, so a plan always covers all three.
_CATEGORY_PROMPTS = {
    "hotel": (
        "你是一名旅行规划助理，负责从给定的检索资料中抽取**真实存在的酒店或民宿**名称。"
        "只输出 JSON 数组，不要输出任何解释或代码块标记。"
        '每个元素形如：{"name": "酒店名称", "category": "hotel", "reason": "一句中文推荐理由", '
        '"price_hint": "资料中的价格原文，未知则填 null", "unit_price": 每晚房价整数元或 null, "price_unit": "night"}。'
        "name 必须是资料正文中出现的具体酒店名，严禁把网页标题、榜单名或平台名当作 name。"
        "**严禁输出「夫子庙地铁站附近酒店」这类描述性名称；资料里没有真实酒店名时宁可少输出。**"
    ),
    "restaurant": (
        "你是一名旅行规划助理，负责从给定的检索资料中抽取**真实存在的餐厅、小吃或咖啡馆**名称。"
        "只输出 JSON 数组，不要输出任何解释或代码块标记。"
        '每个元素形如：{"name": "餐厅名称", "category": "restaurant", "reason": "一句中文推荐理由", '
        '"price_hint": "资料中的价格原文，未知则填 null", "unit_price": 每人每餐人均整数元或 null, '
        '"price_unit": "person_meal", "meal_slot": "breakfast|lunch|dinner|any", '
        '"image_query": "这道菜的英文关键词（2-4 个英文单词，如 tonkotsu ramen）"}。'
        "name 必须是资料正文中出现的具体店名，严禁把网页标题、榜单名或平台名当作 name。"
        "image_query 写**这道招牌菜**的英文名或英文类别，**不要写店名**（图库没有这家店的实拍，"
        "但有这道菜的照片）。"
        "**至少给出 3 家，必须包含至少 1 家早餐店（meal_slot=breakfast，例如包子、汤包、"
        "豆浆、油条、早茶、粥、面馆、锅贴等），其余标注 lunch 或 dinner。不要重复同一家店。**"
    ),
    "attraction": (
        "你是一名旅行规划助理，负责从给定的检索资料中抽取**真实存在的景点、博物馆或古镇**名称。"
        "只输出 JSON 数组，不要输出任何解释或代码块标记。"
        '每个元素形如：{"name": "景点名称", "category": "attraction", "reason": "一句中文推荐理由", '
        '"price_hint": "资料中的价格原文，未知则填 null", "unit_price": 每人门票整数元或 null, "price_unit": "ticket", '
        '"image_query": "这个景点的英文名（如 Sensoji Temple）"}。'
        "name 必须是资料正文中出现的具体景点名，严禁把网页标题、榜单名或平台名当作 name。"
        "image_query 必须是**英文**，写这个景点自己的英文名；查不到英文名就写最接近的英文类别"
        "（如 japanese temple、art museum），不要写中文。"
    ),
}
_CATEGORY_TAIL = (
    "价格只能取自资料中出现的数字，资料没写就填 null，禁止凭常识编造价格。"
    "unit_price 必须是**人民币整数**：资料里写的是外币（日元、美元、欧元等）时，"
    "按上面给出的汇率换算成人民币再填，price_hint 里保留资料原文和原币种。"
    "不得编造资料中不存在的地点或价格；资料不足时宁可少输出。不要输出重复地点。"
    # The workspace map only sketches the day's route, so the position has to be roughly
    # right rather than exact -- but it must be a real position, and null beats a guess:
    # a confident wrong pin sends the traveller to the wrong side of the city.
    "每个元素再带上 \"latitude\" 与 \"longitude\" 两个字段，填这个地点的**大致**经纬度"
    "（数字，保留 3~4 位小数，例如 31.2397 / 121.4900）。只要求大致位置，几公里的误差没关系；"
    "确实不知道就两个都填 null，**不要**用城市中心或 0 占位，也不要只填其中一个。"
    # The model named one venue in Chinese on one day and in English on another ("咸亨酒店"
    # and "Xian Heng Restaurant"), and the string comparison could not see they were the
    # same restaurant, so the trip was sent to it twice.
    "地点名称一律使用资料中的中文原名，不要翻译成英文或其他语言。"
)
CATEGORY_SEARCH_HINTS = {
    "hotel": "酒店 住宿 推荐 房价",
    "restaurant": "美食 餐厅 小吃 推荐 人均",
    "attraction": "景点 必去 门票",
}
# Live search is asked a question rather than given keywords, because a searching model
# retrieves better for a question. These are the phrases that go into that question.
CATEGORY_SEARCH_QUESTIONS = {
    "hotel": "口碑好、交通方便的酒店和民宿",
    "restaurant": "本地人常去的餐厅、小吃店和老字号",
    "attraction": "值得一去的景点和博物馆",
}
BREAKFAST_SEARCH_QUESTION = "本地人吃早餐的店，比如包子、汤包、粥、糕团、鸭血粉丝、早茶"
# The live search is told which meal a batch is for; the hint carries that, not the category.
_BREAKFAST_HINT_MARKER = "早餐"


def _live_search_question(category: str, hint: str | None) -> str:
    """Turn a search hint into the question a live search should answer."""
    if hint and _BREAKFAST_HINT_MARKER in hint:
        return BREAKFAST_SEARCH_QUESTION
    return CATEGORY_SEARCH_QUESTIONS.get(category, hint or category)
# A whole trip's worth of candidates cannot come from one model call on this provider.
# Asking for fifteen restaurants at once was answered either with a 30-second timeout or
# with a flat "data_inspection_failed" rejection, so each category is instead gathered by
# several smaller lookups that use different search phrasings and run concurrently.
# Each batch also states which meal it is for, because the daily plan needs one breakfast
# venue and two main-meal venues per day, and an unguided search returns mostly the latter.
# A line repeated with the same hint means "another batch from the same search": searches
# are charged per request, so one search is reused by the batches that follow it.
CATEGORY_SEARCH_VARIANTS = {
    "hotel": (("酒店 住宿 推荐 房价", None),),
    "restaurant": (
        ("美食 餐厅 小吃 推荐 人均", "main"),
        ("美食 餐厅 小吃 推荐 人均", "main"),
        ("早餐 早点 小吃 人均", "breakfast"),
        ("早餐 早点 小吃 人均", "breakfast"),
    ),
    "attraction": (
        ("景点 必去 门票", None),
        ("景点 必去 门票", None),
    ),
}
SLOT_INSTRUCTIONS = {
    "breakfast": "本批只要早餐店（包子、汤包、粥、糕团、早茶等），meal_slot 一律填 breakfast。",
    "main": "本批只要正餐餐厅（午餐或晚餐），不要早餐店；meal_slot 填 lunch 或 dinner。",
}
# The most one call is asked for. Measured against this provider: eight, ten and twelve
# places all came back complete and in under 25 seconds, twice in a row, while a fifteen
# place answer was rejected outright ("data_inspection_failed").
MAX_ITEMS_PER_CALL = 12
# Output ceilings. The mixed call answers for every category at once, so it needs the most
# room; a batch answers for six places. Both were previously too small, and a truncated
# answer is indistinguishable from an empty search — the plan just fell back to keyword
# rules and showed article titles as place names.
#
# Measured in a full four-day plan: 3000 was still too small for this prompt's own request.
# `_count_hint` asks the mixed call for 27 places (12 sights, 12 restaurants, 3 hotels), and
# the answer was cut off after 7229 characters (finish_reason=length) — roughly fourteen of
# them. The ceiling now covers the request with headroom, and a truncated answer no longer
# costs the whole list: `_parse_json_array` keeps every place that did arrive.
MIXED_MAX_TOKENS = 6000
BATCH_MAX_TOKENS = 2000
# Per-call allowance for the extraction calls, which produce a list rather than a
# sentence. Bounded so a stalled provider still cannot hold the request open for minutes.
EXTRACTION_TIMEOUT = 90.0
MAX_CATEGORY_FILLS = 3
# Hard wall-clock budget for the per-place price lookups, for the same reason as the
# picture phase: an httpx timeout does not bound DNS resolution.
PRICE_PHASE_BUDGET = 30.0
# Per-place price lookups are independent, short and bounded; the category searches they run
# beside are long. Sharing one worker count between them meant the price phase, which has its
# own 30s budget, ran out of clock with places still unpriced -- and an unpriced place shows a
# labelled default estimate, so the cost of that showed up as wrong prices rather than as a
# slow plan. Measured on this provider: six real lookups took 2.8s to 9.6s, median 4.2s.
PRICE_LOOKUP_WORKERS = 8
# What one price lookup is allowed to take. The old 8.0 was written for the metered search
# provider that was since removed and was never in force afterwards; measured, 8s still cut
# off one lookup in six, and the answer it would have returned is a real price. 12s clears
# the slowest of six measurements by a quarter while cutting the inherited 120s bound by 90%.
PRICE_LOOKUP_TIMEOUT = 12.0
# The fallback material search is a paragraph, not a price line, so it gets a bound sized for
# material -- but still a bound: unset, it inherited the 120s planning timeout.
MATERIAL_FALLBACK_TIMEOUT = 30.0

# Category material is by far the slowest part of a plan and, unlike per-place prices,
# remains useful across runs. Keep only this raw material on disk. In particular, do not
# route ``search_snippets`` price lookups or ``place_facts`` through this cache: both are
# intentionally fresh on every plan.
CATEGORY_MATERIAL_CACHE_TTL_HOURS = 24.0
CATEGORY_MATERIAL_CACHE_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "category_material_cache.sqlite3"
)
_category_cache_lock = threading.RLock()
_category_cache_clear_applied = False


def _env_enabled(name: str) -> bool:
    return os.getenv(name, "").strip().casefold() in {"1", "true", "yes", "on"}


def _category_cache_path() -> Path:
    configured = os.getenv("CATEGORY_MATERIAL_CACHE_PATH", "").strip()
    return Path(configured).expanduser() if configured else CATEGORY_MATERIAL_CACHE_PATH


def _category_cache_ttl_seconds() -> float:
    try:
        hours = float(os.getenv("CATEGORY_MATERIAL_CACHE_TTL_HOURS", str(CATEGORY_MATERIAL_CACHE_TTL_HOURS)))
    except ValueError:
        hours = CATEGORY_MATERIAL_CACHE_TTL_HOURS
    return max(0.0, hours * 3600.0)


def _open_category_cache() -> sqlite3.Connection:
    path = _category_cache_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=10.0)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS category_material (
            destination TEXT NOT NULL,
            category TEXT NOT NULL,
            hint TEXT NOT NULL,
            fetched_at REAL NOT NULL,
            material_json TEXT NOT NULL,
            PRIMARY KEY (destination, category, hint)
        )
        """
    )
    return connection


def clear_category_material_cache() -> None:
    """Clear the persistent category-material cache without touching any other cache."""
    with _category_cache_lock:
        try:
            with closing(_open_category_cache()) as connection, connection:
                connection.execute("DELETE FROM category_material")
        except (OSError, sqlite3.Error):
            logger.warning("could not clear category material cache", exc_info=True)


def _apply_category_cache_clear_switch() -> None:
    """Apply the startup-style clear switch once, not before every category lookup."""
    global _category_cache_clear_applied
    if _category_cache_clear_applied or not _env_enabled("CATEGORY_MATERIAL_CACHE_CLEAR"):
        return
    with _category_cache_lock:
        if not _category_cache_clear_applied:
            clear_category_material_cache()
            _category_cache_clear_applied = True


def _read_category_material_cache(
    destination: str, category: str, hint: str
) -> list[dict[str, Any]] | None:
    _apply_category_cache_clear_switch()
    if _env_enabled("CATEGORY_MATERIAL_CACHE_FORCE_REFRESH"):
        return None
    ttl = _category_cache_ttl_seconds()
    if ttl <= 0:
        return None
    with _category_cache_lock:
        try:
            with closing(_open_category_cache()) as connection, connection:
                row = connection.execute(
                    "SELECT fetched_at, material_json FROM category_material "
                    "WHERE destination = ? AND category = ? AND hint = ?",
                    (destination, category, hint),
                ).fetchone()
                if row is None:
                    return None
                if time.time() - float(row[0]) >= ttl:
                    connection.execute(
                        "DELETE FROM category_material WHERE destination = ? AND category = ? AND hint = ?",
                        (destination, category, hint),
                    )
                    return None
                parsed = json.loads(str(row[1]))
        except (OSError, sqlite3.Error, ValueError, TypeError):
            logger.warning("could not read category material cache", exc_info=True)
            return None
    return parsed if isinstance(parsed, list) else None


def _write_category_material_cache(
    destination: str, category: str, hint: str, material: list[dict[str, Any]]
) -> None:
    if not material or _category_cache_ttl_seconds() <= 0:
        return
    payload = json.dumps(material, ensure_ascii=False, separators=(",", ":"))
    with _category_cache_lock:
        try:
            with closing(_open_category_cache()) as connection, connection:
                connection.execute(
                    """
                    INSERT INTO category_material(destination, category, hint, fetched_at, material_json)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(destination, category, hint) DO UPDATE SET
                        fetched_at = excluded.fetched_at,
                        material_json = excluded.material_json
                    """,
                    (destination, category, hint, time.time(), payload),
                )
        except (OSError, sqlite3.Error):
            logger.warning("could not write category material cache", exc_info=True)



def _has_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in text)


def _prefer_chinese(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the Chinese-language blocks, and fall back to all of them if that empties it.

    The answers are written in Chinese, and a place named from an English page comes back
    in English ("Confucius Temple"), so the same venue could appear once per language and
    the plan then served it twice — "咸亨酒店" for lunch and "Xian Heng Restaurant" for
    dinner on one day. The blocks are not dropped outright: a thin search that found only
    English pages is still better than no material at all.
    """
    chinese = [block for block in blocks if _has_cjk(f"{block.get('title', '')}{block.get('snippet', '')}")]
    return chinese or blocks


def _material(
    web_results: list[dict[str, Any]] | None,
    web_limit: int = WEB_MATERIAL_LIMIT,
) -> list[dict[str, Any]]:
    """Collect trimmed, grounded text blocks for extraction."""
    material: list[dict[str, Any]] = []
    used = 0
    for result in web_results or []:
        if used >= web_limit:
            break
        title = str(result.get("title") or "").strip()
        snippet = str(result.get("snippet") or "").strip()
        if not title and not snippet:
            continue
        material.append({"title": title, "snippet": snippet[:SNIPPET_CHARS]})
        used += 1
    return _prefer_chinese(material)


def _material_text(material: list[dict[str, Any]]) -> str:
    return "\n".join(f"{block['title']}\n{block['snippet']}" for block in material)


def _classify_keywords(text: str) -> str:
    lowered = str(text or "").lower()
    for category, keywords in _CATEGORY_KEYWORDS:
        if any(keyword in lowered for keyword in keywords):
            return category
    return ""


# A name carrying one of these is a *description of a category*, not a place: search
# pages are full of headings like "夫子庙地铁站附近酒店" and "南京住宿推荐", and the
# prompt already forbids them. The 3-character grounding test below cannot catch them,
# because "夫子庙" really is in the material — so they are rejected explicitly.
_DESCRIPTIVE_NAME_MARKERS = (
    "附近", "周边", "一带", "沿线", "推荐", "排行", "榜单", "攻略",
    "大全", "盘点", "汇总", "指南", "必去", "值得去", "怎么去", "哪里",
    # Traditional-Chinese forms of the same words: a page titled
    # "南京住宿推薦｜新街口…步行可達10間精選酒店" got through on 推薦.
    "推薦", "週邊", "精選", "懶人包", "整理", "熱門", "景點攻略", "住宿推薦",
)
# Characters that only appear in article titles, never in a business name.
_TITLE_SEPARATORS = ("|", "｜", "・", "·", "【", "】", "《", "》")
# Search-provider and platform artifacts. "Tavily 联网检索摘要" was actually rendered
# as a recommendation card name when the model was unavailable, which is both a
# meaningless card and a leak of the search provider into the user interface.
_PROVIDER_NAME_MARKERS = (
    "tavily", "联网检索", "检索摘要", "搜索结果", "网页摘要",
    "携程", "知乎", "马蜂窝", "大众点评", "小红书", "本地宝", "去哪儿", "飞猪", "穷游",
)

def _is_descriptive_name(name: str) -> bool:
    """True when a name looks like a search heading rather than a real place."""
    folded = name.casefold()
    if any(marker in folded for marker in _PROVIDER_NAME_MARKERS):
        return True
    if any(separator in name for separator in _TITLE_SEPARATORS):
        return True
    return any(marker in name for marker in _DESCRIPTIVE_NAME_MARKERS)


def _is_grounded(name: str, text: str) -> bool:
    """Reject names the material never mentioned, tolerating light normalization.

    The fallback accepts a name sharing any three consecutive characters with the
    material, which is deliberately loose so that minor punctuation differences do
    not discard a real place. It is also why `_is_descriptive_name` runs first: a
    search heading like "夫子庙地铁站附近酒店" shares "夫子庙" with the material and
    would otherwise pass as a place.

    The descriptive rejection applies to the keyword fallback too. That path uses a
    page's own title as the place name, and with the model unavailable the titles are
    what the traveller actually sees — one of them was literally "Tavily 联网检索摘要",
    which is both useless and a leak of the search provider into the UI.
    """
    if _is_descriptive_name(name):
        return False
    if name in text:
        return True
    if len(name) < 3:
        return False
    return any(name[index : index + 3] in text for index in range(len(name) - 2))


def _first_sentence(text: str, limit: int = 60) -> str | None:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return None
    sentence = re.split(r"[。！？!?；;\n]", cleaned, maxsplit=1)[0].strip()
    return sentence[:limit] or None


def _clean_price_hint(value: Any) -> str | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return str(value)
    text = str(value).strip()
    if not text or text.lower() in {"null", "none", "n/a", "未知"}:
        return None
    return text[:40]


def _clean_image_query(value: Any) -> str | None:
    """Keep an English image keyword, drop anything else.

    The photo library is indexed in English, so a keyword only helps if it really is
    English: a Chinese keyword searched as-is is what returned a photo of Osaka Castle for
    three separate Tokyo museums. A Chinese keyword is therefore discarded rather than
    passed on, and the image step falls back to its own category wording.
    """
    text = " ".join(str(value or "").split())
    if not text or text.lower() in {"null", "none", "n/a", "未知"}:
        return None
    if len(text) > 60:
        return None
    letters = sum(1 for char in text if char.isascii() and char.isalpha())
    if letters < 3:
        return None
    if any("\u4e00" <= char <= "\u9fff" for char in text):
        return None
    return text


# The workspace map draws a straight-line sketch, so a place only has to be in roughly the
# right part of the city -- a few hundred metres of error is invisible at that zoom. What it
# must never do is draw a pin somewhere impossible, so every coordinate is range-checked here
# and a half-answered pair (latitude without longitude) is dropped rather than guessed.
LATITUDE_LIMIT = 85.0
LONGITUDE_LIMIT = 180.0


def _clean_coordinate(value: Any, limit: float) -> float | None:
    """Read one coordinate, or None when the value cannot be trusted as a position."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(str(value).strip())
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    if abs(number) > limit:
        return None
    return round(number, 4)


def _clean_coordinates(item: dict[str, Any]) -> tuple[float | None, float | None]:
    """Read the rough position of a place, keeping it only when both halves are usable."""
    latitude = _clean_coordinate(item.get("latitude"), LATITUDE_LIMIT)
    longitude = _clean_coordinate(item.get("longitude"), LONGITUDE_LIMIT)
    if latitude is None or longitude is None:
        return None, None
    return latitude, longitude


def _clean_unit_price(value: Any) -> int | None:
    """Read a per-unit reference price in CNY, ignoring unusable values."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        price = int(value)
    else:
        match = re.search(r"\d+(?:\.\d+)?", str(value).replace(",", ""))
        if match is None:
            return None
        price = int(float(match.group()))
    return price if price > 0 else None


def _clean_price_unit(value: Any, category: str) -> str:
    unit = str(value or "").strip().lower()
    if unit in PRICE_UNITS:
        return unit
    return DEFAULT_PRICE_UNIT.get(category, "ticket")


def _bounded_unit_price(value: Any, category: str, destination: str | None = None) -> int | None:
    """Re-check a price that is ALREADY in CNY against the bounds, without converting it again.

    ``_plausible_unit_price`` is the right gate for raw material: it has to decide whether "1,480"
    is CNY or yen. It is the wrong gate for a number the pipeline already vetted and stored,
    because running the foreign-currency rule on a CNY figure converts it a second time -- a real
    1,200 CNY omakase dinner in Tokyo became 58 CNY, and then the local floor raised it to 720.
    Everything reaching this function has been through ``_plausible_unit_price`` (or came from the
    default table, which is CNY by construction), so only the range is re-checked.
    """
    price = _clean_unit_price(value)
    if price is None:
        return None
    category = normalize_category(category) or ""
    low, high = PRICE_BOUNDS.get(category, (1, 100000))
    local_ceiling = meal_prices.realistic_ceiling(destination, category)
    if local_ceiling is not None:
        high = max(high, local_ceiling)
    return price if low <= price <= high else None


def _plausible_unit_price(value: Any, category: str, destination: str | None = None) -> int | None:
    """Read a per-unit price and reject values outside a believable range.

    Guards against search material where a phone number, review count or star
    rating is mistaken for a price.

    For a foreign destination the number may still be in the local currency even though
    everything here is CNY: a Tokyo restaurant quoting "1,480" is 71 元, not 1,480 元, and
    1,480 is inside the CNY bounds so nothing else would catch it. Two rules, both confined
    to foreign destinations so no Chinese plan can be affected:

    * a per-meal or per-ticket figure far above what anyone pays in CNY is read as the
      local currency and converted;
    * anything still outside the bounds is converted and re-checked, which is what catches
      a yen hotel rate (30,000) that the CNY ceiling would otherwise reject outright.
    """
    price = _clean_unit_price(value)
    if price is None:
        return None
    category = normalize_category(category) or ""
    low, high = PRICE_BOUNDS.get(category, (1, 100000))
    # A destination-aware ceiling: a 25,000 yen omakase dinner really costs 1,200 CNY, and the
    # flat 2,000 CNY restaurant bound used to throw that price away and replace it with the
    # mid-range default. The bound only ever widens, never tightens.
    local_ceiling = meal_prices.realistic_ceiling(destination, category)
    if local_ceiling is not None:
        high = max(high, local_ceiling)
    local = currency.currency_for_destination(destination) if currency.is_foreign(destination) else None
    if local is not None:
        threshold = FOREIGN_LOCAL_THRESHOLDS.get(category)
        # The trigger for "this must be the local currency" moves with the destination: 1,200 is a
        # real CNY omakase dinner in Tokyo (25,000 yen), and the flat 600 threshold turned it into
        # 58 CNY. Never below the flat threshold, so a plain number still gets converted.
        local_ceiling = meal_prices.realistic_ceiling(destination, category)
        if threshold is not None and local_ceiling is not None:
            threshold = max(threshold, local_ceiling)
        if threshold is not None and price > threshold:
            price = currency.to_cny(price, local)
        if not low <= price <= high:
            price = currency.to_cny(_clean_unit_price(value) or price, local)
    return price if low <= price <= high else None


# Text in the material that means the place costs nothing to enter.
_FREE_MARKERS = ("免费", "免票", "无需门票", "不需门票", "free", "no admission")
# Characters that separate thousands in written prices.
_PRICE_NOISE = re.compile(r"[,\s，]")


def _mentions_free(price_hint: Any) -> bool:
    """True when the quoted price text says the place is free."""
    hint = str(price_hint or "").strip().casefold()
    return bool(hint) and any(marker in hint for marker in _FREE_MARKERS)


def _is_price_grounded(value: int, price_hint: Any, text: str) -> bool:
    """True when a price actually appears in the material it claims to come from.

    Telling the model a budget ceiling made it write that ceiling back as
    ``unit_price`` even when the material quoted no price at all — an 800 CNY
    ceiling produced hotels "quoted" at exactly 800 and meals at exactly 53, each
    labelled "按检索到的价格估算". A number with nothing behind it is not a price, so
    it is rejected here and the caller falls back to the honest default instead.
    """
    needle = str(int(value))
    hint = _PRICE_NOISE.sub("", str(price_hint or ""))
    if hint and needle in hint:
        return True
    return needle in _PRICE_NOISE.sub("", str(text or ""))


def _grounded_unit_price(
    raw_value: Any, price_hint: Any, category: str, text: str, destination: str | None = None
) -> tuple[int | None, bool]:
    """Return (price, is_free) for one item, accepting only grounded numbers.

    A free attraction is a real, useful fact — charging the 100 CNY default for a
    museum the material calls free would make the budget wrong in the traveller's
    favour of spending money they do not owe.

    Grounding is checked against the number **as the material wrote it**, before any
    currency conversion, because that is the number that can actually be found in the
    source. A converted figure is derived from a grounded one, so it inherits the check;
    running the check on the converted value instead would reject every foreign price.
    """
    if normalize_category(category) == "attraction" and _mentions_free(price_hint):
        return 0, True
    raw = _clean_unit_price(raw_value)
    stated = currency.hint_currency(price_hint, destination)
    # A price the material states in CNY, foreign quotation beside it, is the source's own
    # conversion and beats ours: "630日元（约32元人民币）" is 32. It is checked against the
    # CNY bounds only -- running it through the foreign rule converted the source's own
    # figure a second time (¥598元/人（12,400日元） became 29), which is worse than either
    # number on its own.
    explicit = currency.explicit_cny_amount(price_hint)
    if explicit is not None and stated not in (None, "CNY"):
        price = _plausible_unit_price(explicit, category)
        if price is not None and (
            raw is None
            or _is_price_grounded(explicit, price_hint, text)
            or _is_price_grounded(raw, price_hint, text)
        ):
            return price, False
    # Failing that, read the amount out of the source's own price line whenever the model's
    # number is missing or is not the number that line quotes. Both happened in a measured
    # Tokyo run: "1" for a 620 yen ticket, and no answer at all for several museums, which
    # fell back to the 100 CNY category default.
    marked = currency.marked_amount(price_hint, destination)
    if marked is not None and (raw is None or not _is_price_grounded(raw, price_hint, "")):
        amount, unit = marked
        candidate = amount if unit == "CNY" else currency.to_cny(amount, unit)
        price = _plausible_unit_price(candidate, category)
        if price is not None:
            return price, False
    if raw is None:
        return None, False
    converted = currency.convert_foreign_number(raw, price_hint, destination)
    if converted is not None:
        price = _plausible_unit_price(converted[0], category)
        if price is not None and _is_price_grounded(raw, price_hint, text):
            return price, False
    price = _plausible_unit_price(raw, category, destination)
    if price is None:
        return None, False
    if not _is_price_grounded(price, price_hint, text):
        return None, False
    return price, False


def _normalize_item(item: Any, text: str, destination: str) -> dict[str, Any] | None:
    """Validate one model item against the grounded material."""
    if not isinstance(item, dict):
        return None
    name = " ".join(str(item.get("name") or "").split())
    if not name or len(name) > NAME_CHARS:
        return None
    category = normalize_category(item.get("category")) or _classify_keywords(name)
    if not category:
        return None
    # Reject a descriptive heading before grounding sees it: the 3-character test
    # below cannot catch "夫子庙地铁站附近酒店", because "夫子庙" really is in the
    # material. See `_is_grounded`.
    if not _is_grounded(name, text):
        return None
    price_hint = _clean_price_hint(item.get("price_hint"))
    unit_price, is_free = _grounded_unit_price(
        item.get("unit_price"), price_hint, category, text, destination
    )
    latitude, longitude = _clean_coordinates(item)
    return {
        "name": name,
        "type": category,
        "address": destination,
        "price_level": DEFAULT_PRICE_LEVELS.get(category, 2),
        "image_url": None,
        # A rough position, so the workspace map can sketch where this place is and how far
        # it is from the next one. It is the model's estimate, not a surveyed address.
        "latitude": latitude,
        "longitude": longitude,
        # English keywords for the image search: the photo library is indexed in English,
        # and a Chinese venue name searched as-is returned pictures of other cities.
        "image_query": _clean_image_query(item.get("image_query")),
        "reason": _first_sentence(item.get("reason"), limit=80),
        "price_hint": price_hint,
        "unit_price": unit_price,
        "price_unit": _clean_price_unit(item.get("price_unit"), category),
        # Only restaurants carry a meal slot; it is what lets the UI split breakfast
        # from lunch and dinner instead of showing one undifferentiated food list.
        "meal_slot": classify_meal_slot(name, item.get("meal_slot")) if category == "restaurant" else None,
        "is_free": is_free,
    }


def _dedupe_and_cap(
    items: list[dict[str, Any]],
    limit_per_category: int | None = None,
    limits: dict[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Drop duplicates and cap each category.

    `limits` gives a per-category cap (restaurants need more cards than hotels, since
    they have to cover three meals); `limit_per_category` is the uniform cap used when
    a caller wants the same number for everything.
    """
    kept: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts: dict[str, int] = {}
    for item in items:
        name = str(item.get("name") or "").strip()
        key = name.casefold()
        if not name or key in seen:
            continue
        category = str(item.get("type") or "")
        if limits is not None:
            cap = limits.get(category, limit_per_category if limit_per_category is not None else 3)
        else:
            cap = limit_per_category if limit_per_category is not None else 3
        if counts.get(category, 0) >= cap:
            continue
        seen.add(key)
        counts[category] = counts.get(category, 0) + 1
        kept.append(item)
    return kept


def _sort_by_category(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order = {category: index for index, category in enumerate(CATEGORY_ORDER)}
    return sorted(items, key=lambda item: order.get(str(item.get("type") or ""), len(CATEGORY_ORDER)))


def _salvage_json_array(text: str) -> list[Any] | None:
    """Keep the complete places from an answer that was cut off mid-array.

    A provider that runs into its output ceiling stops mid-object, so there is no closing
    bracket and the whole answer used to be thrown away. Losing a nearly finished list is
    the expensive outcome: the plan then fell back to keyword rules, which surfaced page
    and exhibition titles as place names, and every place arrived without a grounded
    price. Completing the bracket after the last whole object recovers all of them.

    The scan tracks string state as well as depth, so a brace or bracket inside a value
    ("招牌菜 [时令]") cannot be mistaken for the end of an object.
    """
    depth = 0
    in_string = False
    escaped = False
    last_complete = -1
    for index, character in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue
        if character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
        elif character in "]}":
            depth -= 1
            if character == "}" and depth == 1:
                last_complete = index
    if last_complete == -1:
        return None
    try:
        parsed = json.loads(text[: last_complete + 1] + "]")
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, list) and parsed else None


def _parse_json_array(content: str) -> list[Any] | None:
    text = str(content or "").strip()
    if not text:
        return None
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    if start == -1:
        return None
    end = text.rfind("]")
    if end > start:
        try:
            parsed = json.loads(text[start : end + 1])
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, list):
            return parsed
    return _salvage_json_array(text[start:])


def _count_hint(limits: dict[str, int] | None) -> str:
    """Tell the model how many of each category this trip actually needs.

    Without this the answer was whatever the model felt like producing — often three
    or four places in total — and a five-day trip then had to reuse them. The number is
    deliberately modest: a larger single answer is not returned by this provider, and
    the rest of the pool is gathered afterwards by the per-category batch lookups.
    """
    if not limits:
        return ""
    parts = []
    if limits.get("attraction"):
        parts.append(f"景点 {min(limits['attraction'], MAX_ITEMS_PER_CALL)} 个")
    if limits.get("restaurant"):
        parts.append(f"餐厅 {min(limits['restaurant'], MAX_ITEMS_PER_CALL)} 家（含早餐店）")
    if limits.get("hotel"):
        parts.append(f"酒店 {min(limits['hotel'], MAX_ITEMS_PER_CALL)} 家")
    if not parts:
        return ""
    return (
        "**本次输出需要："
        + "、".join(parts)
        + "。所有项目必须互不相同，不得重复同一个地点。**"
    )


def _extract_with_model(
    destination: str,
    material: list[dict[str, Any]],
    allowances: dict[str, int] | None = None,
    limits: dict[str, int] | None = None,
) -> list[Any] | None:
    """Return the model's raw item list, or None when the model is unavailable."""
    client = model_client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": _SYSTEM_PROMPT
                    + currency.prompt_hint(destination)
                    # The same local price reality and budget tiers the focused extraction gets:
                    # this is the first answer the pool is built from, so leaving it with only a
                    # flat ceiling is what produced a mid-range list with no ends on it.
                    + meal_prices.price_hint_for_prompt(destination)
                    + (
                        meal_prices.tier_brief(
                            meal_prices.tier_plan(
                                (allowances or {}).get("restaurant") or 0,
                                floor=meal_prices.realistic_floor(destination, "restaurant") or 0,
                            ),
                            "restaurant",
                        )
                        or ""
                    )
                    + _count_hint(limits),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "destination": destination,
                            "needed": limits,
                            "budget_hint": _budget_hint(allowances),
                            "material": material,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.1,
            extra_body={"enable_thinking": False},
            # The shared client bounds every call at 30s so a stalled provider cannot hold
            # a request open for minutes, but a list of recommendations is legitimately
            # slower than that: the answer was thrown away and the plan silently fell back
            # to keyword rules. These calls get their own, still bounded, allowance.
            timeout=EXTRACTION_TIMEOUT,
            # This one call covers every category at once, so it needs room for the whole
            # mixed answer. At 1600 the reply was cut off mid-JSON (finish_reason=length),
            # parsing failed, and the plan silently fell back to keyword rules — which
            # surfaced page and exhibition titles ("玉堂佳器-馆藏精品展") as place names.
            max_tokens=MIXED_MAX_TOKENS,
        )
        choice = response.choices[0]
        content = choice.message.content or ""
    except Exception as exc:
        # A silent fallback here is indistinguishable from a thin search result, which
        # made a dead model key look like "the search found little" for a whole round.
        logger.warning("recommendation extraction call failed: %s", exc)
        return None
    parsed = _parse_json_array(content)
    if parsed is None:
        logger.warning(
            "recommendation extraction was not usable JSON (finish_reason=%s, chars=%d)",
            getattr(choice, "finish_reason", None),
            len(content),
        )
    elif getattr(choice, "finish_reason", None) == "length":
        # Recovered, not lost -- but the trip is still short of the places the prompt asked
        # for, and that is worth knowing rather than discovering as a thin itinerary.
        logger.warning(
            "recommendation extraction was cut off by the output ceiling; kept %d places",
            len(parsed),
        )
    return parsed


def _fallback_items(material: list[dict[str, Any]], destination: str, text: str) -> list[dict[str, Any]]:
    """Keyword fallback used only when the chat model is unavailable."""
    items: list[dict[str, Any]] = []
    for block in material:
        name = " ".join(str(block["title"]).split())[:NAME_CHARS]
        category = _classify_keywords(f"{name} {block['snippet']}")
        if not name or not category or not _is_grounded(name, text):
            continue
        items.append(
            {
                "name": name,
                "type": category,
                "address": destination,
                "price_level": DEFAULT_PRICE_LEVELS.get(category, 2),
                "image_url": None,
                "reason": _first_sentence(block["snippet"]),
                "price_hint": None,
                "unit_price": None,
                "price_unit": DEFAULT_PRICE_UNIT.get(category, "ticket"),
                "meal_slot": classify_meal_slot(name) if category == "restaurant" else None,
            }
        )
    return items


_PRICE_QUERY_HINTS = {
    "hotel": "房价 每晚 多少钱",
    "restaurant": "人均消费 多少钱",
    "attraction": "门票价格 多少钱",
}

_PRICE_SYSTEM_PROMPT = (
    "你是一名旅行成本分析员。你会收到若干地点及其检索到的网页资料，"
    "请为每个地点给出一个参考单价（整数人民币元）。"
    "只输出 JSON 对象，形如 {\"地点名称\": 数字}，不要输出解释或代码块标记。"
    "hotel 填每晚房价；restaurant 填每人每餐人均；attraction 填每人门票。"
    "资料里的价格若是外币，必须按上面给出的汇率换算成人民币整数再填；"
    "不要直接照抄外币数字。"
    "价格必须能从资料中推断出来；资料完全没有价格线索时填 null，禁止凭空编造。"
    "只输出输入中出现过的地点名称，不要新增地点。"
)


def _price_query(name: str, category: str, destination: str) -> str:
    hint = _PRICE_QUERY_HINTS.get(category, "价格")
    return f"{destination} {name} {hint}".strip()


def _lookup_price_text(name: str, category: str, destination: str) -> str:
    """Gather price material for one place; never raises."""
    try:
        snippets = search_snippets(
            _price_query(name, category, destination), limit=3, timeout=PRICE_LOOKUP_TIMEOUT
        )
    except Exception:
        return ""
    return " ".join(snippet for snippet in snippets if snippet)[:SNIPPET_CHARS]


def _extract_prices_with_model(destination: str, material: list[dict[str, Any]]) -> dict[str, int] | None:
    client = model_client()
    if client is None or not material:
        return None
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": _PRICE_SYSTEM_PROMPT + currency.prompt_hint(destination)},
                {
                    "role": "user",
                    "content": json.dumps({"destination": destination, "places": material}, ensure_ascii=False),
                },
            ],
            temperature=0.0,
            extra_body={"enable_thinking": False},
            max_tokens=800,
        )
        content = response.choices[0].message.content or ""
    except Exception:
        return None
    parsed = _parse_json_object(content)
    if parsed is None:
        return None
    prices: dict[str, int] = {}
    for key, value in parsed.items():
        price = _clean_unit_price(value)
        if price is not None:
            prices[str(key).strip()] = price
    return prices

def _parse_json_object(content: str) -> dict[str, Any] | None:
    """Kept as the local name this module has always used; the parser itself is shared."""
    return parse_json_object(content)


def enrich_prices_with_search(
    items: list[dict[str, Any]],
    destination: str,
    max_lookups: int = MAX_IMAGE_LOOKUPS,
) -> list[dict[str, Any]]:
    """Fill in missing unit prices from a per-place search.

    Only items without a price are looked up, one bounded search each, and a single
    batched model call turns that material into numbers. Any failure leaves the item
    unpriced so the caller can fall back to a labelled default estimate.
    """
    # A place the material calls free is already priced at 0; looking it up again
    # would only risk replacing a known fact with a guess.
    pending = [
        item
        for item in items
        if not item.get("is_free") and _clean_unit_price(item.get("unit_price")) is None
    ][:max_lookups]
    if not pending:
        return items
    pool = ThreadPoolExecutor(max_workers=min(PRICE_LOOKUP_WORKERS, len(pending)))
    try:
        futures = [
            pool.submit(
                _lookup_price_text, str(item.get("name") or ""), str(item.get("type") or ""), destination
            )
            for item in pending
        ]
        # A hard wall-clock budget: an httpx timeout does not cover DNS resolution,
        # so a stalling search provider must not be able to hold the request open.
        done, not_done = wait(futures, timeout=PRICE_PHASE_BUDGET)
        texts: list[str] = []
        for future in futures:
            if future in done:
                try:
                    texts.append(future.result())
                except Exception:
                    texts.append("")
            else:
                future.cancel()
                texts.append("")
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    material = [
        {"name": item.get("name"), "category": item.get("type"), "material": text}
        for item, text in zip(pending, texts)
        if text.strip()
    ]
    prices = _extract_prices_with_model(destination, material) or {}
    for item in pending:
        name = str(item.get("name") or "")
        if name in prices:
            # Bound against the item's own category: search material regularly
            # contains phone numbers and review counts that are not prices.
            item["unit_price"] = _plausible_unit_price(
                prices[name], str(item.get("type") or ""), destination
            )
    return items


def _cost_formula(
    category: str, unit_price: int, unit: str, days: int, nights: int, people: int, count: int = 1
) -> tuple[int, str]:
    """Return (amount, explanation) for one place scaled to this trip.

    ``count`` is how many times the traveller uses THIS place: nights slept at this
    hotel, or meals eaten at this restaurant. It deliberately starts at 1.

    A restaurant used to be charged ``unit_price * days * people``, i.e. as if the same
    dish were eaten there for every meal of the whole trip. Ticking one restaurant is
    usually one meal (two or three at most), so a five-day trip for two quoted a single
    restaurant at 5x what it would cost -- and the same double-charging showed up in the
    running total. The same reasoning gives a hotel ONE night by default, since a trip may
    move between hotels.
    """
    del days
    if unit == "night" or category == "hotel":
        amount = unit_price * count
        return amount, f"约 {unit_price} 元/晚 × {count} 晚 = {amount} 元"
    if unit == "person_meal" or category == "restaurant":
        amount = unit_price * people * count
        return amount, f"约 {unit_price} 元/人/餐 × {people} 人 × {count} 次 = {amount} 元"
    amount = unit_price * people
    return amount, f"约 {unit_price} 元/人 × {people} 人 = {amount} 元"


# How many times one place is used before the traveller says otherwise. Both start at one:
# a second hotel is another booking, and a restaurant is one meal (the traveller can raise
# it on the card, and the budget follows).
DEFAULT_QUANTITY = {"hotel": 1, "restaurant": 1}
# The unit the count is expressed in, for the card and for what gets saved.
QUANTITY_UNITS = {"hotel": "晚", "restaurant": "次"}


# How much of the traveller's budget each category may reasonably claim. Lodging
# dominates a real trip; three meals a day is the next biggest line.
CATEGORY_BUDGET_SHARE = {"hotel": 0.40, "restaurant": 0.30, "attraction": 0.20}
# Meals assumed per day when working out what one meal may cost.
MEALS_PER_DAY = 3


def budget_allowances(
    trip_budget: int | None,
    trip_days: int,
    traveler_profile: dict[str, Any] | None = None,
) -> dict[str, int]:
    """The most one *unit* of each category may cost to still fit the budget.

    Recommendations used to be picked with no idea what the trip cost: an 8000 CNY
    five-day trip was offered hotels at 3840 and 4800 (one hotel = 60% of the whole
    budget). These ceilings are derived from the trip's own budget and are handed to
    the extraction prompt so affordable options are preferred, and used afterwards to
    label anything that still exceeds them.

    Returns an empty dict when the trip has no budget, in which case nothing is
    filtered or labelled.
    """
    try:
        budget = int(trip_budget or 0)
    except (TypeError, ValueError):
        return {}
    if budget <= 0:
        return {}

    days = max(1, int(trip_days or 1))
    nights = max(1, days - 1)
    people = max(1, int((traveler_profile or {}).get("total_people") or 1))

    allowances: dict[str, int] = {}
    hotel_share = CATEGORY_BUDGET_SHARE["hotel"]
    # One room is assumed, which matches how the hotel cost is charged today.
    allowances["hotel"] = max(1, int(budget * hotel_share / nights))
    meal_share = CATEGORY_BUDGET_SHARE["restaurant"]
    allowances["restaurant"] = max(1, int(budget * meal_share / (days * MEALS_PER_DAY * people)))
    ticket_share = CATEGORY_BUDGET_SHARE["attraction"]
    allowances["attraction"] = max(1, int(budget * ticket_share / people))
    return allowances


def _budget_hint(allowances: dict[str, int] | None) -> str | None:
    """One line telling the model what each category may affordably cost."""
    if not allowances:
        return None
    parts = []
    if "hotel" in allowances:
        parts.append(f"酒店每晚不超过约 {allowances['hotel']} 元")
    if "restaurant" in allowances:
        parts.append(f"每餐每人不超过约 {allowances['restaurant']} 元")
    if "attraction" in allowances:
        parts.append(f"门票每人不超过约 {allowances['attraction']} 元")
    if not parts:
        return None
    return (
        "本次旅行的预算上限参考：" + "；".join(parts) + "。"
        "请在资料允许的范围内优先挑选符合该预算的选项；"
        "如果资料里只有更贵的，也可以给出，但必须如实填写资料中的价格。"
        "**budget_hint 只是筛选参考，不是价格：绝对不能把 budget_hint 里的数字填进 "
        "unit_price。unit_price 必须逐字取自 material 原文中的数字；"
        "material 里没有价格就填 null。**"
    )


# A plan should cover a whole day of eating, so restaurants are not capped like the
# other categories: the user asked for at least three meals a day with breakfast
# kept separate from lunch and dinner.
CATEGORY_LIMITS = {"hotel": 3, "restaurant": 6, "attraction": 4}
# How many candidates a whole trip needs so that no day repeats another: a morning and
# an afternoon sight, three different meals, and a few hotels to choose between.
# Four attractions over five days meant the fifth day had to reuse the first one.
SIGHTS_PER_DAY = 2
HOTEL_CHOICES = 3
# Labels are imperfect: a venue the model does not call a breakfast shop counts as a
# main-meal option, so the pool is gathered with slack rather than exactly the minimum.
# A five-day trip needs five breakfasts and ten main-meal venues; aiming at exactly
# fifteen left the last day with nothing for lunch or dinner.
CATEGORY_SLACK = {"restaurant": 6, "attraction": 4}
# Upper bounds keep one request's JSON answer from growing without limit; a 30-day trip
# does not need ninety restaurants to be useful.
MAX_ATTRACTIONS = 18
MAX_RESTAURANTS = 27


def category_limits(trip_days: int) -> dict[str, int]:
    """How many candidates to gather so every day of the trip can differ.

    The daily plan gives each day a morning sight, an afternoon sight and three
    meals, and never offers the same place twice. That is only possible when the
    candidate pool is at least as large as the itinerary, so the targets scale with
    the trip instead of being a fixed small number.
    """
    days = max(1, int(trip_days or 1))
    return {
        "hotel": HOTEL_CHOICES,
        "restaurant": min(days * MEALS_PER_DAY + CATEGORY_SLACK["restaurant"], MAX_RESTAURANTS),
        "attraction": min(days * SIGHTS_PER_DAY + CATEGORY_SLACK["attraction"], MAX_ATTRACTIONS),
    }


# Breakfast venues are recognisable from their own names; anything unlabelled is
# treated as a lunch/dinner option rather than guessed at.
BREAKFAST_KEYWORDS = (
    "早餐", "早饭", "早茶", "包子", "汤包", "豆浆", "油条", "锅贴", "煎饼",
    "粥", "烧饼", "蒸饺", "小笼", "糕团", "馄饨", "breakfast",
)
MEAL_SLOTS = ("breakfast", "lunch", "dinner", "any")


def classify_meal_slot(name: str, declared: str | None = None) -> str:
    """Work out whether a restaurant is a breakfast place or a lunch/dinner one.

    The model is asked to declare this, but its answer is only trusted when it is a
    value we know; otherwise the name itself decides, and anything unrecognised is
    left as a general option instead of being mislabelled.
    """
    slot = str(declared or "").strip().lower()
    slot_map = {
        "早餐": "breakfast", "早饭": "breakfast", "breakfast": "breakfast",
        "午餐": "lunch", "午饭": "lunch", "中餐": "lunch", "lunch": "lunch",
        "晚餐": "dinner", "晚饭": "dinner", "dinner": "dinner",
        "any": "any", "正餐": "any", "": "any",
    }
    if slot in slot_map:
        resolved = slot_map[slot]
        if resolved != "any":
            return resolved
    folded = str(name or "").casefold()
    if any(keyword in folded for keyword in BREAKFAST_KEYWORDS):
        return "breakfast"
    return "any"


# Every place has to come back with somewhere a person could actually go. The extraction used
# to ignore the address, and the plan then carried "地址待确认" into the itinerary and the
# exported markdown, which tells the traveller nothing. The material being read usually does
# contain the address, so asking for it here costs nothing.
_ADDRESS_HINT = (
    "\n每个地点都必须带 address 字段，写成能直接导航的形式：所在区域 + 街道/门牌，"
    "或最靠近的车站与出口（例如「东京都台东区上野1-1-1」「涩谷站 八公口 步行3分钟」）。"
    "禁止只写城市名、禁止写「未知」「待确认」；素材里确实没有地址时，写所属街区或最近的地标。"
)


def _extract_category_with_model(
    destination: str,
    category: str,
    material: list[dict[str, Any]],
    allowances: dict[str, int] | None = None,
    wanted: int | None = None,
    slot: str | None = None,
    exclude: tuple[str, ...] = (),
    tier_plan: meal_prices.TierPlan | None = None,
) -> list[Any] | None:
    """Run a category-focused extraction, or None when the model is unavailable."""
    client = model_client()
    if client is None:
        return None
    # Asking for a specific number is what makes a top-up actually arrive: the older
    # prompt said "as many as possible", and a thin category stayed thin.
    count_hint = f"请给出至少 {wanted} 个互不相同的选项。" if wanted else ""
    slot_hint = SLOT_INSTRUCTIONS.get(slot or "", "")
    # What eating out actually costs where the traveller is going, plus the three price tiers
    # their own budget implies. Both are行情 hints: the number itself must still come from the
    # material, so a card can never quote a band as if the venue had published it.
    price_hint = (
        meal_prices.price_hint_for_prompt(destination)
        + (meal_prices.tier_brief(tier_plan, category) or "" if tier_plan is not None else "")
    ) if category in ("restaurant", "hotel") else ""
    # Later batches read the same search material as the earlier ones, so they are told
    # what has already been used; otherwise every batch returns the same famous few.
    exclude_hint = (
        f"以下地点已经选过，一律不要再列：{'、'.join(exclude)}。" if exclude else ""
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",
                    "content": _CATEGORY_PROMPTS[category]
                    + _CATEGORY_TAIL
                    + _ADDRESS_HINT
                    + currency.prompt_hint(destination)
                    + price_hint
                    + count_hint
                    + slot_hint
                    + exclude_hint,
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "destination": destination,
                            "category": category,
                            "needed": wanted,
                            "meal_slot": slot,
                            "already_used": list(exclude),
                            "budget_hint": _budget_hint(allowances),
                            "material": material,
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.1,
            extra_body={"enable_thinking": False},
            timeout=EXTRACTION_TIMEOUT,
            # A batch is small, so this ceiling is about never truncating mid-JSON rather
            # than about allowing a long list.
            max_tokens=BATCH_MAX_TOKENS,
        )
        choice = response.choices[0]
        content = choice.message.content or ""
    except Exception as exc:
        logger.warning("focused %s extraction call failed: %s", category, exc)
        return None
    parsed = _parse_json_array(content)
    if parsed is None:
        logger.warning(
            "focused %s extraction was not usable JSON (finish_reason=%s, chars=%d)",
            category,
            getattr(choice, "finish_reason", None),
            len(content),
        )
    elif getattr(choice, "finish_reason", None) == "length":
        # Salvaged rather than lost, but the batch is still short of what was asked for.
        logger.warning(
            "focused %s extraction was cut off by the output ceiling; kept %d places",
            category,
            len(parsed),
        )
    return parsed


def _live_category_material(
    destination: str, category: str, hint: str | None = None
) -> list[dict[str, Any]] | None:
    """Material from the model's own live search, or None when it is unavailable.

    Preferred over the search provider because it answers in Chinese with current
    information and does not spend the provider's per-request quota, which a multi-day
    plan exhausts quickly. Returns None on any failure so the caller falls back.
    """
    if not dashscope_search.search_available():
        return None
    question = _live_search_question(category, hint)
    answer = dashscope_search.grounded_answer(
        f"{destination}有哪些{question}？"
        f"请列出至少 {LIVE_MATERIAL_ITEMS} 个真实存在的地点原名，每个附一句简短说明；"
        "资料里若有门票或人均价格，请写出价格原文。名称一律用中文，不要翻译成其他语言。"
    )
    if not answer:
        return None
    text = answer.strip()
    if len(text) < MIN_MATERIAL_CHARS:
        return None
    # A live answer is the whole material for this category rather than one snippet among
    # several, so it is kept in full: cutting it at SNIPPET_CHARS removed half the list.
    return [{"title": f"{destination}{category}", "snippet": text[:LIVE_MATERIAL_CHARS]}]


def _category_material(destination: str, category: str, hint: str | None = None) -> list[dict[str, Any]]:
    """Gather material for one category; never raises.

    The cache is keyed by the question the search is actually asked, not by the hint that
    asked it. Several hints collapse onto one question -- ``_live_search_question`` sends the
    plain category question for the price-tier wording ("便宜 实惠 当地人常去…" and
    "米其林 名店…" both become "本地人常去的餐厅、小吃店和老字号") -- and keying on the hint
    paid for that identical answer once per tier. Measured in a four-day plan: the two tier
    backfills re-searched wording-for-wording what the main restaurant group had already
    fetched, and those two searches sat on the critical path. The tier filter that uses them
    is applied afterwards, to the extracted places, so the material itself is unchanged.
    """
    destination = str(destination or "").strip()
    category = str(category or "").strip()
    hint = str(hint or CATEGORY_SEARCH_HINTS.get(category, category)).strip()
    search_key = _live_search_question(category, hint)
    cached = _read_category_material_cache(destination, category, search_key)
    if cached is not None:
        return cached
    live = _live_category_material(destination, category, hint)
    if live:
        _write_category_material_cache(destination, category, search_key, live)
        return live
    try:
        snippets = [
            str(snippet).strip()
            for snippet in search_snippets(
                f"{destination} {search_key}", limit=5, timeout=MATERIAL_FALLBACK_TIMEOUT
            )
        ]
    except Exception:
        return []
    snippets = [snippet for snippet in snippets if snippet]
    # Prefer the Chinese pages: a name read off an English page comes back in English, and
    # the same venue then appeared once per language ("咸亨酒店" / "Xian Heng Restaurant").
    # English snippets are still used when nothing else was found.
    chinese = [snippet for snippet in snippets if _has_cjk(snippet)]
    text = " ".join(chinese or snippets)
    if len(text) < MIN_MATERIAL_CHARS:
        return []
    material = [{"title": f"{destination}{hint}", "snippet": text[:SNIPPET_CHARS]}]
    _write_category_material_cache(destination, category, search_key, material)
    return material


def _category_batch_plan(category: str, shortfall: int) -> list[tuple[str, list[tuple[int, str | None]]]]:
    """Split a shortfall into batches, grouped by the search they share.

    Returns ``[(search hint, [(batch size, meal slot), ...]), ...]``. A search is charged
    per request, so batches with the same hint share one search and are told to avoid the
    names the earlier batch produced.
    """
    hints = CATEGORY_SEARCH_VARIANTS.get(category) or ((CATEGORY_SEARCH_HINTS.get(category, category), None),)
    if category == "restaurant" and len(hints) > 1:
        # A day needs three meals, one of them breakfast, so a third of the pool is
        # reserved for breakfast shops. Filling the batches greedily gave every one of
        # them to main meals, and the mornings then had nothing to offer.
        breakfast_total = max(1, shortfall // MEALS_PER_DAY)
        allocation = [
            (hints[0][0], shortfall - breakfast_total, "main"),
            (hints[-1][0], breakfast_total, "breakfast"),
        ]
    else:
        allocation = [(hints[0][0], shortfall, hints[0][1])]

    groups: list[tuple[str, list[tuple[int, str | None]]]] = []
    for hint, total, slot in allocation:
        remaining = total
        while remaining > 0:
            batch = min(remaining, MAX_ITEMS_PER_CALL)
            if groups and groups[-1][0] == hint:
                groups[-1][1].append((batch, slot))
            else:
                groups.append((hint, [(batch, slot)]))
            remaining -= batch
    return groups


def _fill_category(
    destination: str,
    category: str,
    shortfall: int,
    allowances: dict[str, int] | None = None,
    plan: meal_prices.TierPlan | None = None,
    search_hint: str | None = None,
    exclude: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Gather candidates for one category. Never raises.

    Each group searches once and then runs its batches one after another, because a later
    batch can only avoid repeating the earlier ones if it knows what they returned.

    `search_hint` replaces the category's usual wording, which is how a missing *price tier*
    is looked for by name ("米其林 名店" instead of "美食 推荐").

    ``exclude`` names places that are already spoken for — what a single day's plan passes
    so that it cannot offer something another day already has. They are treated exactly like
    places this call has already collected: the model is told not to list them, and anything
    it lists anyway is dropped rather than returned.
    """
    collected: list[dict[str, Any]] = []
    taken: set[str] = {str(name).strip().casefold() for name in exclude if str(name).strip()}
    remaining = shortfall
    batches_plan = (
        [(search_hint, [(shortfall, "any")])]
        if search_hint
        else _category_batch_plan(category, shortfall)
    )
    for hint, batches in batches_plan:
        material = _category_material(destination, category, hint)
        if not material:
            logger.warning("no material found for the %s search (%s)", category, hint)
            continue
        text = _material_text(material)
        for batch_size, slot in batches:
            if remaining <= 0:
                break
            wanted = min(batch_size, MAX_ITEMS_PER_CALL)
            extracted = _extract_category_with_model(
                destination, category, material, allowances, wanted, slot, tuple(sorted(taken)), plan
            )
            if not extracted:
                continue
            added = 0
            for raw in extracted:
                normalized = _normalize_item(raw, text, destination)
                if normalized is None:
                    continue
                key = str(normalized["name"]).strip().casefold()
                if key in taken:
                    continue
                # A focused prompt can still return the wrong category label; trust the
                # category we searched for so the card lands in the right group.
                normalized["type"] = category
                normalized["price_unit"] = DEFAULT_PRICE_UNIT.get(category, "ticket")
                taken.add(key)
                collected.append(normalized)
                added += 1
            remaining -= added
            if added == 0:
                # Nothing new came back, so another batch over the same search would
                # return the same places again.
                break
    return collected


def ensure_category_coverage(
    destination: str,
    items: list[dict[str, Any]],
    limit_per_category: int | None = None,
    max_fills: int = MAX_CATEGORY_FILLS,
    allowances: dict[str, int] | None = None,
    limits: dict[str, int] | None = None,
    on_category_ready: Callable[[str, list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    """Backfill the categories that are missing, or thin, for this trip.

    With `limits` (the day-aware targets) a category is topped up until it holds
    enough candidates for every day of the trip to differ; without it the historic
    behaviour is kept and only categories with nothing at all are searched.

    A short category needs more candidates than one answer will hold — asking for fifteen
    at once was answered with a timeout or a flat rejection — so it is accumulated in
    batches. The categories are gathered concurrently; the searches themselves are shared
    between batches, because the search plan is charged per request and one search per
    batch exhausted it in an afternoon of testing.
    """
    counts: dict[str, int] = {}
    for item in items:
        category = str(item.get("type") or "")
        counts[category] = counts.get(category, 0) + 1

    if limits is None:
        wanted = [
            (category, limit_per_category if limit_per_category is not None else CATEGORY_LIMITS.get(category, 3))
            for category in CATEGORY_ORDER
            if not counts.get(category)
        ]
    else:
        wanted = [
            (category, limits[category] - counts.get(category, 0))
            for category in CATEGORY_ORDER
            if limits.get(category, 0) > counts.get(category, 0)
        ]
    wanted = wanted[:max_fills]
    if not wanted:
        return items

    existing_by_category = {
        category: [item for item in items if str(item.get("type") or "") == category]
        for category, _ in wanted
    }

    def fill_and_report(entry: tuple[str, int]) -> list[dict[str, Any]]:
        category, shortfall = entry
        added = _fill_category(destination, category, shortfall, allowances)
        if on_category_ready is not None:
            on_category_ready(category, [*existing_by_category.get(category, []), *added])
        return added

    with ThreadPoolExecutor(max_workers=min(len(wanted), LOOKUP_WORKERS)) as pool:
        found = list(
            pool.map(
                fill_and_report,
                wanted,
            )
        )

    seen = {str(item.get("name") or "").strip().casefold() for item in items}
    for (category, shortfall), added in zip(wanted, found):
        added_count = 0
        for normalized in added:
            # Take only what this category was short of. In the historic path that is the
            # category cap; with day-aware targets it is the shortfall, so a category that
            # already holds most of what it needs is not overfilled.
            if added_count >= shortfall:
                break
            # Never show the same place twice: a second category's search can surface a
            # place that is already on the list.
            key = str(normalized["name"]).strip().casefold()
            if key in seen:
                continue
            seen.add(key)
            items.append(normalized)
            added_count += 1
    return items


def ensure_price_spread(
    destination: str,
    items: list[dict[str, Any]],
    trip_budget: int | None,
    trip_days: int,
    traveler_profile: dict[str, Any] | None = None,
    max_fills: int = 2,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Make sure the food list really is a mix, and say how it came out.

    A prompt that asks for three tiers is a request, not a guarantee: measured answers were all
    mid-range, or all cheap, which is exactly what the traveller complained about. So the pool is
    counted after pricing, and the tiers that are thin are searched for *by name* -- "米其林 名店"
    finds the expensive-but-good option that "美食 推荐" does not.

    Only worth doing when the trip has a budget: with no budget there is no allowance to tier
    against, and inventing tiers would be inventing money.

    Returns the (possibly grown) list and the final per-tier counts.
    """
    allowances = budget_allowances(trip_budget, trip_days, traveler_profile)
    allowance = allowances.get("restaurant")
    if not allowance:
        return items, {}
    floor = meal_prices.realistic_floor(destination, "restaurant") or 0
    plan = meal_prices.tier_plan(allowance, floor=floor)
    if plan is None:
        return items, {}
    counts = meal_prices.ensure_tier_spread(items, plan, "restaurant")
    total = sum(1 for item in items if normalize_category(item.get("type")) == "restaurant")
    missing = meal_prices.missing_tiers(counts, total)
    if not missing:
        return items, counts

    # Most wanted first: the expensive-but-good tier is the one a thin search never produces,
    # and the cheap one is what makes the list affordable day after day.
    order = {meal_prices.TIER_SPLURGE: 0, meal_prices.TIER_CHEAP: 1, meal_prices.TIER_MID: 2}
    wanted = sorted(missing, key=lambda tier: order.get(tier, 9))[:max_fills]
    seen = {str(item.get("name") or "").strip().casefold() for item in items}
    for tier in wanted:
        needed = max(1, int(total * meal_prices.TIER_TARGET_SHARES.get(tier, 0.2)) - counts.get(tier, 0))
        found = _fill_category(
            destination,
            "restaurant",
            needed,
            allowances,
            plan,
            meal_prices.tier_search_hint(tier),
        )
        added = 0
        for normalized in found:
            key = str(normalized.get("name") or "").strip().casefold()
            if not key or key in seen:
                continue
            # Only keep what was actually missing: a "cheap" search that returns another
            # mid-range place has not fixed anything.
            priced = _plausible_unit_price(normalized.get("unit_price"), "restaurant", destination)
            if priced is None or plan.tier_of(priced) != tier:
                continue
            seen.add(key)
            items.append(normalized)
            added += 1
        if added:
            logger.info("price spread: added %d %s restaurants", added, tier)
    counts = meal_prices.ensure_tier_spread(items, plan, "restaurant")
    return items, counts


def estimate_trip_costs(
    items: list[dict[str, Any]],
    trip_days: int,
    traveler_profile: dict[str, Any] | None = None,
    trip_budget: int | None = None,
    destination: str | None = None,
) -> list[dict[str, Any]]:
    """Attach an approximate cost per place, scaled to this trip's length and party.

    Prices come from the grounded search material; when the material carried no
    price a destination-shaped default is used and the explanation says so, so a reader is
    never misled into treating a default as a quoted price.

    A price that cannot buy a full meal where the traveller is going is raised to the local
    floor for that kind of place and labelled as adjusted. Measured before this existed: a
    Tokyo sushi restaurant was shown at 53 CNY per person, which is a conveyor-belt lunch.

    When the trip has a budget, each item is also compared against what its category
    may affordably claim. Anything above that is flagged ``over_budget`` and pushed
    after the affordable options, but it is never silently dropped: the user decides,
    and the flag tells them why it does not fit. The one exception is the deliberate
    "worth paying more" tier, which the budget itself asked for.
    """
    days = max(1, int(trip_days or 1))
    nights = max(1, days - 1)
    people = max(1, int((traveler_profile or {}).get("total_people") or 1))
    allowances = budget_allowances(trip_budget, days, traveler_profile)
    plans: dict[str, meal_prices.TierPlan | None] = {}
    for category in ("restaurant", "hotel"):
        allowance = allowances.get(category)
        # The floor matters: a budget too small to eat at all must not turn the cheap tier into
        # "a price nobody in this city charges".
        floor = meal_prices.realistic_floor(destination, category) or 0
        plans[category] = meal_prices.tier_plan(allowance, floor=floor) if allowance else None
    for item in items:
        category = normalize_category(item.get("type")) or "attraction"
        unit = _clean_price_unit(item.get("price_unit"), category)
        # The place's own destination when it has one (recommendations record the destination they
        # were found for), otherwise the trip's.
        place_destination = str(item.get("address") or destination or "")
        # Re-bounded here as well: this is the last gate before a number reaches the
        # user's budget, so an implausible price can never be charged.
        if item.get("is_free"):
            unit_price = 0
            from_default = False
            adjusted_floor = None
        else:
            unit_price = _bounded_unit_price(item.get("unit_price"), category, place_destination)
            from_default = unit_price is None
            if from_default:
                # A destination-shaped default, not a Chinese one: the flat 80 CNY was wrong by
                # a factor of three for a Tokyo sushi dinner before anyone even looked at it.
                unit_price = meal_prices.default_unit_price(
                    place_destination, category, item.get("name"), item.get("price_hint")
                ) or DEFAULT_UNIT_PRICES.get(category, 100)
            # The last gate before a number reaches the traveller's budget. A price below the
            # cheapest realistic full meal for this kind of place is raised and *labelled*, so an
            # adjusted figure can never be mistaken for a quoted one.
            unit_price, adjusted_floor = meal_prices.adjust_price(
                unit_price, place_destination, category, item.get("name"), item.get("price_hint")
            )
        amount, formula = _cost_formula(category, unit_price, unit, days, nights, people)
        item["unit_price"] = unit_price
        item["price_unit"] = unit
        # Which price tier of the budget this is: the cheap everyday option, the middle, or the
        # expensive one worth the money. The card shows it, and it is what proves the list is a
        # mix rather than one price point repeated.
        plan = plans.get(category)
        tier = plan.tier_of(unit_price) if plan is not None else (meal_prices.TIER_MID if category in plans else "")
        item["price_tier"] = tier
        item["price_tier_label"] = meal_prices.TIER_LABELS.get(tier, "")
        # How many times this place is used, and what one use costs. The card lets the
        # traveller raise the count (three nights, two meals) and recomputes the cost from
        # these two numbers, so they travel with the item.
        item["quantity"] = DEFAULT_QUANTITY.get(category, 1)
        item["quantity_unit"] = QUANTITY_UNITS.get(category)
        item["quantity_max"] = max(1, nights if category == "hotel" else days)
        item["estimated_cost"] = amount
        if item.get("is_free"):
            source = "资料显示免费"
        elif adjusted_floor is not None:
            source = (
                f"资料价低于当地同类常见人均，已按常见人均下限 {adjusted_floor} 元估计"
            )
        elif from_default:
            source = "按当地中档常见价估算"
        else:
            source = "按检索到的价格估算"
        item["price_adjusted"] = adjusted_floor is not None
        # Sent separately so the card can rebuild the formula after the count changes
        # without having to parse this string back apart.
        item["cost_source"] = source
        item["cost_explanation"] = f"{formula}（{source}）"

        limit = allowances.get(category)
        if limit is None:
            item.pop("budget_limit", None)
            item["over_budget"] = False
            item.pop("budget_note", None)
            continue
        # "Over budget" has to mean the same thing as the 超出预算 badge next to it, or one card
        # reads "affordable middle" and "over budget" at once. The ceiling is therefore the top of
        # the tier this category still calls affordable -- 1.3x the allowance -- and for food the
        # deliberate splurge is allowed on top of that, up to 2.5x, because the traveller asked for
        # one expensive-but-good meal. Hotels get no such allowance: a 3,840 CNY hotel against an
        # 8,000 CNY trip is still exactly the problem this flag was added for.
        ceiling = plan.mid_max if plan is not None else limit
        if category == "restaurant" and plan is not None:
            ceiling = plan.splurge_max
        over = unit_price > ceiling
        splurge = bool(
            category == "restaurant" and plan is not None and unit_price > plan.mid_max
        )
        item["budget_limit"] = limit
        item["over_budget"] = over
        if splurge and not over:
            item["budget_note"] = (
                f"比人均参考（{limit} 元）高一些，属于按你的预算留出的“值得多花钱”那一档"
            )
        elif over:
            item["budget_note"] = f"超出预算参考（住宿约 {limit} 元/晚 以内更合适）" if category == "hotel" else (
                f"超出预算参考（建议单笔不超过 {ceiling} 元）"
            )
        else:
            item["budget_note"] = ""

    # Affordable options first, keeping the model's own ordering within each group.
    return sorted(items, key=lambda entry: bool(entry.get("over_budget")))


def extract_recommendations(
    destination: str,
    web_results: list[dict[str, Any]] | None,
    limit_per_category: int | None = None,
    allowances: dict[str, int] | None = None,
    limits: dict[str, int] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    """Return named recommendation items plus a status for the agent event log.

    Status is ``completed`` when the model produced the list, ``fallback`` when
    keyword rules were used, and ``skipped`` when there was nothing grounded to
    work with (which also avoids a pointless model call).

    `allowances` carries the trip's affordable per-unit ceilings so the model prefers
    options the traveller can actually pay for; it never removes a place by itself.
    `limits` carries how many of each category this trip needs so that no day has to
    repeat another day's places.
    """
    material = _material(web_results)
    text = _material_text(material)
    if len(text) < MIN_MATERIAL_CHARS:
        return [], "skipped"
    if limits is not None:
        caps = limits
    else:
        caps = CATEGORY_LIMITS if limit_per_category is None else None
    extracted = _extract_with_model(destination, material, allowances, limits)
    if extracted is None:
        fallback = _fallback_items(material, destination, text)
        return _sort_by_category(_dedupe_and_cap(fallback, limit_per_category, caps)), "fallback"
    items = [normalized for item in extracted if (normalized := _normalize_item(item, text, destination))]
    return _sort_by_category(_dedupe_and_cap(items, limit_per_category, caps)), "completed"
