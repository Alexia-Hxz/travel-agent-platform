"""Turn a foreign price into CNY, without calling any exchange-rate service.

Why this exists
---------------
Everything downstream of the recommendation step is denominated in CNY: ``Budget.currency``
is ``"CNY"``, the per-category sanity bounds are stated in CNY, and the budget sheet prints
"元". But the *material* the model reads for a Japanese trip is written in yen, and the
prompt told it that ``unit_price`` "只能取自资料中出现的价格" -- so it faithfully copied
630 (yen) into a field the rest of the system reads as 630 CNY. Measured on a real Tokyo
plan: a museum ticket quoted as "630日元（约32元人民币）" was stored as **630**, and the
cheapest restaurants were dropped to the 80 CNY Chinese default because their real yen
prices fell outside the CNY bounds.

Two halves, deliberately
-----------------------
1. ``prompt_hint()`` tells the model the destination's currency and the rate, and that a
   foreign price must be converted. Naming the currency is what makes this reliable: the
   hotel prompt already said "人民币元/晚" and the model converted correctly every time.
2. ``convert_foreign_number()`` is the safety net for when it copies a foreign number
   anyway. It converts **only when the quoted text carries a currency marker** (or an
   explicit RMB figure), so a plain number is never converted twice.

The rates are a static table: no key, no network, no failure mode. They are estimates for
budgeting, not for accounting. Update ``RATES_TO_CNY`` when they drift; the value is
recorded in ``RATES_AS_OF`` so the interface can say how old it is.
"""

from __future__ import annotations

import re

# Static estimates, kept as a table on purpose: an exchange-rate API would add a key, a
# network dependency and a new way for planning to fail, for a number that only ever feeds
# a budget estimate. 1 unit of the currency = this many CNY.
RATES_AS_OF = "2026-09"
RATES_TO_CNY: dict[str, float] = {
    "CNY": 1.0,
    "JPY": 0.048,
    "USD": 7.10,
    "EUR": 7.75,
    "GBP": 9.05,
    "KRW": 0.0051,
    "HKD": 0.91,
    "MOP": 0.88,
    "TWD": 0.22,
    "SGD": 5.30,
    "MYR": 1.60,
    "THB": 0.20,
    "VND": 0.00028,
    "IDR": 0.00045,
    "PHP": 0.125,
    "INR": 0.085,
    "AED": 1.93,
    "AUD": 4.65,
    "NZD": 4.30,
    "CAD": 5.20,
    "CHF": 8.00,
    "TRY": 0.21,
    "RUB": 0.078,
    "EGP": 0.145,
    "MXN": 0.38,
    "BRL": 1.30,
    "ZAR": 0.39,
}

CURRENCY_LABELS = {
    "JPY": "日元", "USD": "美元", "EUR": "欧元", "GBP": "英镑", "KRW": "韩元",
    "HKD": "港币", "MOP": "澳门元", "TWD": "新台币", "SGD": "新加坡元", "MYR": "马来西亚林吉特",
    "THB": "泰铢", "VND": "越南盾", "IDR": "印尼盾", "PHP": "菲律宾比索", "INR": "印度卢比",
    "AED": "迪拉姆", "AUD": "澳元", "NZD": "新西兰元", "CAD": "加元", "CHF": "瑞士法郎",
    "TRY": "里拉", "RUB": "卢布", "EGP": "埃及镑", "MXN": "墨西哥比索", "BRL": "雷亚尔",
    "ZAR": "兰特", "CNY": "人民币",
}

# Destination text -> the currency that country/region actually uses. Matched as substrings,
# so "日本东京" and "东京 5 日" both resolve, and the check is case-insensitive for the
# romanised forms.
_JPY = ("日本", "东京", "東京", "大阪", "京都", "北海道", "札幌", "冲绳", "名古屋", "福冈",
        "神户", "奈良", "横滨", "箱根", "japan", "tokyo", "osaka", "kyoto", "hokkaido",
        "okinawa", "nagoya", "sapporo", "nara")
_KRW = ("韩国", "首尔", "釜山", "济州", "korea", "seoul", "busan", "jeju")
_THB = ("泰国", "曼谷", "清迈", "普吉", "苏梅", "thailand", "bangkok", "chiang mai", "phuket")
_USD = ("美国", "纽约", "洛杉矶", "旧金山", "拉斯维加斯", "夏威夷", "西雅图", "波士顿",
        "usa", "united states", "new york", "los angeles", "san francisco", "las vegas",
        "hawaii", "seattle", "boston")
_EUR = ("法国", "巴黎", "意大利", "罗马", "米兰", "威尼斯", "佛罗伦萨", "德国", "柏林", "慕尼黑",
        "西班牙", "巴塞罗那", "马德里", "荷兰", "阿姆斯特丹", "比利时", "布鲁塞尔", "奥地利",
        "维也纳", "希腊", "雅典", "葡萄牙", "里斯本", "芬兰", "赫尔辛基", "爱尔兰", "都柏林",
        "france", "paris", "italy", "rome", "milan", "venice", "florence", "germany", "berlin",
        "munich", "spain", "barcelona", "madrid", "netherlands", "amsterdam", "belgium",
        "brussels", "austria", "vienna", "greece", "athens", "portugal", "lisbon", "finland")
_GBP = ("英国", "伦敦", "爱丁堡", "曼彻斯特", "uk", "united kingdom", "london", "edinburgh")
_HKD = ("香港", "hong kong")
_MOP = ("澳门", "macau", "macao")
_TWD = ("台湾", "台北", "高雄", "台中", "taiwan", "taipei", "kaohsiung")
_SGD = ("新加坡", "singapore")
_MYR = ("马来西亚", "吉隆坡", "沙巴", "malaysia", "kuala lumpur", "sabah")
_VND = ("越南", "河内", "胡志明", "岘港", "vietnam", "hanoi", "ho chi minh", "danang")
_IDR = ("印尼", "巴厘岛", "雅加达", "indonesia", "bali", "jakarta")
_PHP = ("菲律宾", "马尼拉", "长滩岛", "philippines", "manila", "boracay")
_AED = ("迪拜", "阿联酋", "阿布扎比", "dubai", "abu dhabi", "uae")
_AUD = ("澳大利亚", "澳洲", "悉尼", "墨尔本", "布里斯班", "australia", "sydney", "melbourne")
_NZD = ("新西兰", "奥克兰", "皇后镇", "new zealand", "auckland", "queenstown")
_CAD = ("加拿大", "温哥华", "多伦多", "蒙特利尔", "canada", "vancouver", "toronto", "montreal")
_CHF = ("瑞士", "苏黎世", "日内瓦", "因特拉肯", "switzerland", "zurich", "geneva")
_TRY = ("土耳其", "伊斯坦布尔", "卡帕多奇亚", "turkey", "istanbul", "cappadocia")
_RUB = ("俄罗斯", "莫斯科", "圣彼得堡", "russia", "moscow", "saint petersburg")
_EGP = ("埃及", "开罗", "egypt", "cairo")
_MXN = ("墨西哥", "坎昆", "mexico", "cancun")
_BRL = ("巴西", "里约", "brazil", "rio de janeiro")
_INR = ("印度", "新德里", "孟买", "india", "delhi", "mumbai")

# Home currency first: a Chinese destination must resolve to CNY, not to something a
# substring of it happens to match.
_CNY = ("中国", "国内", "北京", "上海", "南京", "杭州", "成都", "广州", "深圳", "西安", "重庆",
        "三亚", "厦门", "青岛", "长沙", "武汉", "昆明", "丽江", "桂林", "哈尔滨", "乌鲁木齐",
        "苏州", "无锡", "天津", "大连", "沈阳", "济南", "郑州", "贵阳", "拉萨", "敦煌", "西宁",
        "china", "beijing", "shanghai", "nanjing", "hangzhou", "chengdu", "guangzhou",
        "shenzhen", "xian", "chongqing", "sanya", "xiamen", "qingdao", "changsha", "wuhan",
        "kunming", "lijiang", "guilin", "harbin")

_DESTINATION_MARKERS: tuple[tuple[tuple[str, ...], str], ...] = (
    (_CNY, "CNY"),
    (_JPY, "JPY"),
    (_KRW, "KRW"),
    (_THB, "THB"),
    (_HKD, "HKD"),
    (_MOP, "MOP"),
    (_TWD, "TWD"),
    (_SGD, "SGD"),
    (_MYR, "MYR"),
    (_VND, "VND"),
    (_IDR, "IDR"),
    (_PHP, "PHP"),
    (_AED, "AED"),
    (_AUD, "AUD"),
    (_NZD, "NZD"),
    (_CAD, "CAD"),
    (_CHF, "CHF"),
    (_TRY, "TRY"),
    (_RUB, "RUB"),
    (_EGP, "EGP"),
    (_MXN, "MXN"),
    (_BRL, "BRL"),
    (_INR, "INR"),
    (_EUR, "EUR"),
    (_GBP, "GBP"),
    (_USD, "USD"),
)

# Currency names, longest first so "新加坡元" is read before "元" and "日元" before "元".
_CURRENCY_WORDS: tuple[tuple[str, str], ...] = (
    ("人民币", "CNY"), ("新加坡元", "SGD"), ("马来西亚林吉特", "MYR"), ("新西兰元", "NZD"),
    ("瑞士法郎", "CHF"), ("墨西哥比索", "MXN"), ("菲律宾比索", "PHP"), ("埃及镑", "EGP"),
    ("新台币", "TWD"), ("印度卢比", "INR"), ("印尼盾", "IDR"), ("越南盾", "VND"),
    ("日元", "JPY"), ("日圆", "JPY"), ("円", "JPY"), ("美元", "USD"), ("美金", "USD"),
    ("欧元", "EUR"), ("英镑", "GBP"), ("韩元", "KRW"), ("港币", "HKD"),
    ("港元", "HKD"), ("澳门元", "MOP"), ("新币", "SGD"), ("马币", "MYR"), ("泰铢", "THB"),
    ("迪拉姆", "AED"), ("澳元", "AUD"), ("澳币", "AUD"), ("加元", "CAD"), ("加币", "CAD"),
    ("瑞郎", "CHF"), ("里拉", "TRY"), ("卢布", "RUB"), ("兰特", "ZAR"), ("比索", "PHP"),
    ("雷亚尔", "BRL"), ("卢比", "INR"), ("元", "CNY"), ("块钱", "CNY"), ("块", "CNY"),
)

# ISO codes as they appear in price text, e.g. "JPY 1,200" or "4800 JPY".
_CODE_PATTERN = re.compile(
    r"\b(CNY|RMB|JPY|USD|EUR|GBP|KRW|HKD|MOP|TWD|SGD|MYR|THB|VND|IDR|PHP|INR|AED|AUD|NZD|CAD|CHF|TRY|RUB|EGP|MXN|BRL|ZAR)\b",
    re.IGNORECASE,
)
# A CNY figure written out: "约32元", "32元人民币", "约 2150 元/晚". The lookbehind keeps
# "630日元" and "1500美元" out: there the character before 元 is the currency, not a digit.
_CNY_AMOUNT_PATTERN = re.compile(
    r"(\d[\d,]*(?:\.\d+)?)\s*(?<![日韩美欧港台加澳新印菲墨巴埃])\s*元(?![\d])"
)
_ANY_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

# Symbols need the destination (or a neighbouring word) to disambiguate: "¥" is written for
# both the yen and the renminbi, and "$" for a dozen dollars.
_SYMBOLS: tuple[tuple[str, str], ...] = (
    ("€", "EUR"), ("£", "GBP"), ("₩", "KRW"), ("฿", "THB"), ("₫", "VND"), ("₹", "INR"),
    ("₽", "RUB"), ("₺", "TRY"), ("₪", "ILS"), ("₴", "UAH"),
)
_DOLLAR_HINTS: tuple[tuple[str, str], ...] = (
    ("US$", "USD"), ("USD", "USD"), ("HK$", "HKD"), ("港币", "HKD"), ("NT$", "TWD"),
    ("新台币", "TWD"), ("S$", "SGD"), ("新加坡", "SGD"), ("A$", "AUD"), ("澳元", "AUD"),
    ("C$", "CAD"), ("加元", "CAD"), ("NZ$", "NZD"), ("R$", "BRL"),
)
_DOLLAR_CURRENCIES = {"USD", "HKD", "TWD", "SGD", "AUD", "CAD", "NZD", "MXN", "BRL"}


def currency_for_destination(destination: str | None) -> str | None:
    """The currency the destination actually uses, or None when it is not recognised."""
    text = str(destination or "").strip().casefold()
    if not text:
        return None
    for markers, currency in _DESTINATION_MARKERS:
        if any(marker in text for marker in markers):
            return currency
    return None


def is_foreign(destination: str | None) -> bool:
    """True when the destination is outside mainland China and its currency is known."""
    currency = currency_for_destination(destination)
    return currency is not None and currency != "CNY"


def rate_to_cny(currency: str) -> float:
    return RATES_TO_CNY.get(str(currency or "").upper(), 1.0)


def to_cny(amount: float, currency: str) -> int:
    """Convert an amount into whole CNY, never returning 0 for a paid item."""
    converted = float(amount) * rate_to_cny(currency)
    rounded = int(round(converted))
    if rounded <= 0 < amount:
        return 1
    return rounded


def _number_of(text: str) -> float | None:
    """The first plausible number in a fragment, thousands separators removed."""
    match = _ANY_NUMBER.search(str(text or "").replace("，", ","))
    if match is None:
        return None
    try:
        return float(match.group().replace(",", ""))
    except ValueError:
        return None


def explicit_cny_amount(text: str | None) -> int | None:
    """A CNY figure the material states outright, e.g. "630日元（约32元人民币）" -> 32.

    This is the best possible answer: the source already did the conversion, so nothing is
    guessed. Read the last match, which is the phrasing "（约32元人民币）" uses.
    """
    matches = _CNY_AMOUNT_PATTERN.findall(str(text or ""))
    if not matches:
        return None
    value = _number_of(matches[-1])
    if value is None:
        return None
    amount = int(round(value))
    return amount if amount > 0 else None


def _mentioned_currencies(text: str, destination: str | None) -> list[str]:
    """Every currency the text mentions, in the order it mentions them.

    The scan goes longest word first and **blanks out** what it matched, so the bare "元"
    inside "日元" or "韩元" is never counted as a second, separate currency. Symbols are
    resolved with the destination, since "¥" is written for both the yen and the renminbi.
    """
    remaining = str(text or "")
    found: list[tuple[int, str]] = []

    def _record(word: str, currency: str) -> None:
        nonlocal remaining
        index = remaining.find(word)
        while index != -1:
            found.append((index, currency))
            # Blank the match so a shorter currency word inside it cannot match again.
            remaining = remaining[:index] + "\u0000" * len(word) + remaining[index + len(word):]
            index = remaining.find(word)

    for word, currency in sorted(_CURRENCY_WORDS, key=lambda pair: -len(pair[0])):
        if word in remaining:
            _record(word, currency)
    code = _CODE_PATTERN.search(remaining)
    if code is not None:
        found.append((code.start(), "CNY" if code.group(1).upper() == "RMB" else code.group(1).upper()))
    for symbol, currency in _SYMBOLS:
        if symbol in remaining:
            found.append((remaining.find(symbol), currency))
    if "$" in remaining or "＄" in remaining:
        resolved = next((currency for marker, currency in _DOLLAR_HINTS if marker in remaining), None)
        if resolved is None:
            destination_currency = currency_for_destination(destination)
            resolved = destination_currency if destination_currency in _DOLLAR_CURRENCIES else "USD"
        found.append((remaining.find("$") if "$" in remaining else remaining.find("＄"), resolved))
    if "¥" in remaining or "￥" in remaining:
        destination_currency = currency_for_destination(destination)
        resolved = destination_currency if destination_currency in {"JPY", "CNY"} else "CNY"
        index = remaining.find("¥") if "¥" in remaining else remaining.find("￥")
        found.append((index, resolved))
    found.sort(key=lambda pair: pair[0])
    ordered: list[str] = []
    for _, currency in found:
        if currency not in ordered:
            ordered.append(currency)
    return ordered


def hint_currency(text: str | None, destination: str | None = None) -> str | None:
    """Which currency a piece of price text is written in, or None when it is unclear.

    When a hint quotes both currencies -- "630日元（约32元人民币）" is the common shape in
    Chinese travel writing -- the destination's own currency wins: the local price is the
    one being quoted, and the renminbi figure beside it is the source's own conversion.
    Otherwise the first currency mentioned wins, with CNY preferred over a stray symbol.
    """
    mentioned = _mentioned_currencies(text, destination)
    if not mentioned:
        return None
    destination_currency = currency_for_destination(destination)
    if destination_currency is not None and destination_currency in mentioned:
        return destination_currency
    if "CNY" in mentioned:
        return "CNY"
    return mentioned[0]


def convert_foreign_number(
    amount: int, price_hint: str | None, destination: str | None
) -> tuple[int, str] | None:
    """Convert a price the model copied out of foreign material into CNY.

    Returns ``(cny_amount, explanation)`` or None when nothing should be converted. Two
    guards keep this from over-reaching:

    * a hint that quotes the renminbi figure the source already worked out is used as-is
      ("630日元（约32元人民币）" -> 32), because the source's own conversion beats ours;
    * a price with no currency wording at all is only converted when the destination's
      currency is the sole sensible reading **and** the number is too large to be CNY --
      so a value that is already in CNY is never converted twice.
    """
    if amount <= 0:
        return None
    stated = hint_currency(price_hint, destination)
    destination_currency = currency_for_destination(destination)
    if stated not in (None, "CNY"):
        # The source may also print its own CNY conversion; that one is authoritative.
        explicit = explicit_cny_amount(price_hint)
        if explicit is not None:
            return explicit, f"资料已换算（原文为{stated}）"
        return to_cny(amount, stated), f"按{stated}换算"
    if stated == "CNY":
        return None
    if destination_currency in (None, "CNY"):
        return None
    if amount < 300:
        # Small enough to be a plausible CNY price already (a 100 CNY ticket, a 60 CNY
        # meal): converting would be the bigger error.
        return None
    return to_cny(amount, destination_currency), f"按{destination_currency}估算"


# A number written next to the currency it is in, either way round: "2,000日元" and "¥150"
# are the same statement. The suffix alternation is ordered longest first so the bare "元" of
# "日元" is never read as the renminbi, and the prefix list is ordered so "HK$88" is Hong Kong
# dollars rather than a bare dollar sign. Kept as two patterns because "¥598元" says both, and
# the written-out "元" is the more specific of the two.
_SUFFIX_AMOUNT = re.compile(
    r"(?P<number>\d[\d,]*(?:\.\d+)?)\s*"
    r"(?P<marker>日元|日圆|韩元|泰铢|欧元|美元|港元|港币|新台币|英镑|新加坡元|人民币|元|¥|￥|\$|€|£|฿)"
)
_PREFIX_AMOUNT = re.compile(
    r"(?P<marker>HK\$|NT\$|US\$|RMB|人民币|¥|￥|\$|€|£|฿)\s*(?P<number>\d[\d,]*(?:\.\d+)?)"
)
_MARKED_CURRENCIES = {
    "日元": "JPY",
    "日圆": "JPY",
    "韩元": "KRW",
    "泰铢": "THB",
    "欧元": "EUR",
    "美元": "USD",
    "港元": "HKD",
    "港币": "HKD",
    "新台币": "TWD",
    "英镑": "GBP",
    "新加坡元": "SGD",
    "人民币": "CNY",
    "元": "CNY",
    "HK$": "HKD",
    "NT$": "TWD",
    "US$": "USD",
    "RMB": "CNY",
    "€": "EUR",
    "£": "GBP",
    "฿": "THB",
}


def marked_amount(text: str | None, destination: str | None = None) -> tuple[int, str] | None:
    """The first amount the text marks with a currency: ``(amount, currency)`` or None.

    This is what lets a price be read *out of the source's own price line* instead of out of
    the model's reply. Measured on a real Tokyo plan: for "常设展成人620日元" the model
    answered ``1`` (a ticket count, not a price), and for several museums it answered
    nothing, so the cards showed the 100 CNY category default while the material said 600
    yen. Both are read correctly from the text here.
    """
    body = str(text or "").replace("，", ",")
    suffix = _SUFFIX_AMOUNT.search(body)
    prefix = _PREFIX_AMOUNT.search(body)
    # Whichever the text writes first; on the same number the suffix wins, because "598元" is
    # a statement about the renminbi while a bare "¥598" only says "some yen-like currency".
    # Compared at the number itself: in "¥598元" the symbol starts one character earlier than
    # the figure, and it is the figure that decides which one is the same statement.
    if suffix is not None and (prefix is None or suffix.start("number") <= prefix.start("number")):
        match = suffix
    else:
        match = prefix
    if match is None:
        return None
    marker = str(match.group("marker") or "")
    value = _number_of(match.group("number"))
    if not value or value <= 0:
        return None
    currency = _MARKED_CURRENCIES.get(marker)
    if currency is None:
        # A bare symbol: "¥" is written for both the yen and the renminbi, so the
        # destination decides, exactly as it does in `hint_currency`; a bare "$" is the
        # destination's dollars when it has them, and US dollars otherwise.
        currency = hint_currency(marker, destination) or currency_for_destination(destination)
        if currency is None and marker in ("$", "￥", "¥"):
            currency = "USD"
    if currency is None:
        return None
    return int(round(value)), currency


def prompt_hint(destination: str | None) -> str:
    """A sentence for the extraction prompt: which currency, the rate, and what to do.

    Naming the currency and the rate is what makes the model convert reliably. Measured
    without it: a museum quoted at "630日元（约32元人民币）" came back as ``unit_price`` 630.
    """
    currency = currency_for_destination(destination)
    if currency is None or currency == "CNY":
        return ""
    label = CURRENCY_LABELS.get(currency, currency)
    rate = rate_to_cny(currency)
    if rate >= 1:
        rate_text = f"1 {label} ≈ {rate:.2f} 元人民币"
    else:
        rate_text = f"1 元人民币 ≈ {1 / rate:.1f} {label}（即 1 {label} ≈ {rate:.4f} 元人民币）"
    return (
        f"目的地 {destination} 使用的货币是{label}（{currency}），{rate_text}"
        f"（静态估算，{RATES_AS_OF}）。"
        f"**资料里的价格若是{label}，必须先按此汇率换算成人民币整数再填 unit_price**；"
        f"price_hint 里保留资料原文（含原币种）。"
    )
