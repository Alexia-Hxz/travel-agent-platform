"""Generate ``meal_price_data.py`` from the sourced research file.

Run from the repository root:

    backend\\.venv\\Scripts\\python.exe scripts\\build_meal_price_data.py

Input: ``docs/meal_price_research.json`` -- 23 destinations, 271 meal-kind rows, every row
carrying its own source URL (173 distinct sources). Local-currency amounts are converted with the
static rates in ``app/agents/tools/currency.py`` (the same table the rest of the budget uses), so
the generated numbers are CNY and comparable with every price the pipeline stores.

The generator keeps two things honest:

* a kind the research could not source for a destination is **omitted**, not guessed -- the
  runtime falls back to the destination's ``generic`` row, and where that is also missing there
  is the written record in this header rather than a fabricated number;
* the research's own caveats (tax/tip not normalised, thin rows) are copied into the header, so
  the next reader sees them next to the data instead of in a chat log.

The output is committed as a normal module: no network, no key, no failure mode at runtime.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))

from app.agents.tools import currency  # noqa: E402  (path set up above)

SOURCE = REPO / "docs" / "meal_price_research.json"
TARGET = REPO / "backend" / "app" / "agents" / "tools" / "meal_price_data.py"

# The research names a destination by country; the runtime keys on the country name that
# ``destination_service`` resolves, so the two spellings are joined here, once.
DESTINATION_ALIASES = {"阿联酋": "阿拉伯联合酋长国"}

# Research kind -> runtime kind. The only difference is the noodle row.
KIND_ALIASES = {"noodles": "ramen_noodles"}

# Kinds the runtime prices but the research could not source for that destination. They are kept
# from the hand-written first pass and marked as estimates in the generated comment.
KIND_ORDER = (
    "street_food", "breakfast", "fast_food", "ramen_noodles", "casual_local", "cafe_dessert",
    "sushi_kaiten", "sushi_mid", "sushi_fine", "bbq_grill", "izakaya_bar", "hotpot", "seafood",
    "fine_dining", "generic",
)


def to_cny(value: int | float, code: str) -> int:
    """Convert a local amount to whole CNY with the project's own static table."""
    rate = currency.RATES_TO_CNY.get(code)
    if rate is None:
        raise SystemExit(f"no rate for {code}")
    return max(0, int(round(float(value) * rate)))


def band(values: dict, code: str) -> tuple[int, int, int]:
    low, typical, high = (to_cny(values[key], code) for key in ("low", "typical", "high"))
    # Rounding can collapse a band; keep it strictly increasing so the runtime invariants hold.
    typical = max(typical, low + 1)
    high = max(high, typical + 1)
    return low, typical, high


def enforce_sushi_ordering(bands: dict[str, tuple[int, int, int]], note: list[str], where: str) -> None:
    """A conveyor-belt row dearer than a sit-down sushi row is a thin-data artefact.

    Measured in the research: Thailand's ``sushi_kaiten`` typical came out above its
    ``sushi_mid`` typical, which would let the planner claim a budget sushi lunch costs more
    than a sushi restaurant. Each cheaper row is capped just under the dearer one, and the
    correction is printed so it is never silent.
    """
    for cheaper, dearer in (("sushi_kaiten", "sushi_mid"), ("sushi_mid", "sushi_fine")):
        if cheaper not in bands or dearer not in bands:
            continue
        low, typical, high = bands[cheaper]
        if typical < bands[dearer][1]:
            continue
        capped_typical = max(low + 1, int(bands[dearer][1] * 0.8))
        capped_high = max(capped_typical + 1, int(bands[dearer][2] * 0.8))
        capped_low = min(low, capped_typical - 1)
        bands[cheaper] = (capped_low, capped_typical, capped_high)
        note.append(f"{where}: {cheaper} capped to {capped_low}/{capped_typical}/{capped_high}"
                    f" (was {low}/{typical}/{high}, above {dearer})")


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    destinations: dict[str, dict] = data["destinations"]
    corrections: list[str] = []

    # Convert everything once, then sanity-check the shape of the result.
    converted: dict[str, dict[str, tuple[int, int, int]]] = {}
    for key, entry in destinations.items():
        code = entry["currency"]
        table: dict[str, tuple[int, int, int]] = {}
        for kind in KIND_ORDER:
            raw = entry["meals"].get(kind)
            if raw is None:
                for alias, runtime_kind in KIND_ALIASES.items():
                    if runtime_kind == kind:
                        raw = entry["meals"].get(alias)
                        break
            if raw is not None:
                table[kind] = band(raw, code)
        enforce_sushi_ordering(table, corrections, key)
        converted[key] = table
    for line in corrections:
        print("correction:", line)

    lines: list[str] = []
    add = lines.append
    add('"""Where the price bands in ``meal_prices`` come from, per destination. GENERATED FILE.')
    add("")
    add("Do not edit by hand: run ``scripts/build_meal_price_data.py``, which reads a sourced")
    add("research file and rewrites this module. Each row below was measured from the source cited")
    add("in that file (chain price lists, tourism boards, hotel ADR series, Numbeo baselines),")
    add(f"gathered {data['as_of']}, and converted to CNY with ``currency.RATES_TO_CNY``")
    add(f"(rates as of {currency.RATES_AS_OF}) -- the same table the rest of the budget uses.")
    add("")
    add("Coverage: " + f"{len(destinations)} destinations; a kind with no source for a destination is")
    add("omitted rather than guessed, and the runtime then falls back to that destination's")
    add("``generic`` row.")
    add("")
    add("Caveats carried over from the research, because they change what a number means:")
    add("* tax and service are NOT normalised: the United States rows are pre-tip and pre-tax")
    add("  (a sit-down bill adds roughly 20% tip and 9% tax), London excludes the 12.5% service")
    add("  charge, Italy excludes the EUR 2-4 coperto; everywhere else the figures are as charged.")
    add("* a few rows are thin: Switzerland has only a handful of kinds, Malaysia's hotel band")
    add("  rests on a single aggregator, Turkey's bar/seafood rows are indicative average spends.")
    add("* 中国 is a composite of Beijing/Shanghai/Nanjing/Chengdu, and Beijing genuinely eats")
    add("  dearer than the national average; 美国 blends New York and Los Angeles.")
    add('"""')
    add("")
    add("from __future__ import annotations")
    add("")
    add("from app.agents.tools.price_bands import Band, _band")
    add("")
    add("")

    # ---- meals ---------------------------------------------------------------------
    add("# Country -> kind of place -> Band(low, typical, high) in CNY per person for one meal.")
    add("MEAL_BANDS: dict[str, dict[str, Band]] = {")
    for key in sorted(destinations, key=lambda name: -len(destinations[name]["meals"])):
        entry = destinations[key]
        code = entry["currency"]
        name = DESTINATION_ALIASES.get(key, key)
        add(f'    # {key} ({code}, x{currency.RATES_TO_CNY[code]})')
        add(f'    "{name}": {{')
        for kind, (low, typical, high) in converted[key].items():
            add(f'        "{kind}": _band({low}, {typical}, {high}),')
        add("    },")
    add("}")
    add("")
    add("")

    # ---- hotels and tickets --------------------------------------------------------
    add("# Per room per night, and per person per ticket, in CNY.")
    add("HOTEL_BANDS: dict[str, Band] = {")
    for key in destinations:
        entry = destinations[key]
        code = entry["currency"]
        low, typical, high = band(
            {"low": entry["hotel_low"], "typical": entry["hotel_typical"], "high": entry["hotel_high"]}, code
        )
        add(f'    "{DESTINATION_ALIASES.get(key, key)}": _band({low}, {typical}, {high}),')
    add("}")
    add("")
    add("ATTRACTION_BANDS: dict[str, Band] = {")
    for key in destinations:
        entry = destinations[key]
        code = entry["currency"]
        low, typical, high = band(
            {
                "low": entry["attraction_low"],
                "typical": entry["attraction_typical"],
                "high": entry["attraction_high"],
            },
            code,
        )
        add(f'    "{DESTINATION_ALIASES.get(key, key)}": _band({low}, {typical}, {high}),')
    add("}")
    add("")

    TARGET.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {TARGET} ({TARGET.stat().st_size} bytes, {len(lines)} lines)")


if __name__ == "__main__":
    main()
