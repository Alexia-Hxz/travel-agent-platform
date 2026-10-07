"""The one type the price bands are built from.

A separate module on purpose: ``meal_price_data`` is a *generated* table of measurements and
``meal_prices`` holds the policy that reads it, so the shared type lives below both. Keeping it
here is what lets the generated module be rebuilt without importing the module that consumes it.

The numbers in a band are CNY per person for one meal, per room for one night and per person for
one ticket -- always the local price for *that* destination, never a national average.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Band:
    """A realistic per-person spend: the cheap end, the middle, and the posh end."""

    low: int
    typical: int
    high: int

    def clamp_floor(self, price: int) -> int:
        """Raise a price that cannot buy a full meal of this kind, and leave the rest alone."""
        return max(int(price), self.low)


def _band(low: int, typical: int, high: int) -> Band:
    """Terse constructor, so a generated table stays readable one row per line."""
    return Band(low=low, typical=typical, high=high)
