"""Scenario-type helpers shared by evaluation and validation code."""

from __future__ import annotations

from typing import Any, Iterable, Set


SCENARIO_TYPE_ALIASES = {
    "na_cross_system": "cross_system",
}


def canonical_scenario_type(value: Any) -> str:
    """Return the canonical scenario type used for grouping and metrics."""
    key = str(value or "unknown").strip().lower()
    return SCENARIO_TYPE_ALIASES.get(key, key)


def scenario_type_matches(value: Any, expected: Any) -> bool:
    """Return True when two scenario-type labels are equivalent."""
    return canonical_scenario_type(value) == canonical_scenario_type(expected)


def expand_scenario_types(types: Iterable[Any] | None) -> Set[str]:
    """Expand requested scenario types with known aliases.

    This lets old configs that request ``cross_system`` also include the
    SFT-side label ``na_cross_system`` while still keeping per-type metrics
    canonical.
    """
    if not types:
        return set()
    requested = {str(t).strip().lower() for t in types if str(t).strip()}
    expanded = set(requested)
    canonical_requested = {canonical_scenario_type(t) for t in requested}
    for alias, canonical in SCENARIO_TYPE_ALIASES.items():
        if alias in requested or canonical in requested or canonical in canonical_requested:
            expanded.add(alias)
            expanded.add(canonical)
    return expanded
