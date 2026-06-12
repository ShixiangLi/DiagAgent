"""Fault taxonomy helpers for strict and family-level diagnosis scoring."""

from __future__ import annotations

import re
from typing import Any


def normalize_fault_label(value: Any) -> str:
    """Normalize labels from GT, Oracle outputs, and model diagnoses."""
    text = str(value or "").strip().lower()
    # Preserve signed fault intensities.  A plain separator hyphen may be
    # normalized away, but labels such as ``oa_bias_-4`` and ``oa_bias_4`` are
    # physically different and must not collapse to the same string.
    text = re.sub(r"(^|[_\s])\+(?=\d)", r"\1pos_", text)
    text = re.sub(r"(^|[_\s])-(?=\d)", r"\1neg_", text)
    text = text.replace(" ", "_").replace("-", "_")
    text = re.sub(r"[^a-z0-9_]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def _canonical_fault_label(value: Any) -> str:
    """Canonical label for strict exact matching.

    This intentionally keeps signs, while removing non-semantic suffixes used
    by some filenames/models.
    """
    text = normalize_fault_label(value)
    for suffix in ("_default",):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text.strip("_")


def normalize_node_id(value: Any) -> str:
    """Normalize a topology node id for robust exact/sub-string matching."""
    return str(value or "").strip().lower()


def node_system(node_id: Any) -> str:
    """Return the system prefix for a node id or system node."""
    node = normalize_node_id(node_id)
    if node.startswith("system::"):
        return node.split("::", 1)[1]
    if "::" in node:
        return node.split("::", 1)[0]
    return ""


def is_no_fault_label(value: Any) -> bool:
    return normalize_fault_label(value) in {
        "",
        "normal",
        "none",
        "no_fault",
        "no_fault_detected",
    }


def is_no_fault_node(value: Any) -> bool:
    return normalize_node_id(value) in {"", "none", "normal", "no_fault", "no fault"}


def non_empty_substring_match(expected: Any, actual: Any) -> bool:
    expected_norm = normalize_node_id(expected)
    actual_norm = normalize_node_id(actual)
    return bool(
        expected_norm
        and actual_norm
        and (expected_norm in actual_norm or actual_norm in expected_norm)
    )


def fault_exact_match(expected: Any, actual: Any) -> bool:
    expected_norm = _canonical_fault_label(expected)
    actual_norm = _canonical_fault_label(actual)
    return bool(expected_norm and actual_norm and expected_norm == actual_norm)


def fault_family(label: Any) -> str:
    """Map detailed LBNL labels to coarse physical fault families.

    These families are intentionally conservative: they group labels that are
    physically close or commonly confused by the system Oracle, while keeping
    unrelated systems/fault mechanisms separate.
    """
    s = normalize_fault_label(label)
    if is_no_fault_label(s):
        return "normal"

    groups = [
        ("chiller_bypass", ("bypass_leakage", "bypass_stuck", "bypass")),
        ("chiller_temperature_sensor_bias", ("chiller_bias",)),
        ("cooling_tower_sensor_bias", ("coolingtower_bias", "cooling_tower_bias")),
        ("cooling_tower_fouling", ("coolingtower_fouling", "cooling_tower_fouling")),
        ("chilled_water_pressure", ("secondary_chilled_water_pressure",)),
        ("boiler_sensor_bias", ("boiler_bias", "hot_water_temp_bias")),
        ("boiler_pressure", ("hot_water_pressure",)),
        ("boiler_fouling", ("boiler_foul", "boiler_fouling")),
        ("rtu_refrigerant_circuit", (
            "overcharge", "undercharge", "evapfouling", "condfouling",
            "evaporator_fouling", "condenser_fouling",
        )),
        ("rtu_pipe_sensor", ("liquidpipe", "suctionpipe")),
        ("fcu_airside", (
            "oadmpr", "filterrestriction", "fanoutletblockage",
            "control_heatingreverse", "oablockage",
        )),
        ("ddahu_hot_deck", ("sensorbias_hsa", "sensorbias_hsp", "dmprstuck_hot")),
        ("ddahu_cold_deck", ("sensorbias_csa", "sensorbias_csp", "dmprstuck_cold")),
        ("ddahu_outdoor_air_damper", ("dmprstuck_oa",)),
        ("water_valve_cooling", ("vlvstuck_cooling", "vlvleak_cooling")),
        ("water_valve_heating", ("vlvstuck_heating", "vlvleak_heating")),
        ("coil_fouling_cooling", ("fouling_cooling",)),
        ("coil_fouling_heating", ("fouling_heating",)),
        ("ahu_supply_air_sensor", ("sa_bias", "sensorbias_sa")),
        ("ahu_outdoor_air_sensor", ("oa_bias", "sensorbias_oa")),
        ("ahu_cooling_coil_valve", ("coi_leakage", "coi_stuck")),
        ("ahu_damper", ("damper_stuck", "dmpr")),
        ("fan", ("fan",)),
        ("pump", ("pump",)),
    ]

    for family, tokens in groups:
        if any(token in s for token in tokens):
            return family

    # Drop trailing intensities and keep a stable mechanism prefix.
    parts = [
        p for p in s.split("_")
        if p not in {"default", "minor", "moderate", "severe"}
        and not re.fullmatch(r"[+-]?\d+(?:\.\d+)?%?c?", p)
    ]
    return "_".join(parts[:2]) if len(parts) >= 2 else s


def same_fault_family(expected: Any, actual: Any) -> bool:
    expected_family = fault_family(expected)
    actual_family = fault_family(actual)
    return bool(
        expected_family
        and actual_family
        and expected_family != "normal"
        and expected_family == actual_family
    )
