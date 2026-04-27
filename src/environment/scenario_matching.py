"""Scenario matching helpers for SFT/RL evaluation.

Generated SFT sample ids append a scenario-type suffix such as
``_a_low_confidence_107`` to the original scenario id.  The original id may
itself carry a semantic prefix such as ``a_lc_`` or ``nf_``.  Matching must
therefore remove the sample suffix before considering lower-priority legacy
prefix stripping; otherwise low-confidence and no-fault prompts can be mapped
to ordinary fault scenarios.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCENARIO_TYPE_SUFFIX_RE = re.compile(
    r"_(na_single_system|a_single_system|na_cross_system|a_cross_system"
    r"|na_no_fault|no_fault|na_low_confidence|a_low_confidence)_\d+$"
)

DERIVED_PREFIXES = ("a_lc_", "na_lc_", "a_", "nf_")


def scenario_id_candidates(scenario_id: str) -> List[str]:
    """Return ordered id candidates, preserving semantic prefixes first."""
    sid = str(scenario_id or "")
    candidates: List[str] = []

    def add(value: str) -> None:
        if value and value not in candidates:
            candidates.append(value)

    add(sid)
    without_suffix = SCENARIO_TYPE_SUFFIX_RE.sub("", sid)
    add(without_suffix)

    for prefix in DERIVED_PREFIXES:
        if without_suffix.startswith(prefix):
            add(without_suffix[len(prefix):])
            break

    return candidates


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _get(obj: Any, key: str, default: Any = "") -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _scenario_type_rank(candidate: Any, expected_type: str) -> int:
    expected = _norm(expected_type)
    actual = _norm(_get(candidate, "scenario_type", ""))
    if not expected:
        return 1
    if actual == expected:
        return 0
    if "low_confidence" in expected and "low_confidence" in actual:
        return 1
    if "no_fault" in expected and "no_fault" in actual:
        return 1
    return 5


def match_scenario(
    scenario_id: str,
    lookup: Dict[str, Any],
    ground_truth: Optional[Dict[str, Any]] = None,
    scenario_type: str = "",
    allow_ground_truth_fallback: bool = True,
) -> Tuple[Optional[Any], str, str]:
    """Match a prompt/sample id to an original scenario.

    Returns:
        ``(scenario, matched_id, strategy)``.  ``strategy`` is one of
        ``direct_or_suffix``, ``ground_truth``, or ``missing``.
    """
    for candidate_id in scenario_id_candidates(scenario_id):
        scenario = lookup.get(candidate_id)
        if scenario is not None:
            return scenario, candidate_id, "direct_or_suffix"

    if not allow_ground_truth_fallback or not ground_truth:
        return None, "", "missing"

    gt_system = _get(ground_truth, "root_cause_system", "")
    gt_node = _get(ground_truth, "root_cause_node", "")
    gt_fault = _get(ground_truth, "fault_type", "")
    is_no_fault = (
        _norm(gt_fault) in ("normal", "no_fault", "none", "")
        or _norm(gt_node) in ("none", "")
    )

    matches: List[Any] = []
    for scenario in lookup.values():
        if _get(scenario, "root_cause_system", "") != gt_system:
            continue
        if is_no_fault:
            stype = _norm(_get(scenario, "scenario_type", ""))
            sfault = _norm(_get(scenario, "fault_type", ""))
            snode = _norm(_get(scenario, "root_cause_node", ""))
            if "no_fault" in stype or sfault in ("normal", "no_fault") or snode in ("none", ""):
                matches.append(scenario)
        elif _get(scenario, "fault_type", "") == gt_fault:
            if gt_node and _get(scenario, "root_cause_node", "") not in ("", gt_node):
                continue
            matches.append(scenario)

    if not matches:
        return None, "", "missing"

    matches.sort(
        key=lambda s: (
            _scenario_type_rank(s, scenario_type),
            _get(s, "scenario_id", ""),
        )
    )
    matched = matches[0]
    return matched, _get(matched, "scenario_id", ""), "ground_truth"

