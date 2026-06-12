"""Topology-Constrained Evidential Active Diagnosis (TC-EAD).

This module treats tool outputs as imperfect industrial evidence instead of
ground truth.  It assigns reliability to every diagnostic observation, fuses
the evidence by topology-aware node/system hypotheses, and exposes compact
scores that can be used by metrics and RL rewards.

The implementation is deliberately deterministic and dependency-free so it can
run inside every evaluation and training callback.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.evaluation.fault_taxonomy import (
    fault_exact_match,
    is_no_fault_label,
    is_no_fault_node,
    node_system,
    same_fault_family,
)


ABNORMAL_STATUSES = {"fault", "warning", "abnormal", "alert"}
NORMAL_STATUSES = {"normal", "success"}
UNSUPPORTED_CALIBRATIONS = {
    "data_unavailable_for_system",
    "oracle_features_unavailable",
    "insufficient_feature_coverage",
}
WEAK_CALIBRATIONS = {"weak_fallback_model"}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _parse_json_result(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def parse_tool_results(tool_results: Optional[Iterable[Any]]) -> List[Dict[str, Any]]:
    """Parse JSON tool result strings into dictionaries."""
    parsed: List[Dict[str, Any]] = []
    for raw in tool_results or []:
        item = _parse_json_result(raw)
        if item is not None:
            parsed.append(item)
    return parsed


def flatten_diagnostic_observations(
    tool_results: Optional[Iterable[Any]],
) -> List[Dict[str, Any]]:
    """Return direct node observations plus summary entries."""
    flattened: List[Dict[str, Any]] = []
    for result in parse_tool_results(tool_results):
        status = _norm(result.get("status"))
        if result.get("node_id") or status in ABNORMAL_STATUSES | NORMAL_STATUSES | {"unknown"}:
            flattened.append(result)
        for key in ("node_statuses", "candidate_nodes"):
            for node_status in result.get(key, []) or []:
                if isinstance(node_status, dict):
                    flattened.append(node_status)
    return flattened


@dataclass
class EvidenceObservation:
    node_id: str
    system_id: str
    status: str
    fault_type: str
    confidence: float
    reliability: float
    polarity: float
    model_source: str
    calibration: str

    @property
    def is_abnormal(self) -> bool:
        return self.status in ABNORMAL_STATUSES

    @property
    def is_normal(self) -> bool:
        return self.status in NORMAL_STATUSES


def _confidence_factor(confidence: Any) -> float:
    try:
        c = float(confidence)
    except (TypeError, ValueError):
        c = 0.5
    return _clip(0.35 + 0.65 * _clip(c))


def observation_reliability(result: Dict[str, Any]) -> float:
    """Estimate how much a tool observation should be trusted.

    Reliability is not the same as model confidence.  It also reflects whether
    the feature vector existed, which model family produced the result, and
    whether calibration guards marked the output as weak or unavailable.
    """
    status = _norm(result.get("status"))
    calibration = _norm(result.get("calibration"))
    if status in {"error", "unknown"}:
        return 0.0
    if calibration in UNSUPPORTED_CALIBRATIONS:
        return 0.0

    source = _norm(result.get("model_source"))
    if source == "system_oracle":
        base = 0.90
    elif source == "per_node_responsibility":
        base = 0.72
    elif source == "per_node":
        base = 0.28
    elif result.get("node_id"):
        base = 0.40
    else:
        base = 0.15

    if calibration in WEAK_CALIBRATIONS:
        base = min(base, 0.25)
    if calibration == "clean_baseline_false_positive_guard":
        base = 0.95 if status in NORMAL_STATUSES else 0.05

    coverage = result.get("feature_coverage")
    if coverage is not None:
        try:
            base *= _clip(float(coverage) / 0.75)
        except (TypeError, ValueError):
            pass

    return round(_clip(base * _confidence_factor(result.get("confidence"))), 4)


def extract_evidence(tool_results: Optional[Iterable[Any]]) -> List[EvidenceObservation]:
    """Convert raw tool results into reliability-weighted evidence objects."""
    observations: List[EvidenceObservation] = []
    for result in flatten_diagnostic_observations(tool_results):
        node_id = str(result.get("node_id") or "").strip()
        system_id = str(result.get("system_id") or node_system(node_id) or "").strip()
        status = _norm(result.get("status"))
        if not node_id and status not in ABNORMAL_STATUSES | NORMAL_STATUSES:
            continue
        reliability = observation_reliability(result)
        if reliability <= 0.0 and status != "unknown":
            continue

        if status in ABNORMAL_STATUSES:
            severity = {"fault": 1.0, "alert": 1.0, "warning": 0.72, "abnormal": 0.55}.get(status, 0.5)
            polarity = reliability * severity
        elif status in NORMAL_STATUSES:
            polarity = -0.55 * reliability
        else:
            polarity = 0.0

        observations.append(
            EvidenceObservation(
                node_id=node_id,
                system_id=system_id,
                status=status,
                fault_type=str(result.get("fault_type") or "None"),
                confidence=_clip(result.get("confidence") or 0.0),
                reliability=reliability,
                polarity=round(polarity, 4),
                model_source=_norm(result.get("model_source")),
                calibration=_norm(result.get("calibration")),
            )
        )
    return observations


def _softmax(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    max_score = max(scores.values())
    exps = {k: math.exp(v - max_score) for k, v in scores.items()}
    total = sum(exps.values())
    if total <= 0:
        return {k: 0.0 for k in scores}
    return {k: v / total for k, v in exps.items()}


def fuse_evidence(tool_results: Optional[Iterable[Any]]) -> Dict[str, Any]:
    """Fuse all diagnostic evidence into node/system belief distributions."""
    observations = extract_evidence(tool_results)
    node_scores: Dict[str, float] = {}
    system_scores: Dict[str, float] = {}
    fault_votes: Dict[Tuple[str, str], float] = {}
    positive_by_node: Dict[str, float] = {}
    negative_by_node: Dict[str, float] = {}

    for obs in observations:
        if obs.node_id:
            node_scores[obs.node_id] = node_scores.get(obs.node_id, 0.0) + obs.polarity
            if obs.polarity > 0:
                positive_by_node[obs.node_id] = positive_by_node.get(obs.node_id, 0.0) + obs.polarity
            elif obs.polarity < 0:
                negative_by_node[obs.node_id] = negative_by_node.get(obs.node_id, 0.0) + abs(obs.polarity)
        if obs.system_id:
            system_scores[obs.system_id] = system_scores.get(obs.system_id, 0.0) + 0.75 * obs.polarity
        if obs.is_abnormal and obs.node_id:
            fault_key = (obs.node_id, obs.fault_type)
            fault_votes[fault_key] = fault_votes.get(fault_key, 0.0) + obs.polarity

    conflict_nodes = []
    for node_id, pos in positive_by_node.items():
        neg = negative_by_node.get(node_id, 0.0)
        if pos > 0 and neg > 0:
            conflict_nodes.append(min(pos, neg) / max(pos, neg))

    node_belief = _softmax({k: v for k, v in node_scores.items() if v > 0})
    system_belief = _softmax({k: v for k, v in system_scores.items() if v > 0})
    ranked_nodes = sorted(node_belief.items(), key=lambda item: item[1], reverse=True)
    ranked_systems = sorted(system_belief.items(), key=lambda item: item[1], reverse=True)

    return {
        "observations": observations,
        "node_scores": node_scores,
        "system_scores": system_scores,
        "node_belief": node_belief,
        "system_belief": system_belief,
        "ranked_nodes": ranked_nodes,
        "ranked_systems": ranked_systems,
        "fault_votes": fault_votes,
        "conflict_rate": round(sum(conflict_nodes) / max(len(conflict_nodes), 1), 4),
        "positive_evidence_mass": round(sum(max(0.0, o.polarity) for o in observations), 4),
        "negative_evidence_mass": round(sum(abs(min(0.0, o.polarity)) for o in observations), 4),
    }


def diagnosis_evidence_support(
    final_diagnosis: Optional[Dict[str, Any]],
    tool_results: Optional[Iterable[Any]],
) -> float:
    """Score whether the final diagnosis is supported by fused evidence."""
    if not final_diagnosis:
        return 0.0

    fused = fuse_evidence(tool_results)
    observations: List[EvidenceObservation] = fused["observations"]
    diag_node = str(final_diagnosis.get("root_cause_node") or "").strip()
    diag_fault = str(final_diagnosis.get("fault_type") or "")
    diag_status_text = _norm(final_diagnosis.get("status")) + " " + _norm(diag_fault)
    says_no_fault = (
        is_no_fault_node(diag_node)
        or is_no_fault_label(diag_fault)
        or any(token in diag_status_text for token in ("normal", "no fault", "no_fault"))
    )

    abnormal = [obs for obs in observations if obs.is_abnormal]
    if says_no_fault:
        if abnormal:
            return 0.0
        normal_mass = fused["negative_evidence_mass"]
        return round(_clip(0.45 + 0.45 * min(1.0, normal_mass)), 4) if observations else 0.3

    if not diag_node:
        return 0.0

    node_score = max(0.0, fused["node_scores"].get(diag_node, 0.0))
    same_node_abnormal = [obs for obs in abnormal if obs.node_id == diag_node]
    if not same_node_abnormal:
        return 0.0

    fault_bonus = 0.0
    for obs in same_node_abnormal:
        if fault_exact_match(obs.fault_type, diag_fault):
            fault_bonus = max(fault_bonus, 0.35)
        elif same_fault_family(obs.fault_type, diag_fault):
            fault_bonus = max(fault_bonus, 0.25)
        elif _norm(obs.fault_type) in ("", "none", "uncertain"):
            fault_bonus = max(fault_bonus, 0.10)

    belief = fused["node_belief"].get(diag_node, 0.0)
    support = 0.25 + 0.35 * min(1.0, node_score) + 0.25 * belief + fault_bonus
    support -= 0.25 * fused["conflict_rate"]
    return round(_clip(support), 4)


def ground_truth_belief_support(
    ground_truth: Dict[str, Any],
    tool_results: Optional[Iterable[Any]],
) -> float:
    """Return fused node belief assigned to the ground-truth root node."""
    gt_node = str(ground_truth.get("root_cause_node") or "").strip()
    gt_fault = str(ground_truth.get("fault_type") or "")
    if is_no_fault_node(gt_node) or is_no_fault_label(gt_fault):
        return diagnosis_evidence_support(
            {"status": "Normal", "root_cause_node": "none", "fault_type": "Normal"},
            tool_results,
        )
    fused = fuse_evidence(tool_results)
    return round(float(fused["node_belief"].get(gt_node, 0.0)), 4)


def evidence_conflict_rate(tool_results: Optional[Iterable[Any]]) -> float:
    return float(fuse_evidence(tool_results)["conflict_rate"])


def suggested_next_actions(
    tool_results: Optional[Iterable[Any]],
    candidate_nodes: Iterable[str],
    max_actions: int = 5,
) -> List[Dict[str, Any]]:
    """Rank candidate diagnose_node actions by uncertainty reduction proxy.

    This is a deterministic active-diagnosis primitive used for analysis and
    future policy guidance.  It favors nodes with some upstream/system support
    but insufficient direct evidence.
    """
    fused = fuse_evidence(tool_results)
    scored = []
    known_nodes = set(fused["node_scores"].keys())
    for node_id in candidate_nodes:
        system_id = node_system(node_id)
        system_prior = fused["system_belief"].get(system_id, 0.0)
        already_checked_penalty = 0.35 if node_id in known_nodes else 0.0
        score = 0.50 * system_prior + 0.35 * (1.0 - already_checked_penalty)
        scored.append({
            "action": "diagnose_node",
            "node_id": node_id,
            "expected_information_gain": round(_clip(score), 4),
        })
    scored.sort(key=lambda item: item["expected_information_gain"], reverse=True)
    return scored[:max_actions]
