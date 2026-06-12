"""
Metrics — Unified evaluation metrics for diagnostic agent assessment.

Six quantitative metrics:
  1. Diagnostic Accuracy (DA)
  2. Tool Format Validity (TFV)
  3. Diagnostic Completeness (DC)
  4. Search Efficiency (SE)
  5. Reasoning Authenticity (RA)
  6. Tool Invocation Rationality (TIR)
"""

import json
import re
from typing import Any, Dict, List, Optional, Set

from src.evaluation.fault_taxonomy import (
    fault_exact_match,
    is_no_fault_label,
    is_no_fault_node,
    node_system,
    non_empty_substring_match,
    same_fault_family,
)
from src.evaluation.evidence_fusion import (
    diagnosis_evidence_support,
    evidence_conflict_rate,
    ground_truth_belief_support,
    observation_reliability,
)
from src.evaluation.scenario_types import canonical_scenario_type
from src.utils.io_utils import setup_logger

logger = setup_logger(__name__)


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _non_empty_substring_match(expected: str, actual: str) -> bool:
    """Allow normalized substring matches only when both fields are present."""
    return non_empty_substring_match(expected, actual)


def _parse_tool_results(ep: Dict) -> List[Dict[str, Any]]:
    parsed = []
    for raw in ep.get("tool_results", []):
        try:
            result = json.loads(raw)
            if isinstance(result, dict):
                parsed.append(result)
        except (json.JSONDecodeError, TypeError):
            continue
    return parsed


def _episode_has_abnormal_tool_result(ep: Dict) -> bool:
    for result in _parse_tool_results(ep):
        if _norm(result.get("status")) in ("fault", "warning", "abnormal"):
            return True
        for node_status in result.get("node_statuses", []):
            if _norm(node_status.get("status")) in ("fault", "warning", "abnormal"):
                return True
        for node_status in result.get("candidate_nodes", []):
            if _norm(node_status.get("status")) in ("fault", "warning", "abnormal"):
                return True
    return False


def _flatten_tool_results(ep: Dict) -> List[Dict[str, Any]]:
    flattened = []
    for result in _parse_tool_results(ep):
        if result.get("node_id") or _norm(result.get("status")) in (
            "fault", "warning", "abnormal", "normal", "unknown",
        ):
            flattened.append(result)
        for node_status in result.get("node_statuses", []) or []:
            flattened.append(node_status)
        for node_status in result.get("candidate_nodes", []) or []:
            if node_status not in flattened:
                flattened.append(node_status)
    return flattened


def _is_no_fault_episode(ep: Dict) -> bool:
    gt = ep.get("ground_truth", {})
    gt_node = _norm(gt.get("root_cause_node", ""))
    gt_fault = _norm(gt.get("fault_type", ""))
    return is_no_fault_label(gt_fault) or is_no_fault_node(gt_node)


def _diagnosis_says_no_fault(diag: Optional[Dict]) -> bool:
    if diag is None:
        return False
    status_fault = f"{diag.get('status', '')} {diag.get('fault_type', '')}"
    return any(
        kw in _norm(status_fault)
        for kw in ("normal", "no fault", "no_fault", "none")
    )


def _strict_no_fault_match(ep: Dict) -> bool:
    diag = ep.get("final_diagnosis")
    if diag is None:
        return False
    diag_node = diag.get("root_cause_node", "")
    return (
        _diagnosis_says_no_fault(diag)
        and is_no_fault_node(diag_node)
        and not _episode_has_abnormal_tool_result(ep)
    )


def _diag_system(diag: Optional[Dict]) -> str:
    if not diag:
        return ""
    root_node = diag.get("root_cause_node", "")
    system = node_system(root_node)
    if system:
        return system
    affected = diag.get("affected_systems")
    if isinstance(affected, list) and affected:
        return _norm(affected[0])
    systems_checked = diag.get("systems_checked")
    if isinstance(systems_checked, list) and systems_checked:
        return _norm(systems_checked[0])
    return ""


def _extract_tool_calls(outputs: List[str]) -> List[Dict[str, Any]]:
    calls = []
    for text in outputs:
        matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
        for tc_str in matches:
            try:
                tc = json.loads(tc_str)
                if isinstance(tc, dict):
                    calls.append(tc)
            except json.JSONDecodeError:
                calls.append({"_invalid": tc_str})
    return calls


def _executed_tool_calls(ep: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return tool calls that were actually executed by the evaluator.

    Agent outputs may contain blocked, repeated, or invalid tool calls that are
    useful for format diagnostics but should not count as visited topology,
    sensor verification, cross-system tracing, or search steps.
    """
    trace = ep.get("turn_trace") or []
    if trace:
        calls: List[Dict[str, Any]] = []
        for turn in trace:
            if turn.get("event") not in {"tool_call", "tool_error"}:
                continue
            call = turn.get("tool_call")
            if (
                isinstance(call, dict)
                and call.get("name")
                and "parse_error" not in call
            ):
                calls.append(call)
        return calls
    return _extract_tool_calls(ep.get("agent_outputs", []))


def _visited_nodes_and_edges(ep: Dict) -> tuple[Set[str], Set[tuple]]:
    """Recover visible topology objects from executed tool calls/results."""
    calls = _executed_tool_calls(ep)
    results = _parse_tool_results(ep)
    nodes: Set[str] = set()
    edges: Set[tuple] = set()
    for idx, call in enumerate(calls):
        name = call.get("name", "")
        node_arg = _tool_arg_node(call)
        if node_arg:
            nodes.add(node_arg)
        result = results[idx] if idx < len(results) else {}
        if name == "get_node_children":
            parent = _norm(result.get("parent_node")) or node_arg
            if parent:
                nodes.add(parent)
            for child in result.get("children", []) or []:
                child_id = _norm(child.get("node_id"))
                if child_id:
                    nodes.add(child_id)
                    if parent:
                        edges.add((parent, child_id))
        elif name in ("get_upstream_nodes", "get_downstream_nodes"):
            anchor = (
                _norm(result.get("target_node"))
                or _norm(result.get("source_node"))
                or node_arg
            )
            key = "upstream" if name == "get_upstream_nodes" else "downstream"
            for item in result.get(key, []) or []:
                other = _norm(item.get("node_id"))
                if other:
                    nodes.add(other)
                    if anchor:
                        if key == "upstream":
                            edges.add((other, anchor))
                        else:
                            edges.add((anchor, other))
        elif name == "get_related_systems":
            for key in ("upstream_connections", "downstream_connections"):
                for item in result.get(key, []) or []:
                    src = _norm(item.get("connected_via"))
                    dst = _norm(item.get("target_component"))
                    if src:
                        nodes.add(src)
                    if dst:
                        nodes.add(dst)
                    if src and dst:
                        edges.add((src, dst))
        elif name in ("diagnose_node", "get_node_sensors"):
            node_id = (
                _norm(result.get("node_id"))
                or _norm(result.get("component_node"))
                or node_arg
            )
            if node_id:
                nodes.add(node_id)
    return nodes, edges


def _reference_nodes(ep: Dict) -> Set[str]:
    gt = ep.get("ground_truth", {}) or {}
    meta = ep.get("metadata", {}) or {}
    nodes = [
        _norm(x)
        for x in (
            gt.get("diagnostic_path_nodes")
            or gt.get("reference_review_nodes")
            or meta.get("diagnostic_path_nodes")
            or []
        )
        if _norm(x)
    ]
    if not nodes:
        symptom = _norm(gt.get("symptom_node"))
        root = _norm(gt.get("root_cause_node"))
        if symptom:
            nodes.append(symptom)
        if root and not is_no_fault_node(root):
            nodes.append(root)
    return set(nodes)


def _reference_review_nodes(ep: Dict) -> Set[str]:
    gt = ep.get("ground_truth", {}) or {}
    meta = ep.get("metadata", {}) or {}
    nodes = [
        _norm(x)
        for x in (
            gt.get("reference_review_nodes")
            or meta.get("reference_review_nodes")
            or gt.get("diagnostic_path_nodes")
            or meta.get("diagnostic_path_nodes")
            or []
        )
        if _norm(x)
    ]
    return set(nodes)


def _normal_evidence_nodes(ep: Dict, reliability_threshold: float = 0.20) -> Set[str]:
    nodes: Set[str] = set()
    for result in _flatten_tool_results(ep):
        status = _norm(result.get("status"))
        if status not in ("normal", "success"):
            continue
        node = (
            _norm(result.get("node_id"))
            or _norm(result.get("component_node"))
        )
        if not node:
            continue
        if status == "success" and not result.get("readings_available"):
            continue
        if observation_reliability(result) >= reliability_threshold:
            nodes.add(node)
    return nodes


def no_fault_review_coverage(ep: Dict) -> float:
    """Coverage of required no-fault review nodes by reliable normal evidence."""
    if not _is_no_fault_episode(ep):
        return 0.0
    refs = _reference_review_nodes(ep)
    if not refs:
        return 1.0 if _strict_no_fault_match(ep) else 0.0
    normal_nodes = _normal_evidence_nodes(ep)
    return len(refs & normal_nodes) / len(refs)


def _strict_no_fault_evidence_closed(ep: Dict) -> bool:
    if not _strict_no_fault_match(ep):
        return False
    refs = _reference_review_nodes(ep)
    if not refs:
        return bool(_normal_evidence_nodes(ep))
    return refs <= _normal_evidence_nodes(ep)


def _tool_arg_node(call: Dict[str, Any]) -> str:
    args = call.get("arguments", {}) if isinstance(call, dict) else {}
    if not isinstance(args, dict):
        return ""
    return _norm(args.get("node_id"))


def _is_uncertain_diagnostic_result(result: Dict[str, Any]) -> bool:
    status = _norm(result.get("status"))
    if status in ("warning", "indeterminate"):
        return True
    fault_type = _norm(result.get("fault_type"))
    if fault_type == "uncertain":
        return status not in ("normal", "unknown", "error", "data_unavailable")
    return False


def _is_useful_sensor_result(result: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(result, dict):
        return False
    if _norm(result.get("status")) == "error":
        return False
    if result.get("readings_available") is True:
        return True
    readings = result.get("sensor_readings")
    if isinstance(readings, dict) and readings:
        return True
    sensors = result.get("sensors")
    return isinstance(sensors, list) and any(
        isinstance(sensor, dict) and "current_value" in sensor
        for sensor in sensors
    )


def diagnostic_accuracy(episodes: List[Dict]) -> float:
    """
    DA: Fraction of episodes where root cause node AND fault type are correct.
    """
    if not episodes:
        return 0.0

    correct = 0
    for ep in episodes:
        gt = ep.get("ground_truth", {})
        diag = ep.get("final_diagnosis")
        if diag is None:
            continue

        gt_node = _norm(gt.get("root_cause_node", ""))
        gt_fault = _norm(gt.get("fault_type", ""))
        diag_node = _norm(diag.get("root_cause_node", ""))
        diag_fault = _norm(diag.get("fault_type", ""))

        if _is_no_fault_episode(ep):
            if _strict_no_fault_match(ep):
                correct += 1
            continue

        node_match = _non_empty_substring_match(gt_node, diag_node)
        fault_match = fault_exact_match(gt_fault, diag_fault)

        if node_match and fault_match:
            correct += 1

    return correct / len(episodes)


def is_diagnosis_correct(ep: Dict) -> bool:
    """
    Return whether one episode's final diagnosis matches ground truth.

    This mirrors ``diagnostic_accuracy`` for a single episode and is used by
    logging/validation code so empty diagnosis fields cannot be counted as a
    successful substring match.
    """
    gt = ep.get("ground_truth", {})
    diag = ep.get("final_diagnosis")
    if diag is None:
        return False

    gt_node = _norm(gt.get("root_cause_node", ""))
    gt_fault = _norm(gt.get("fault_type", ""))
    diag_node = _norm(diag.get("root_cause_node", ""))
    diag_fault = _norm(diag.get("fault_type", ""))

    if _is_no_fault_episode(ep):
        return _strict_no_fault_match(ep)

    return (
        _non_empty_substring_match(gt_node, diag_node)
        and fault_exact_match(gt_fault, diag_fault)
    )


def tool_format_validity(episodes: List[Dict]) -> float:
    """
    TFV: Fraction of tool calls with syntactically correct JSON matching schema.
    """
    total_calls = 0
    valid_calls = 0

    valid_tool_names = {
        "get_system_overview", "get_node_children", "get_downstream_nodes",
        "get_upstream_nodes", "get_node_sensors", "get_related_systems",
        "diagnose_node",
    }

    for ep in episodes:
        for step in ep.get("agent_outputs", []):
            matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", step, re.DOTALL)
            for tc_str in matches:
                total_calls += 1
                try:
                    tc = json.loads(tc_str)
                    if (isinstance(tc, dict) and
                        "name" in tc and
                        "arguments" in tc and
                        isinstance(tc["arguments"], dict) and
                        tc["name"] in valid_tool_names):
                        valid_calls += 1
                    elif isinstance(tc, dict) and "name" in tc:
                        valid_calls += 0.5  # Partial credit
                except json.JSONDecodeError:
                    pass

    return valid_calls / max(total_calls, 1)


def diagnostic_completeness(episodes: List[Dict]) -> float:
    """
    DC: Fraction of final diagnoses that include all required fields.
    """
    if not episodes:
        return 0.0

    complete = 0
    total = 0

    for ep in episodes:
        diag = ep.get("final_diagnosis")
        total += 1

        if diag is None:
            continue

        required = ["root_cause_node", "fault_type", "confidence"]
        has_all = all(
            diag.get(f) is not None and str(diag.get(f)).strip() not in ("", "None")
            for f in required
        )

        if has_all:
            complete += 1
        else:
            # Partial credit
            present = sum(
                1 for f in required
                if diag.get(f) is not None and str(diag.get(f)).strip() not in ("", "None")
            )
            complete += present / len(required)

    return complete / max(total, 1)


def search_efficiency(episodes: List[Dict]) -> float:
    """
    SE: optimal_steps / actual_steps averaged across episodes.
    Perfect efficiency = 1.0.
    """
    if not episodes:
        return 0.0

    efficiencies = []
    for ep in episodes:
        actual = ep.get("n_tool_calls", 0)
        optimal = ep.get(
            "optimal_path_length",
            ep.get("ground_truth", {}).get("optimal_path_length", 3),
        )
        try:
            optimal = int(optimal)
        except (TypeError, ValueError):
            optimal = 3
        if optimal <= 0:
            optimal = 3

        if actual == 0 or ep.get("final_diagnosis") is None:
            efficiencies.append(0.0)
        else:
            efficiencies.append(min(1.0, optimal / actual))

    return sum(efficiencies) / len(efficiencies)


def _reference_edges(ep: Dict) -> Set[tuple]:
    """Reference propagation edges (TR, eq:exp_tr).

    These are the key symptom->...->root propagation/topology relations the agent
    is expected to traverse. Prefer explicit edge labels; fall back to deriving
    consecutive pairs from the ordered diagnostic path nodes.
    """
    gt = ep.get("ground_truth", {}) or {}
    meta = ep.get("metadata", {}) or {}
    raw_edges = (
        gt.get("reference_propagation_edges")
        or meta.get("reference_propagation_edges")
        or []
    )
    edges: Set[tuple] = set()
    for e in raw_edges:
        if isinstance(e, (list, tuple)) and len(e) == 2:
            u, v = _norm(e[0]), _norm(e[1])
            if u and v:
                edges.add((u, v))
    if edges:
        return edges
    # Fallback: consecutive pairs from ordered path nodes.
    path = [
        _norm(x)
        for x in (gt.get("diagnostic_path_nodes") or meta.get("diagnostic_path_nodes") or [])
        if _norm(x)
    ]
    for i in range(len(path) - 1):
        edges.add((path[i], path[i + 1]))
    return edges


def topology_rationality(episodes: List[Dict]) -> float:
    """TR: coverage of the reference propagation relations (eq:exp_tr).

    Paper definition: |R* ∩ A| / |R*| where R* is the set of key propagation
    EDGES (topology relations) on the symptom->root path and A is the set of
    visited relations. Edge coverage (rather than mere endpoint-node coverage)
    is what distinguishes genuine upstream tracing from incidental node visits,
    so this is the primary TR signal. Direction-insensitive matching is used
    because the agent may traverse a relation either way.

    For no-fault episodes (no propagation path), TR falls back to reference
    review-node coverage; if neither is available, to tool-invocation
    rationality so the metric is still defined.
    """
    if not episodes:
        return 0.0
    scores = []
    for ep in episodes:
        ref_edges = _reference_edges(ep)
        if ref_edges:
            _, visited_edges = _visited_nodes_and_edges(ep)
            if not visited_edges:
                scores.append(0.0)
                continue
            # Direction-insensitive intersection.
            visited_norm = set(visited_edges) | {(b, a) for (a, b) in visited_edges}
            hit = sum(
                1 for (u, v) in ref_edges
                if (u, v) in visited_norm or (v, u) in visited_norm
            )
            scores.append(hit / len(ref_edges))
            continue

        # No reference edges (e.g. no-fault): fall back to review-node coverage.
        ref_nodes = _reference_nodes(ep)
        if not ref_nodes:
            scores.append(tool_invocation_rationality([ep]))
            continue
        visited_nodes, _ = _visited_nodes_and_edges(ep)
        if not visited_nodes:
            scores.append(0.0)
            continue
        scores.append(len(ref_nodes & visited_nodes) / len(ref_nodes))
    return sum(scores) / len(scores)


def evidence_closure_rate(episodes: List[Dict]) -> float:
    """ECR: fraction of episodes whose final claim is supported and not conflicted."""
    if not episodes:
        return 0.0
    closed = 0
    for ep in episodes:
        diag = ep.get("final_diagnosis")
        if not diag:
            continue
        conflict = evidence_conflict_rate(ep.get("tool_results", []))
        support = diagnosis_evidence_support(diag, ep.get("tool_results", []))
        if _is_no_fault_episode(ep):
            if _strict_no_fault_evidence_closed(ep) and conflict <= 0.05:
                closed += 1
            continue
        if support >= 0.55 and conflict <= 0.35:
            closed += 1
    return closed / len(episodes)


def paper_metric_set(episodes: List[Dict]) -> Dict[str, float]:
    """Return the four metrics used in the experiments section."""
    return {
        "DA": diagnostic_accuracy(episodes),
        "TR": topology_rationality(episodes),
        "ECR": evidence_closure_rate(episodes),
        "SE": search_efficiency(episodes),
    }


def reasoning_authenticity(episodes: List[Dict]) -> float:
    """
    RA: Fraction of tool calls preceded by logically coherent reasoning.

    Checks for:
    - Presence of <think> tags before tool calls
    - Reasoning length (>20 chars)
    - No hallucinated facts (references to real tool results)
    - No logical jumps
    """
    total_calls = 0
    authentic_calls = 0

    for ep in episodes:
        outputs = ep.get("agent_outputs", [])
        tool_results = ep.get("tool_results", [])

        for i, text in enumerate(outputs):
            if "<tool_call>" not in text:
                continue

            total_calls += 1
            think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)

            if think_match:
                thought = think_match.group(1).strip()

                # Check reasoning quality
                score = 0.0

                # Length check
                if len(thought) >= 30:
                    score += 0.4
                elif len(thought) >= 15:
                    score += 0.2

                # References previous results (not hallucinating)
                if i > 0 and tool_results:
                    # Check if reasoning references data from previous tool results
                    prev_results = tool_results[:i]
                    references_data = any(
                        any(keyword in thought.lower() for keyword in
                            ["result", "shows", "indicates", "found", "detected",
                             "normal", "fault", "temperature", "pressure"])
                        for _ in prev_results
                    )
                    if references_data:
                        score += 0.3
                    else:
                        score += 0.1  # At least has reasoning

                # Logical connection to tool being called
                tc_match = re.search(r'"name":\s*"(\w+)"', text)
                if tc_match:
                    tool_name = tc_match.group(1)
                    tool_keywords = {
                        "get_system_overview": ["system", "overview", "building", "layout"],
                        "get_node_children": ["component", "children", "drill", "internal"],
                        "get_downstream_nodes": ["downstream", "propagat", "effect", "impact"],
                        "get_upstream_nodes": ["upstream", "source", "root cause", "origin"],
                        "diagnose_node": ["diagnos", "check", "status", "health", "fault"],
                        "get_node_sensors": ["sensor", "reading", "measurement", "data"],
                        "get_related_systems": ["related", "connected", "cross", "inter-system"],
                    }
                    keywords = tool_keywords.get(tool_name, [])
                    if any(kw in thought.lower() for kw in keywords):
                        score += 0.3
                    else:
                        score += 0.1

                authentic_calls += min(1.0, score)

    return authentic_calls / max(total_calls, 1)


def tool_invocation_rationality(episodes: List[Dict]) -> float:
    """
    TIR: Fraction of tool calls that are contextually appropriate.

    Checks for:
    - Not re-diagnosing already-checked nodes
    - Not calling get_downstream on leaf nodes
    - Logical progression of investigation
    - Not calling topology tools after finding root cause
    """
    total_calls = 0
    rational_calls = 0

    for ep in episodes:
        calls = _executed_tool_calls(ep)
        results = _parse_tool_results(ep)
        diagnosed_nodes: Set[str] = set()
        queried_children: Set[str] = set()
        discovered_nodes: Set[str] = set()
        found_fault = False

        for idx, tc in enumerate(calls):
            total_calls += 1
            is_rational = "_invalid" not in tc

            tool_name = tc.get("name", "")
            args = tc.get("arguments", {}) if isinstance(tc.get("arguments", {}), dict) else {}

            if idx == 0 and tool_name not in (
                "get_system_overview", "get_node_children",
            ):
                is_rational = False

            if tool_name == "get_node_children":
                node_id = args.get("node_id", "")
                if node_id in queried_children:
                    is_rational = False
                queried_children.add(node_id)

            if tool_name == "diagnose_node":
                node_id = args.get("node_id", "")
                if node_id in diagnosed_nodes:
                    is_rational = False
                diagnosed_nodes.add(node_id)
                if (
                    discovered_nodes
                    and not node_id.startswith("system::")
                    and node_id not in discovered_nodes
                ):
                    is_rational = False

            if found_fault and tool_name in (
                "get_system_overview", "get_related_systems", "get_node_children",
            ):
                is_rational = False

            if idx < len(results):
                result = results[idx]
                for child in result.get("children", []):
                    if isinstance(child, dict) and child.get("node_id"):
                        discovered_nodes.add(child["node_id"])
                if _norm(result.get("status")) == "fault":
                    found_fault = True

            if is_rational:
                rational_calls += 1

        if calls and ep.get("final_diagnosis") is None:
            total_calls += 1

    return rational_calls / max(total_calls, 1)


def root_system_accuracy(episodes: List[Dict]) -> float:
    """Fraction of episodes whose final diagnosis selects the correct system."""
    if not episodes:
        return 0.0
    correct = 0
    for ep in episodes:
        if _is_no_fault_episode(ep):
            correct += int(_strict_no_fault_match(ep))
            continue
        gt_system = _norm(ep.get("ground_truth", {}).get("root_cause_system", ""))
        diag_system = _diag_system(ep.get("final_diagnosis"))
        correct += int(bool(gt_system and diag_system == gt_system))
    return correct / len(episodes)


def root_node_accuracy(episodes: List[Dict]) -> float:
    """Fraction of episodes whose final diagnosis selects the correct node."""
    if not episodes:
        return 0.0
    correct = 0
    for ep in episodes:
        if _is_no_fault_episode(ep):
            correct += int(_strict_no_fault_match(ep))
            continue
        gt_node = ep.get("ground_truth", {}).get("root_cause_node", "")
        diag = ep.get("final_diagnosis") or {}
        correct += int(_non_empty_substring_match(gt_node, diag.get("root_cause_node", "")))
    return correct / len(episodes)


def fault_family_accuracy(episodes: List[Dict]) -> float:
    """Node-correct episodes with exact or same-family fault type credit."""
    if not episodes:
        return 0.0
    correct = 0
    for ep in episodes:
        if _is_no_fault_episode(ep):
            correct += int(_strict_no_fault_match(ep))
            continue
        gt = ep.get("ground_truth", {})
        diag = ep.get("final_diagnosis") or {}
        node_ok = _non_empty_substring_match(
            gt.get("root_cause_node", ""),
            diag.get("root_cause_node", ""),
        )
        fault_ok = (
            fault_exact_match(gt.get("fault_type", ""), diag.get("fault_type", ""))
            or same_fault_family(gt.get("fault_type", ""), diag.get("fault_type", ""))
        )
        correct += int(node_ok and fault_ok)
    return correct / len(episodes)


def strict_no_fault_accuracy(episodes: List[Dict]) -> float:
    no_fault = [ep for ep in episodes if _is_no_fault_episode(ep)]
    if not no_fault:
        return 0.0
    return sum(1 for ep in no_fault if _strict_no_fault_match(ep)) / len(no_fault)


def no_fault_review_coverage_aggregate(episodes: List[Dict]) -> float:
    no_fault = [ep for ep in episodes if _is_no_fault_episode(ep)]
    if not no_fault:
        return 0.0
    return sum(no_fault_review_coverage(ep) for ep in no_fault) / len(no_fault)


def wrong_system_rate(episodes: List[Dict]) -> float:
    fault_eps = [ep for ep in episodes if not _is_no_fault_episode(ep)]
    if not fault_eps:
        return 0.0
    wrong = 0
    for ep in fault_eps:
        gt_system = _norm(ep.get("ground_truth", {}).get("root_cause_system", ""))
        diag_system = _diag_system(ep.get("final_diagnosis"))
        if not diag_system or diag_system != gt_system:
            wrong += 1
    return wrong / len(fault_eps)


def oracle_consistency(episodes: List[Dict]) -> float:
    """How often the final diagnosis is supported by tool observations."""
    if not episodes:
        return 0.0
    scores = []
    for ep in episodes:
        diag = ep.get("final_diagnosis")
        if diag is None:
            scores.append(0.0)
            continue
        abnormal = [
            r for r in _flatten_tool_results(ep)
            if _norm(r.get("status")) in ("fault", "warning", "abnormal")
        ]
        if _diagnosis_says_no_fault(diag):
            scores.append(1.0 if not abnormal and _strict_no_fault_match(ep) else 0.0)
            continue

        diag_node = _norm(diag.get("root_cause_node", ""))
        diag_fault = diag.get("fault_type", "")
        same_node = [
            r for r in abnormal
            if _norm(r.get("node_id", "")) == diag_node
        ]
        if not same_node:
            scores.append(0.0)
            continue
        if any(
            fault_exact_match(r.get("fault_type", ""), diag_fault)
            or same_fault_family(r.get("fault_type", ""), diag_fault)
            or _norm(r.get("fault_type", "")) in ("", "none", "uncertain")
            for r in same_node
        ):
            scores.append(1.0)
        else:
            scores.append(0.5)
    return sum(scores) / len(scores)


def unsupported_tool_evidence_rate(episodes: List[Dict]) -> float:
    """Fraction of final fault diagnoses lacking same-node abnormal evidence."""
    fault_finals = [
        ep for ep in episodes
        if ep.get("final_diagnosis") is not None
        and not _diagnosis_says_no_fault(ep.get("final_diagnosis"))
    ]
    if not fault_finals:
        return 0.0
    unsupported = 0
    for ep in fault_finals:
        diag_node = _norm((ep.get("final_diagnosis") or {}).get("root_cause_node", ""))
        abnormal_nodes = {
            _norm(r.get("node_id", ""))
            for r in _flatten_tool_results(ep)
            if _norm(r.get("status")) in ("fault", "warning", "abnormal")
        }
        if diag_node not in abnormal_nodes:
            unsupported += 1
    return unsupported / len(fault_finals)


def tcead_evidence_support(episodes: List[Dict]) -> float:
    """Average TC-EAD support for the agent's final diagnosis."""
    if not episodes:
        return 0.0
    scores = [
        diagnosis_evidence_support(ep.get("final_diagnosis"), ep.get("tool_results", []))
        for ep in episodes
    ]
    return sum(scores) / len(scores)


def tcead_root_belief(episodes: List[Dict]) -> float:
    """Average fused belief assigned to the ground-truth root cause."""
    if not episodes:
        return 0.0
    scores = [
        ground_truth_belief_support(ep.get("ground_truth", {}), ep.get("tool_results", []))
        for ep in episodes
    ]
    return sum(scores) / len(scores)


def tcead_conflict_rate(episodes: List[Dict]) -> float:
    """Average reliability-weighted evidence conflict rate."""
    if not episodes:
        return 0.0
    scores = [evidence_conflict_rate(ep.get("tool_results", [])) for ep in episodes]
    return sum(scores) / len(scores)


def sensor_verification_rate(episodes: List[Dict]) -> float:
    """Fraction of episodes with uncertain node evidence that verify sensors.

    A verification is counted only when ``diagnose_node`` returns Warning or
    indeterminate evidence for a node and a later ``get_node_sensors`` call on
    the same node returns usable runtime readings.  This mirrors the SFT/RL
    policy and avoids giving credit for blind sensor calls after high-confidence
    Normal/Fault results.
    """
    eligible = 0
    verified = 0
    for ep in episodes:
        calls = _executed_tool_calls(ep)
        results = _parse_tool_results(ep)
        pending: Set[str] = set()
        saw_uncertain = False
        ep_verified = False

        for idx, call in enumerate(calls):
            name = call.get("name", "")
            node_id = _tool_arg_node(call)
            result = results[idx] if idx < len(results) else {}
            if name == "get_node_sensors":
                result_node = _norm(result.get("component_node")) or _norm(result.get("node_id")) or node_id
                if (
                    (node_id in pending or result_node in pending)
                    and _is_useful_sensor_result(result)
                ):
                    ep_verified = True
                    pending.discard(node_id)
                    pending.discard(result_node)
                continue

            if name != "diagnose_node" or not isinstance(result, dict):
                continue
            if _is_uncertain_diagnostic_result(result):
                result_node = _norm(result.get("node_id")) or node_id
                if result_node:
                    pending.add(result_node)
                    saw_uncertain = True

        if saw_uncertain:
            eligible += 1
            verified += int(ep_verified)

    return verified / max(eligible, 1)


def blind_sensor_call_rate(episodes: List[Dict]) -> float:
    """Fraction of sensor calls not justified by prior same-node uncertainty."""
    sensor_calls = 0
    blind = 0
    for ep in episodes:
        calls = _executed_tool_calls(ep)
        results = _parse_tool_results(ep)
        pending: Set[str] = set()
        for idx, call in enumerate(calls):
            name = call.get("name", "")
            node_id = _tool_arg_node(call)
            result = results[idx] if idx < len(results) else {}
            if name == "get_node_sensors":
                sensor_calls += 1
                result_node = (
                    _norm(result.get("component_node"))
                    or _norm(result.get("node_id"))
                    or node_id
                )
                if node_id not in pending and result_node not in pending:
                    blind += 1
                pending.discard(node_id)
                pending.discard(result_node)
                continue
            if (
                name == "diagnose_node"
                and isinstance(result, dict)
                and _is_uncertain_diagnostic_result(result)
            ):
                result_node = _norm(result.get("node_id")) or node_id
                if result_node:
                    pending.add(result_node)
    return blind / max(sensor_calls, 1)


def max_step_no_diagnosis_rate(episodes: List[Dict]) -> float:
    """Fraction of episodes that hit the rollout cap without final diagnosis."""
    if not episodes:
        return 0.0
    no_diag = 0
    for ep in episodes:
        if ep.get("final_diagnosis") is not None:
            continue
        reason = str(ep.get("termination_reason") or "")
        max_steps = int(ep.get("max_steps") or 15)
        if reason in {"max_steps", "max_outputs"} or int(ep.get("n_tool_calls", 0)) >= max_steps:
            no_diag += 1
    return no_diag / len(episodes)


def no_action_rate(episodes: List[Dict]) -> float:
    """Share of assistant outputs that contain no executable action."""
    total_outputs = 0
    total_no_action = 0
    for ep in episodes:
        outputs = int(ep.get("n_agent_outputs") or len(ep.get("agent_outputs", []) or []))
        if outputs <= 0:
            outputs = len(ep.get("turn_trace", []) or [])
        no_actions = int(ep.get("n_no_actions") or 0)
        if no_actions <= 0:
            no_actions = sum(
                1 for turn in ep.get("turn_trace", []) or []
                if turn.get("event") == "no_action"
            )
        total_outputs += outputs
        total_no_action += no_actions
    return total_no_action / max(total_outputs, 1)


def truncation_rate(episodes: List[Dict]) -> float:
    """Share of assistant outputs that appear truncated before an action."""
    total_outputs = 0
    total_truncated = 0
    for ep in episodes:
        outputs = int(ep.get("n_agent_outputs") or len(ep.get("agent_outputs", []) or []))
        if outputs <= 0:
            outputs = len(ep.get("turn_trace", []) or [])
        truncated = int(ep.get("n_truncated_outputs") or 0)
        if truncated <= 0:
            truncated = sum(
                1 for turn in ep.get("turn_trace", []) or []
                if bool(turn.get("truncated"))
            )
        total_outputs += outputs
        total_truncated += truncated
    return total_truncated / max(total_outputs, 1)


def tool_error_rate(episodes: List[Dict]) -> float:
    """Share of tool calls that returned hard errors."""
    total_tools = 0
    total_errors = 0
    for ep in episodes:
        tools = int(ep.get("n_tool_calls") or 0)
        if tools <= 0:
            tools = sum(
                1 for turn in ep.get("turn_trace", []) or []
                if turn.get("tool_call")
            )
        errors = int(ep.get("n_tool_errors") or 0)
        if errors <= 0:
            errors = sum(
                1 for turn in ep.get("turn_trace", []) or []
                if turn.get("event") == "tool_error"
            )
        if errors <= 0:
            errors = sum(
                1 for result in _parse_tool_results(ep)
                if _norm(result.get("status")) == "error"
            )
        total_tools += tools
        total_errors += errors
    return total_errors / max(total_tools, 1)


def generate_timeout_rate(episodes: List[Dict]) -> float:
    """Share of assistant generations that exceeded the generation budget."""
    total_outputs = 0
    total_timeouts = 0
    for ep in episodes:
        outputs = int(ep.get("n_agent_outputs") or len(ep.get("agent_outputs", []) or []))
        if outputs <= 0:
            outputs = len(ep.get("turn_trace", []) or [])
        timeouts = int(ep.get("n_generate_timeouts") or 0)
        if timeouts <= 0:
            timeouts = sum(
                1 for turn in ep.get("turn_trace", []) or []
                if bool(turn.get("generate_timeout"))
            )
        total_outputs += outputs
        total_timeouts += timeouts
    return total_timeouts / max(total_outputs, 1)


def action_completion_rate(episodes: List[Dict]) -> float:
    """Share of assistant outputs that produce a tool call or final diagnosis."""
    return max(0.0, 1.0 - no_action_rate(episodes))


def low_confidence_sensor_use(episodes: List[Dict]) -> float:
    """Fraction of low-confidence scenarios that call get_node_sensors."""
    low_conf = [
        ep for ep in episodes
        if "low_confidence" in str(ep.get("scenario_type", "")).lower()
        or "low_confidence" in str(ep.get("metadata", {}).get("scenario_type", "")).lower()
    ]
    if not low_conf:
        return 0.0
    used = 0
    for ep in low_conf:
        calls = _executed_tool_calls(ep)
        if any(call.get("name") == "get_node_sensors" for call in calls):
            used += 1
    return used / len(low_conf)


def _scenario_type_episodes(episodes: List[Dict], scenario_type: str) -> List[Dict]:
    key = canonical_scenario_type(scenario_type)
    return [
        ep for ep in episodes
        if canonical_scenario_type(ep.get("scenario_type", "")) == key
        or canonical_scenario_type(
            ep.get("metadata", {}).get("scenario_type", "")
        ) == key
    ]


def _has_tool_call(ep: Dict, tool_name: str) -> bool:
    return any(
        call.get("name") == tool_name
        for call in _executed_tool_calls(ep)
    )


def cross_trace_tool_use(episodes: List[Dict]) -> float:
    """Fraction of cross-system episodes using relationship/upstream tools."""
    cross_eps = [
        ep for ep in episodes
        if "cross_system" in str(ep.get("scenario_type", "")).lower()
        or "cross_system" in str(ep.get("metadata", {}).get("scenario_type", "")).lower()
    ]
    if not cross_eps:
        return 0.0
    used = sum(
        1 for ep in cross_eps
        if _has_tool_call(ep, "get_related_systems")
        or _has_tool_call(ep, "get_upstream_nodes")
    )
    return used / len(cross_eps)


def cross_root_system_accuracy(episodes: List[Dict]) -> float:
    """Root-system accuracy on cross-system episodes only."""
    cross_eps = [
        ep for ep in episodes
        if "cross_system" in str(ep.get("scenario_type", "")).lower()
        or "cross_system" in str(ep.get("metadata", {}).get("scenario_type", "")).lower()
    ]
    return root_system_accuracy(cross_eps) if cross_eps else 0.0


def compute_all_metrics(episodes: List[Dict]) -> Dict[str, float]:
    """
    Compute all 6 evaluation metrics for a set of episodes.

    Args:
        episodes: List of episode dicts, each containing:
            - ground_truth: Dict with root cause info
            - final_diagnosis: Parsed diagnosis dict
            - agent_outputs: List of agent text outputs
            - tool_results: List of tool result strings
            - n_tool_calls: Number of tool calls
            - optimal_path_length: Optimal path length

    Returns:
        Dict with all metric values.
    """
    metrics = {
        "n_episodes": len(episodes),
        "diagnostic_accuracy": diagnostic_accuracy(episodes),
        "root_system_accuracy": root_system_accuracy(episodes),
        "root_node_accuracy": root_node_accuracy(episodes),
        "fault_family_accuracy": fault_family_accuracy(episodes),
        "tool_format_validity": tool_format_validity(episodes),
        "diagnostic_completeness": diagnostic_completeness(episodes),
        "search_efficiency": search_efficiency(episodes),
        "topology_rationality": topology_rationality(episodes),
        "evidence_closure_rate": evidence_closure_rate(episodes),
        "reasoning_authenticity": reasoning_authenticity(episodes),
        "tool_invocation_rationality": tool_invocation_rationality(episodes),
        "oracle_consistency": oracle_consistency(episodes),
        "wrong_system_rate": wrong_system_rate(episodes),
        "unsupported_tool_evidence_rate": unsupported_tool_evidence_rate(episodes),
        "tcead_evidence_support": tcead_evidence_support(episodes),
        "tcead_root_belief": tcead_root_belief(episodes),
        "tcead_conflict_rate": tcead_conflict_rate(episodes),
        "sensor_verification_rate": sensor_verification_rate(episodes),
        "low_confidence_sensor_use": low_confidence_sensor_use(episodes),
        "blind_sensor_call_rate": blind_sensor_call_rate(episodes),
        "max_step_no_diagnosis_rate": max_step_no_diagnosis_rate(episodes),
        "no_action_rate": no_action_rate(episodes),
        "truncation_rate": truncation_rate(episodes),
        "tool_error_rate": tool_error_rate(episodes),
        "generate_timeout_rate": generate_timeout_rate(episodes),
        "action_completion_rate": action_completion_rate(episodes),
        "cross_trace_tool_use": cross_trace_tool_use(episodes),
        "cross_root_system_accuracy": cross_root_system_accuracy(episodes),
        "no_fault_review_coverage": no_fault_review_coverage_aggregate(episodes),
    }

    fault_episodes = [ep for ep in episodes if not _is_no_fault_episode(ep)]
    no_fault_episodes = [ep for ep in episodes if _is_no_fault_episode(ep)]
    fault_da = diagnostic_accuracy(fault_episodes) if fault_episodes else 0.0
    no_fault_da = diagnostic_accuracy(no_fault_episodes) if no_fault_episodes else 0.0
    if fault_episodes and no_fault_episodes:
        balanced_da = 0.5 * (fault_da + no_fault_da)
    elif fault_episodes:
        balanced_da = fault_da
    elif no_fault_episodes:
        balanced_da = no_fault_da
    else:
        balanced_da = 0.0
    metrics["fault_diagnostic_accuracy"] = fault_da
    metrics["no_fault_diagnostic_accuracy"] = no_fault_da
    metrics["strict_no_fault_accuracy"] = strict_no_fault_accuracy(episodes)
    metrics["balanced_diagnostic_accuracy"] = balanced_da

    cross_system_eps = _scenario_type_episodes(episodes, "cross_system")
    a_cross_eps = _scenario_type_episodes(episodes, "a_cross_system")
    metrics["cross_system_diagnostic_accuracy"] = (
        diagnostic_accuracy(cross_system_eps) if cross_system_eps else 0.0
    )
    metrics["a_cross_system_diagnostic_accuracy"] = (
        diagnostic_accuracy(a_cross_eps) if a_cross_eps else 0.0
    )
    if cross_system_eps and a_cross_eps:
        cross_balanced_da = 0.5 * (
            metrics["cross_system_diagnostic_accuracy"]
            + metrics["a_cross_system_diagnostic_accuracy"]
        )
    elif cross_system_eps:
        cross_balanced_da = metrics["cross_system_diagnostic_accuracy"]
    elif a_cross_eps:
        cross_balanced_da = metrics["a_cross_system_diagnostic_accuracy"]
    else:
        cross_balanced_da = 0.0
    metrics["cross_balanced_diagnostic_accuracy"] = cross_balanced_da

    # Compute aggregate score (weighted average)
    weights = {
        "diagnostic_accuracy": 0.25,
        "tool_format_validity": 0.15,
        "diagnostic_completeness": 0.15,
        "search_efficiency": 0.15,
        "reasoning_authenticity": 0.15,
        "tool_invocation_rationality": 0.15,
    }
    metrics["aggregate_score"] = sum(
        metrics[k] * w for k, w in weights.items()
    )
    metrics["selection_score"] = (
        0.34 * metrics["balanced_diagnostic_accuracy"]
        + 0.19 * metrics["root_node_accuracy"]
        + 0.15 * metrics["fault_family_accuracy"]
        + 0.10 * metrics["search_efficiency"]
        + 0.10 * metrics["tool_invocation_rationality"]
        + 0.04 * metrics["oracle_consistency"]
        + 0.05 * metrics["tcead_evidence_support"]
        + 0.02 * metrics["sensor_verification_rate"]
        + 0.04 * metrics["cross_balanced_diagnostic_accuracy"]
        + 0.02 * metrics["cross_trace_tool_use"]
        + 0.03 * metrics["reasoning_authenticity"]
        - 0.03 * metrics["tcead_conflict_rate"]
        - 0.08 * metrics["no_action_rate"]
        - 0.04 * metrics["truncation_rate"]
        - 0.05 * metrics["tool_error_rate"]
        - 0.03 * metrics["generate_timeout_rate"]
    )
    metrics["selection_score"] -= 0.05 * max(
        0.0, 0.40 - metrics["cross_balanced_diagnostic_accuracy"]
    )
    metrics["paper_DA"] = metrics["diagnostic_accuracy"]
    metrics["paper_TR"] = metrics["topology_rationality"]
    metrics["paper_ECR"] = metrics["evidence_closure_rate"]
    metrics["paper_SE"] = metrics["search_efficiency"]
    metrics["paper_mean"] = 0.25 * (
        metrics["paper_DA"]
        + metrics["paper_TR"]
        + metrics["paper_ECR"]
        + metrics["paper_SE"]
    )

    return metrics
