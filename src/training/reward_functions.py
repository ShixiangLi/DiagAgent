"""
Reward Functions — Multi-signal reward computation for RL training.

Implements a composite reward function for diagnostic agent episodes:
  R_total = w1*R_accuracy + w2*R_efficiency + w3*R_format + w4*R_reasoning + w5*R_completeness

Each signal is computed from the episode trajectory and ground truth.

v2 changes:
  - Sharper accuracy gradient: correct=1.0, system_match=0.3, wrong=0.0 (was 0.1)
  - Efficiency gated by accuracy >= 0.5 (only for correct diagnoses)
  - Completeness gated by accuracy (no reward for complete-but-wrong)
  - Accuracy dominates total reward via exponent bonus
"""

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple

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
)
from src.training.epo import compute_epo_reward
from src.utils.io_utils import setup_logger

logger = setup_logger(__name__)

# Default reward weights (v7: accuracy-first, efficiency-aware)
# Efficiency and completeness are still gated by accuracy, so fast wrong
# answers do not receive path-length credit.
DEFAULT_WEIGHTS = {
    "accuracy": 0.65,
    "efficiency": 0.08,
    "format": 0.04,
    "reasoning": 0.04,
    "completeness": 0.05,
    "topology": 0.06,
    "consistency": 0.06,
    "evidence": 0.04,
    "sensor_verification": 0.08,
    "cross_trace": 0.06,
    "process": 0.04,
    "epo": 0.20,
}


_UNSUPPORTED_HEALTH_PATTERNS = (
    r"\banomaly score\b",
    r"\banomaly scores\b",
    r"\b\d+(?:\.\d+)?%\s*anomaly\b",
    r"\banomaly:\s*\d",
    r"\bhealth indicators?\b",
    r"\bsystem health\b",
    r"\bhealth scan\b",
    r"\bhighest-scoring system\b",
    r"\bhighest anomaly\b",
    r"\bsignificantly higher than other systems\b",
)


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


def _non_empty_substring_match(expected: str, actual: str) -> bool:
    """Allow fuzzy label matching only after both sides provide a value."""
    return non_empty_substring_match(expected, actual)


def _diag_system(final_diagnosis: Optional[Dict]) -> str:
    if not final_diagnosis:
        return ""
    system = node_system(final_diagnosis.get("root_cause_node", ""))
    if system:
        return system
    affected = final_diagnosis.get("affected_systems")
    if isinstance(affected, list) and affected:
        return _norm_text(affected[0])
    systems_checked = final_diagnosis.get("systems_checked")
    if isinstance(systems_checked, list) and systems_checked:
        return _norm_text(systems_checked[0])
    return ""


def _diagnosis_says_no_fault(final_diagnosis: Optional[Dict]) -> bool:
    if final_diagnosis is None:
        return False
    diag_text = (
        _norm_text(final_diagnosis.get("status"))
        + " "
        + _norm_text(final_diagnosis.get("fault_type"))
        + " "
        + _norm_text(final_diagnosis.get("root_cause_node"))
    )
    return any(kw in diag_text for kw in ("normal", "no fault", "no_fault", "none"))


def _parse_tool_results(tool_results: Optional[List[str]]) -> List[Dict[str, Any]]:
    parsed = []
    for result_text in tool_results or []:
        try:
            result = json.loads(result_text)
            if isinstance(result, dict):
                parsed.append(result)
        except (json.JSONDecodeError, TypeError):
            continue
    return parsed


def _extract_tool_calls(agent_outputs: Optional[List[str]]) -> List[Dict[str, Any]]:
    calls: List[Dict[str, Any]] = []
    for text in agent_outputs or []:
        for tc_str in re.findall(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            text,
            re.DOTALL,
        ):
            try:
                call = json.loads(tc_str)
                if isinstance(call, dict):
                    calls.append(call)
            except json.JSONDecodeError:
                continue
    return calls


def _tool_arg_node(call: Dict[str, Any]) -> str:
    args = call.get("arguments", {}) if isinstance(call, dict) else {}
    if not isinstance(args, dict):
        return ""
    return _norm_text(args.get("node_id"))


def _tool_name(call: Dict[str, Any]) -> str:
    return _norm_text(call.get("name"))


def _tool_signature(call: Dict[str, Any]) -> Tuple[str, str]:
    return (_tool_name(call), _tool_arg_node(call))


def _tool_result_is_error(result: Dict[str, Any]) -> bool:
    return isinstance(result, dict) and _norm_text(result.get("status")) == "error"


def _exact_node_ids_from_result(result: Dict[str, Any]) -> Set[str]:
    """Collect exact visible node ids from a tool result."""
    nodes: Set[str] = set()
    if not isinstance(result, dict):
        return nodes
    for key in ("node_id", "component_node", "parent_node", "source_node", "target_node"):
        value = _norm_text(result.get(key))
        if value and "::" in value:
            nodes.add(value)
    for child in result.get("children", []) or []:
        if isinstance(child, dict):
            value = _norm_text(child.get("node_id"))
            if value and "::" in value:
                nodes.add(value)
    for item_key in ("upstream", "downstream"):
        for item in result.get(item_key, []) or []:
            if isinstance(item, dict):
                value = _norm_text(item.get("node_id"))
                if value and "::" in value:
                    nodes.add(value)
    for conn_key in ("upstream_connections", "downstream_connections"):
        for item in result.get(conn_key, []) or []:
            if not isinstance(item, dict):
                continue
            for node_key in ("connected_via", "target_component"):
                value = _norm_text(item.get(node_key))
                if value and "::" in value:
                    nodes.add(value)
    return nodes


def _is_uncertain_diagnostic_result(result: Dict[str, Any]) -> bool:
    """Return True when a node diagnosis needs sensor-level verification."""
    status = _norm_text(result.get("status"))
    if status in ("warning", "indeterminate"):
        return True
    fault_type = _norm_text(result.get("fault_type"))
    if fault_type == "uncertain":
        return status not in ("normal", "unknown", "error", "data_unavailable")
    return False


def _is_useful_sensor_result(result: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(result, dict):
        return False
    if _norm_text(result.get("status")) == "error":
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


def _flatten_diagnostic_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten direct diagnose_node outputs and optional internal audit entries."""
    flattened = []
    for result in results:
        if result.get("node_id") or _norm_text(result.get("status")) in (
            "fault", "warning", "abnormal", "normal",
        ):
            flattened.append(result)
        for node_status in result.get("node_statuses", []) or []:
            flattened.append(node_status)
        for node_status in result.get("candidate_nodes", []) or []:
            if node_status not in flattened:
                flattened.append(node_status)
    return flattened


def compute_accuracy_reward(
    final_diagnosis: Optional[Dict],
    ground_truth: Dict[str, Any],
) -> float:
    """
    Compute diagnostic accuracy reward.

    v6: Added -0.5 penalty for false no_fault to prevent shortcut learning.

    +1.0: Correct root cause node AND fault type
    +0.35: Correct node, partially matching fault type
    +0.25: Correct system and matching fault type, wrong node
    +0.15: Correct system only
     0.0: Wrong system, no diagnosis, or hallucinated fault type
    -0.3: Model raises a fault on a no-fault scenario
    -0.6: Model says no_fault but ground truth has a real fault (PENALTY)
    """
    if final_diagnosis is None:
        return 0.0

    gt_node = _norm_text(ground_truth.get("root_cause_node", ""))
    gt_fault = _norm_text(ground_truth.get("fault_type", ""))
    gt_system = _norm_text(ground_truth.get("root_cause_system", ""))

    diag_node = _norm_text(final_diagnosis.get("root_cause_node", ""))
    diag_fault = _norm_text(final_diagnosis.get("fault_type", ""))
    diag_status = _norm_text(final_diagnosis.get("status", ""))

    # Handle no-fault scenarios
    is_no_fault_gt = is_no_fault_label(gt_fault) or is_no_fault_node(gt_node)
    if is_no_fault_gt:
        diag_lower = (diag_status + " " + diag_fault).lower()
        diag_node_is_none = is_no_fault_node(diag_node)
        if (
            diag_node_is_none
            and any(kw in diag_lower for kw in ("normal", "no fault", "no_fault", "none"))
        ):
            return 1.0
        return -0.3

    # CRITICAL v6: Detect false no_fault — model says "no fault" on a faulted scenario
    # This is the primary shortcut the model exploits for easy reward.
    # Active penalty (-0.5) makes this strategy unprofitable.
    if _diagnosis_says_no_fault(final_diagnosis) and not is_no_fault_gt:
        return -0.6

    if not diag_node or not diag_fault:
        return 0.0

    # Exact match: node + fault type
    node_exact = _non_empty_substring_match(gt_node, diag_node)
    fault_match = fault_exact_match(gt_fault, diag_fault)
    family_match = same_fault_family(gt_fault, diag_fault)

    if node_exact and fault_match:
        return 1.0

    # Correct node with a physically close fault family remains a useful
    # diagnostic step. It should be rewarded below exact GT, but above any
    # wrong-node/system diagnosis so RL can learn stable localization.
    if node_exact and family_match:
        return 0.70

    # Correct node, Oracle-supported or otherwise plausible but wrong fault.
    if node_exact:
        return 0.45

    # System-level match
    diag_system = _diag_system(final_diagnosis)
    system_match = diag_system == gt_system or gt_system in str(final_diagnosis).lower()
    if system_match and fault_match:
        return 0.25

    if system_match and family_match:
        return 0.20

    if system_match:
        return 0.10

    # Wrong system entirely → 0.0 (was 0.1 in v1)
    # This prevents the model from getting partial credit for hallucinated diagnoses
    return 0.0


def compute_efficiency_reward(
    n_tool_calls: int,
    optimal_path_length: int,
    max_steps: int = 15,
) -> float:
    """
    Compute search efficiency reward.

    Rewards shorter diagnostic paths relative to the optimal.
    Score = max(0, 1 - (actual - optimal) / max_steps)
    """
    if optimal_path_length <= 0:
        optimal_path_length = 3

    excess = max(0, n_tool_calls - optimal_path_length)
    reward = max(0.0, 1.0 - excess / max_steps)
    return reward


def compute_format_reward(agent_outputs: List[str]) -> float:
    """
    Compute tool format validity reward.

    Checks what fraction of tool calls have valid JSON format.
    """
    tool_calls = []
    for text in agent_outputs:
        matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
        tool_calls.extend(matches)

    if not tool_calls:
        return 0.5  # No tool calls is neutral

    valid = 0
    for tc_str in tool_calls:
        try:
            tc = json.loads(tc_str)
            # Check required fields
            if "name" in tc and "arguments" in tc:
                valid += 1
            elif "name" in tc:
                valid += 0.5
        except json.JSONDecodeError:
            pass

    return valid / len(tool_calls)


def compute_reasoning_reward(agent_outputs: List[str]) -> float:
    """
    Compute reasoning authenticity reward.

    Checks that tool calls are preceded by reasoning in <think> tags,
    and that reasoning is non-trivial (not just repeating the tool name).
    """
    total_calls = 0
    reasoned_calls = 0

    for text in agent_outputs:
        has_tool_call = "<tool_call>" in text
        if not has_tool_call:
            continue

        total_calls += 1

        # Check for think tags before tool call
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if think_match:
            thought = think_match.group(1).strip()
            # Reasoning should be at least 20 chars and not just the tool name
            if len(thought) >= 20:
                reasoned_calls += 1
            elif len(thought) >= 5:
                reasoned_calls += 0.5

    if total_calls == 0:
        return 0.5

    return reasoned_calls / total_calls


def compute_visible_evidence_grounding_penalty(
    agent_outputs: List[str],
    tool_results: Optional[List[str]] = None,
) -> float:
    """Penalize references to hidden health-score fields.

    The main RL/SFT experiment hides ``anomaly_score`` and ``health_status`` in
    ``get_system_overview``. If a rollout still claims to have seen anomaly
    scores, that reasoning is unsupported even when the final answer happens to
    be correct.
    """
    agent_text = "\n".join(agent_outputs or []).lower()
    if not agent_text:
        return 0.0

    visible_tool_text = "\n".join(tool_results or []).lower()
    health_fields_visible = (
        "anomaly_score" in visible_tool_text
        or "health_status" in visible_tool_text
    )
    if health_fields_visible:
        return 0.0

    hits = 0
    for pattern in _UNSUPPORTED_HEALTH_PATTERNS:
        hits += len(re.findall(pattern, agent_text))
    if hits <= 0:
        return 0.0

    return -min(0.30, 0.08 * hits)


def compute_completeness_reward(final_diagnosis: Optional[Dict]) -> float:
    """
    Compute diagnostic completeness reward.

    Checks presence of required fields in the final diagnosis.
    """
    if final_diagnosis is None:
        return 0.0

    required_fields = ["root_cause_node", "fault_type", "confidence"]
    optional_fields = ["affected_systems", "status"]

    score = 0.0
    total_weight = 0.0

    for field in required_fields:
        total_weight += 1.0
        value = final_diagnosis.get(field)
        if value is not None and str(value).strip() not in ("", "None", "null"):
            score += 1.0

    for field in optional_fields:
        total_weight += 0.5
        value = final_diagnosis.get(field)
        if value is not None and str(value).strip() not in ("", "None", "null"):
            score += 0.5

    return score / total_weight if total_weight > 0 else 0.0


def compute_topology_reward(agent_outputs: List[str], tool_results: List[str] = None) -> float:
    """
    Compute topology-adherence reward.

    Penalizes diagnosing nodes that were not previously discovered
    via get_node_children. Reinforces the BFS discovery pattern
    learned during SFT.

    +1.0: All diagnose_node calls target previously discovered nodes
    -scaled: Penalty proportional to fraction of "ghost" diagnoses
    """
    discovered_nodes = set()
    diagnosed_nodes = []

    # Parse tool results (observations) for discovered children
    if tool_results:
        for result_text in tool_results:
            try:
                result = json.loads(result_text)
                for child in result.get("children", []):
                    discovered_nodes.add(child.get("node_id", ""))
            except (json.JSONDecodeError, TypeError):
                pass

    # Also parse observations from agent_outputs (tool_call args)
    for text in agent_outputs:
        # Find all diagnose_node calls
        diag_matches = re.findall(
            r'"name":\s*"diagnose_node".*?"node_id":\s*"([^"]+)"', text
        )
        diagnosed_nodes.extend(diag_matches)

        # Find all get_node_children results embedded in the text
        children_matches = re.findall(
            r'"node_id":\s*"([^"]+)".*?"name":\s*"([^"]+)"', text
        )
        for nid, name in children_matches:
            if not nid.startswith("system::"):
                discovered_nodes.add(nid)

    if not diagnosed_nodes:
        return 0.0  # No diagnostic probing should not earn topology credit

    # Count how many diagnose_node targets were previously discovered
    valid = 0
    for node_id in diagnosed_nodes:
        if node_id.startswith("system::"):  # System-level always OK
            valid += 1
        elif node_id in discovered_nodes:
            valid += 1

    return valid / len(diagnosed_nodes)


def compute_tool_consistency_reward(
    final_diagnosis: Optional[Dict],
    tool_results: Optional[List[str]] = None,
) -> float:
    """
    Reward whether the final diagnosis is supported by tool observations.

    +1.0: final fault is backed by a prior Fault result on the same node/fault
    +0.7: no-fault conclusion after all diagnostic results are Normal/unknown
    +0.4: final fault is on a diagnosed node with non-normal evidence nearby
     0.0: no final diagnosis or no diagnostic evidence
    -1.0: final says no-fault despite Fault/Warning/Abnormal tool evidence
    -0.5: final fault contradicts the tool-confirmed fault node
    -0.4: final fault is on a node whose diagnostic evidence is Normal
    """
    if final_diagnosis is None:
        return 0.0

    results = _parse_tool_results(tool_results)
    diagnostic_results = _flatten_diagnostic_results(results)
    if not diagnostic_results:
        return 0.0

    abnormal = [
        r for r in diagnostic_results
        if _norm_text(r.get("status")) in ("fault", "warning", "abnormal")
    ]
    diag_node = _norm_text(final_diagnosis.get("root_cause_node"))
    diag_fault = _norm_text(final_diagnosis.get("fault_type"))

    if _diagnosis_says_no_fault(final_diagnosis):
        return -1.0 if abnormal else 0.7

    diagnosed_nodes = {_norm_text(r.get("node_id")) for r in diagnostic_results}
    if diag_node and diag_node in diagnosed_nodes:
        same_node_results = [
            r for r in diagnostic_results
            if _norm_text(r.get("node_id")) == diag_node
        ]
        same_node_abnormal = [
            r for r in abnormal
            if _norm_text(r.get("node_id")) == diag_node
        ]
        for r in same_node_abnormal:
            r_node = _norm_text(r.get("node_id"))
            r_fault = _norm_text(r.get("fault_type"))
            weak_fallback = (
                _norm_text(r.get("model_source")) == "per_node"
                or _norm_text(r.get("calibration")) == "weak_fallback_model"
            )
            if r_node == diag_node and (
                not r_fault
                or r_fault in ("none", "uncertain")
                or fault_exact_match(r_fault, diag_fault)
                or same_fault_family(r_fault, diag_fault)
            ):
                if weak_fallback:
                    return 0.25
                return 1.0
        if same_node_abnormal:
            return 0.5
        if any(_norm_text(r.get("status")) == "normal" for r in same_node_results):
            return -0.4
        return 0.4 if abnormal else -0.3

    if abnormal:
        return -0.5
    return -0.3


def summarize_sensor_verification(
    agent_outputs: List[str],
    tool_results: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Summarize uncertainty-gated sensor acquisition behavior.

    A useful sensor call is intentionally conditional: first the agent observes
    uncertain node evidence from diagnose_node, then it calls get_node_sensors
    for that same node. This makes sensor acquisition a local uncertainty
    resolver instead of a blanket extra step that would slow down high
    confidence single-system, cross-system, or no-fault cases.
    """
    calls = _extract_tool_calls(agent_outputs)
    parsed_results = _parse_tool_results(tool_results)

    pending_uncertain: set[str] = set()
    verified_uncertain: set[str] = set()
    high_conf_fault_nodes: set[str] = set()
    saw_uncertain = False
    sensor_calls = 0
    useful_sensor_calls = 0
    blind_sensor_calls = 0
    wrong_or_stale_sensor_calls = 0
    post_fault_sensor_calls = 0

    for idx, call in enumerate(calls):
        tool_name = call.get("name", "")
        node_id = _tool_arg_node(call)
        result = parsed_results[idx] if idx < len(parsed_results) else None

        if tool_name == "get_node_sensors":
            sensor_calls += 1
            useful = _is_useful_sensor_result(result)
            if useful:
                useful_sensor_calls += 1

            if node_id and node_id in pending_uncertain:
                if useful:
                    verified_uncertain.add(node_id)
                    pending_uncertain.discard(node_id)
            else:
                if not saw_uncertain:
                    blind_sensor_calls += 1
                else:
                    wrong_or_stale_sensor_calls += 1
                if node_id and node_id in high_conf_fault_nodes:
                    post_fault_sensor_calls += 1

        if not isinstance(result, dict) or tool_name != "diagnose_node":
            continue

        result_node = _norm_text(result.get("node_id")) or node_id
        if not result_node:
            continue

        if _is_uncertain_diagnostic_result(result):
            saw_uncertain = True
            pending_uncertain.add(result_node)
            continue

        status = _norm_text(result.get("status"))
        try:
            confidence = float(result.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        if status == "fault" and confidence >= 0.70:
            high_conf_fault_nodes.add(result_node)

    total_uncertain = len(verified_uncertain) + len(pending_uncertain)
    coverage = (
        len(verified_uncertain) / total_uncertain
        if total_uncertain > 0 else 0.0
    )

    return {
        "sensor_calls": sensor_calls,
        "useful_sensor_calls": useful_sensor_calls,
        "uncertain_nodes": total_uncertain,
        "verified_uncertain_nodes": len(verified_uncertain),
        "pending_uncertain_nodes": len(pending_uncertain),
        "uncertainty_coverage": coverage,
        "blind_sensor_calls": blind_sensor_calls,
        "wrong_or_stale_sensor_calls": wrong_or_stale_sensor_calls,
        "post_fault_sensor_calls": post_fault_sensor_calls,
        "saw_uncertain": saw_uncertain,
    }


def compute_sensor_verification_reward(
    agent_outputs: List[str],
    tool_results: Optional[List[str]] = None,
) -> float:
    """Reward targeted sensor verification after uncertain node evidence.

    get_node_sensors should not be called mechanically on every node. It is
    valuable when diagnose_node returns Warning/unknown or a low-confidence
    Normal, because that is the point where the policy should verify raw
    measurements instead of prematurely concluding no-fault or looping through
    unrelated components.

    Score range:
      +1.0: every uncertain diagnosed node is followed by get_node_sensors
      +0.5: sensor verification is used, but not for every uncertain node
       0.0: no uncertain diagnostic evidence was seen
      -1.0: uncertainty was observed but never sensor-verified
      -0.4: sensors were called without an uncertainty signal
    """
    summary = summarize_sensor_verification(agent_outputs, tool_results)
    if summary["sensor_calls"] <= 0 and summary["uncertain_nodes"] <= 0:
        return 0.0

    if summary["uncertain_nodes"] <= 0:
        misuse = summary["blind_sensor_calls"] + summary["post_fault_sensor_calls"]
        if misuse <= 0:
            return 0.0
        return max(-0.5, -0.25 - 0.10 * (misuse - 1))

    coverage = float(summary["uncertainty_coverage"])
    stale_penalty = min(0.25, 0.08 * summary["wrong_or_stale_sensor_calls"])
    if coverage >= 0.999:
        return max(0.75, 1.0 - stale_penalty)
    if coverage > 0.0:
        return max(-0.2, 0.5 * coverage - stale_penalty)
    return -1.0


def compute_process_reward(
    agent_outputs: List[str],
    tool_results: Optional[List[str]] = None,
    final_diagnosis: Optional[Dict] = None,
    n_tool_calls: int = 0,
    optimal_path_length: int = 3,
    max_steps: int = 15,
) -> float:
    """Reward concise, progressive tool use and penalize stalled trajectories.

    This is a process-level shaping signal, not an accuracy substitute. It
    targets the RL failure mode where the agent keeps calling tools until the
    rollout cap and still emits no diagnosis. The signal is intentionally small
    in the default weighted total, but it gives GRPO a cleaner contrast between
    efficient diagnostic paths and repeated/unfinished searches.
    """
    calls = _extract_tool_calls(agent_outputs)
    if not calls:
        return 0.0

    score = 1.0
    max_steps = max(1, int(max_steps or 15))
    optimal_path_length = max(1, int(optimal_path_length or 3))

    # Strongly mark trajectories that hit the cap without a final answer.
    if final_diagnosis is None and n_tool_calls >= max_steps:
        score -= 1.2
    elif final_diagnosis is None:
        score -= 0.6

    # Penalize long excess search even when a final answer exists.
    excess = max(0, n_tool_calls - optimal_path_length)
    if excess:
        score -= min(0.5, 0.08 * excess)

    # Repeating exactly the same tool/node is usually a stall in this
    # environment; a second confirmation is allowed, heavier repetition is not.
    seen_counts: Dict[Tuple[str, str], int] = {}
    repeated = 0
    for call in calls:
        sig = _tool_signature(call)
        if not sig[0]:
            continue
        seen_counts[sig] = seen_counts.get(sig, 0) + 1
        if seen_counts[sig] > 2:
            repeated += 1
    if repeated:
        score -= min(0.5, 0.12 * repeated)

    parsed = _parse_tool_results(tool_results)

    # Tool errors are allowed as visible evidence, but a healthy policy should
    # recover from them and should not receive process credit for ghost-node
    # calls such as boiler_plant::Heating_Coil.
    tool_errors = sum(1 for result in parsed if _tool_result_is_error(result))
    if tool_errors:
        score -= min(0.8, 0.30 * tool_errors)

    visible_nodes: Set[str] = set()
    ghost_diagnoses = 0
    for idx, call in enumerate(calls):
        node_id = _tool_arg_node(call)
        if _tool_name(call) == "diagnose_node" and node_id:
            if (
                visible_nodes
                and not node_id.startswith("system::")
                and node_id not in visible_nodes
            ):
                ghost_diagnoses += 1
        if idx < len(parsed):
            visible_nodes.update(_exact_node_ids_from_result(parsed[idx]))
    if ghost_diagnoses:
        score -= min(0.7, 0.25 * ghost_diagnoses)

    # If a Fault/Warning is observed, the policy should either verify that
    # candidate or conclude soon, not continue a broad system sweep.
    first_abnormal_idx: Optional[int] = None
    for idx, result in enumerate(parsed):
        if _norm_text(result.get("status")) in ("fault", "warning", "abnormal"):
            first_abnormal_idx = idx
            break
    if first_abnormal_idx is not None:
        later_calls = max(0, len(calls) - first_abnormal_idx - 1)
        if later_calls > 3:
            score -= min(0.4, 0.08 * (later_calls - 3))

    if score >= 0.8:
        return 1.0
    if score >= 0.4:
        return 0.5
    if score >= 0.0:
        return 0.0
    return max(-1.0, score)


def _is_cross_system_ground_truth(ground_truth: Dict[str, Any]) -> bool:
    affected = ground_truth.get("affected_systems")
    if isinstance(affected, list):
        systems = {
            _norm_text(s)
            for s in affected
            if _norm_text(s) and _norm_text(s) not in ("none", "normal")
        }
        if len(systems) > 1:
            return True
    root_system = _norm_text(ground_truth.get("root_cause_system"))
    root_node = _norm_text(ground_truth.get("root_cause_node"))
    if root_system in ("boiler_plant", "chiller_plant"):
        # Central plant faults that affect downstream systems are the main
        # cross-system cases in this benchmark.
        return root_node.startswith(root_system + "::")
    return False


def compute_cross_trace_reward(
    agent_outputs: List[str],
    final_diagnosis: Optional[Dict],
    ground_truth: Dict[str, Any],
    tool_results: Optional[List[str]] = None,
) -> float:
    """Reward cross-system upstream tracing only for cross-system GT cases.

    This is intentionally neutral for ordinary single-system/no-fault cases.
    For cross-system faults, the agent should not stop at the downstream symptom
    system. It should query a relationship/upstream tool and diagnose the
    central root system before issuing a conclusion.
    """
    calls = _extract_tool_calls(agent_outputs)
    names = [_tool_name(call) for call in calls]
    diagnosed_nodes = [
        _tool_arg_node(call)
        for call in calls
        if _tool_name(call) == "diagnose_node"
    ]
    traced_topology = any(
        name in ("get_related_systems", "get_upstream_nodes")
        for name in names
    )

    if not _is_cross_system_ground_truth(ground_truth):
        if not traced_topology:
            return 0.0
        score = -0.25
        if final_diagnosis is None:
            score -= 0.25
        elif _diag_system(final_diagnosis):
            gt_system = _norm_text(ground_truth.get("root_cause_system"))
            diag_system = _diag_system(final_diagnosis)
            if gt_system and diag_system != gt_system:
                score -= 0.20
        if "get_upstream_nodes" in names:
            score -= 0.10
        return max(-1.0, min(0.0, score))

    gt_system = _norm_text(ground_truth.get("root_cause_system"))
    gt_node = _norm_text(ground_truth.get("root_cause_node"))
    diagnosed_root_system = any(
        node.startswith(gt_system + "::")
        for node in diagnosed_nodes
    )
    diagnosed_root_node = gt_node in diagnosed_nodes

    score = 0.0
    if traced_topology:
        score += 0.15
    if "get_upstream_nodes" in names:
        score += 0.15
    if diagnosed_root_system:
        score += 0.35
    if diagnosed_root_node:
        score += 0.35

    if final_diagnosis is not None:
        diag_node = _norm_text(final_diagnosis.get("root_cause_node"))
        diag_system = node_system(diag_node)
        if _diagnosis_says_no_fault(final_diagnosis):
            score -= 0.70
        elif diag_system and diag_system != gt_system:
            score -= 0.45
        elif diag_system == gt_system:
            score += 0.10

    if not traced_topology and not diagnosed_root_system:
        score -= 0.35
    elif traced_topology and not diagnosed_root_system:
        score -= 0.35
    if final_diagnosis is None:
        score -= 0.45

    return max(-1.0, min(1.0, score))


def compute_total_reward(
    agent_outputs: List[str],
    final_diagnosis: Optional[Dict],
    ground_truth: Dict[str, Any],
    n_tool_calls: int,
    optimal_path_length: int = 3,
    max_steps: int = 15,
    weights: Optional[Dict[str, float]] = None,
    tool_results: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute the total composite reward for an episode.

    v3 additions:
    - topology_adherence: penalizes ghost-node diagnoses
    - Maintains v2 accuracy-gating for efficiency/completeness

    Args:
        agent_outputs: List of all assistant text outputs in the episode.
        final_diagnosis: Parsed final diagnosis dict.
        ground_truth: Ground truth dict with root_cause_node, fault_type, etc.
        n_tool_calls: Number of tool calls made in the episode.
        optimal_path_length: Optimal number of tool calls.
        max_steps: Maximum allowed steps.
        weights: Optional custom weights for each reward signal.
        tool_results: List of tool result strings for topology checking.

    Returns:
        Dict with individual rewards and total reward.
    """
    w = weights or DEFAULT_WEIGHTS

    r_accuracy = compute_accuracy_reward(final_diagnosis, ground_truth)
    r_efficiency = compute_efficiency_reward(n_tool_calls, optimal_path_length, max_steps)
    r_format = compute_format_reward(agent_outputs)
    r_reasoning = compute_reasoning_reward(agent_outputs)
    r_completeness = compute_completeness_reward(final_diagnosis)
    r_topology = compute_topology_reward(agent_outputs, tool_results)
    r_consistency = compute_tool_consistency_reward(final_diagnosis, tool_results)
    r_evidence = diagnosis_evidence_support(final_diagnosis, tool_results)
    r_conflict = evidence_conflict_rate(tool_results)
    r_grounding = compute_visible_evidence_grounding_penalty(agent_outputs, tool_results)
    sensor_summary = summarize_sensor_verification(agent_outputs, tool_results)
    r_sensor_verification = compute_sensor_verification_reward(agent_outputs, tool_results)
    r_cross_trace = compute_cross_trace_reward(
        agent_outputs=agent_outputs,
        final_diagnosis=final_diagnosis,
        ground_truth=ground_truth,
        tool_results=tool_results,
    )
    r_process = compute_process_reward(
        agent_outputs=agent_outputs,
        tool_results=tool_results,
        final_diagnosis=final_diagnosis,
        n_tool_calls=n_tool_calls,
        optimal_path_length=optimal_path_length,
        max_steps=max_steps,
    )
    epo_config = w.get("epo_config") if isinstance(w.get("epo_config"), dict) else None
    epo_reward = compute_epo_reward(
        agent_outputs=agent_outputs,
        final_diagnosis=final_diagnosis,
        ground_truth=ground_truth,
        n_tool_calls=n_tool_calls,
        optimal_path_length=optimal_path_length,
        max_steps=max_steps,
        tool_results=tool_results,
        config=epo_config,
    )
    r_epo = epo_reward["epo"]

    # Gate efficiency and completeness by accuracy:
    # No reward for "fast but wrong" or "complete but wrong"
    if r_accuracy < 0.3:
        r_efficiency = 0.0
        r_completeness = 0.0
    if final_diagnosis is None:
        r_reasoning = min(r_reasoning, 0.3)
        r_topology = 0.0
    elif r_accuracy <= 0.0:
        r_completeness = 0.0
        # A wrong diagnosis can still cite a weak tool observation; do not let
        # evidence shaping compete with ground-truth diagnostic accuracy.
        r_evidence = min(r_evidence, 0.20)

    total = (
        w.get("accuracy", 0.45) * r_accuracy
        + w.get("efficiency", 0.10) * r_efficiency
        + w.get("format", 0.10) * r_format
        + w.get("reasoning", 0.10) * r_reasoning
        + w.get("completeness", 0.10) * r_completeness
        + w.get("topology", 0.15) * r_topology
        + w.get("consistency", 0.0) * r_consistency
        + w.get("evidence", 0.0) * r_evidence
        + w.get("sensor_verification", 0.0) * r_sensor_verification
        + w.get("cross_trace", 0.0) * r_cross_trace
        + w.get("process", 0.0) * r_process
        + w.get("epo", 0.0) * r_epo
    )

    # Explicit terminal penalties keep "valid but unfinished" trajectories from
    # competing with real diagnoses.
    if final_diagnosis is None:
        total -= 0.30
        if n_tool_calls >= max_steps:
            total -= 0.15
    elif r_accuracy <= 0.0:
        total -= 0.12

    if r_consistency < 0:
        total += 0.15 * r_consistency
    if r_conflict > 0:
        total -= 0.05 * r_conflict
    if r_sensor_verification < 0:
        total += 0.10 * r_sensor_verification
    if r_cross_trace < 0:
        total += 0.10 * r_cross_trace
    if r_process < 0:
        total += 0.08 * r_process
    if epo_reward.get("epo_mode_violated", 0.0) > 0:
        total -= 0.08 * epo_reward.get("epo_violation_rate", 0.0)
    total += r_grounding

    # Add micro-noise based on tool call count to break exact reward ties
    # between rollouts. This ensures leave-one-out advantages are non-zero
    # even when all rollouts achieve the same diagnosis outcome.
    # Scale: ±0.005 (negligible vs total reward but breaks ties)
    efficiency_noise = -n_tool_calls * 0.001
    total += efficiency_noise

    result = {
        "total": round(total, 4),
        "accuracy": round(r_accuracy, 4),
        "efficiency": round(r_efficiency, 4),
        "format": round(r_format, 4),
        "reasoning": round(r_reasoning, 4),
        "completeness": round(r_completeness, 4),
        "topology": round(r_topology, 4),
        "consistency": round(r_consistency, 4),
        "evidence": round(r_evidence, 4),
        "sensor_verification": round(r_sensor_verification, 4),
        "cross_trace": round(r_cross_trace, 4),
        "sensor_calls": int(sensor_summary["sensor_calls"]),
        "sensor_uncertain_nodes": int(sensor_summary["uncertain_nodes"]),
        "sensor_verified_uncertain_nodes": int(sensor_summary["verified_uncertain_nodes"]),
        "sensor_post_fault_calls": int(sensor_summary["post_fault_sensor_calls"]),
        "sensor_blind_calls": int(sensor_summary["blind_sensor_calls"]),
        "process": round(r_process, 4),
        "evidence_conflict": round(r_conflict, 4),
        "grounding": round(r_grounding, 4),
        "n_tool_calls": n_tool_calls,
        "optimal_path_length": optimal_path_length,
    }
    result.update(epo_reward)
    return result
