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
from typing import Any, Dict, List, Optional, Tuple

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
    "consistency": 0.08,
}


def _norm_text(value: Any) -> str:
    return str(value or "").strip().lower()


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


def _flatten_diagnostic_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten direct diagnose_node outputs and status-summary node entries."""
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
    +0.5: Correct node, partially matching fault type
    +0.3: Correct system but wrong specific node/fault
     0.0: Wrong system, no diagnosis, or hallucinated fault type
    -0.5: Model says no_fault but ground truth has a real fault (PENALTY)
    """
    if final_diagnosis is None:
        return 0.0

    gt_node = ground_truth.get("root_cause_node", "").lower()
    gt_fault = ground_truth.get("fault_type", "").lower()
    gt_system = ground_truth.get("root_cause_system", "").lower()

    diag_node = str(final_diagnosis.get("root_cause_node", "")).lower()
    diag_fault = str(final_diagnosis.get("fault_type", "")).lower()
    diag_status = str(final_diagnosis.get("status", "")).lower()

    # Handle no-fault scenarios
    is_no_fault_gt = (
        gt_fault in ("normal", "no_fault", "none", "")
        or gt_node in ("none", "")
    )
    if is_no_fault_gt:
        diag_lower = (diag_status + " " + diag_fault).lower()
        if any(kw in diag_lower for kw in ("normal", "no fault", "no_fault", "none")):
            return 1.0
        return 0.0

    # CRITICAL v6: Detect false no_fault — model says "no fault" on a faulted scenario
    # This is the primary shortcut the model exploits for easy reward.
    # Active penalty (-0.5) makes this strategy unprofitable.
    diag_says_no_fault = any(
        kw in (diag_status + " " + diag_fault).lower()
        for kw in ("normal", "no fault", "no_fault", "none")
    )
    if diag_says_no_fault and not is_no_fault_gt:
        return -0.5

    # Exact match: node + fault type
    node_exact = gt_node in diag_node or diag_node in gt_node
    fault_match = gt_fault in diag_fault or diag_fault in gt_fault

    if node_exact and fault_match:
        return 1.0

    # Correct node, wrong fault
    if node_exact:
        return 0.5

    # System-level match
    system_match = gt_system in diag_node or gt_system in str(final_diagnosis).lower()
    if system_match and fault_match:
        return 0.5

    if system_match:
        return 0.3

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
    +0.4: final fault is at least on a node that was diagnosed
     0.0: no final diagnosis or no diagnostic evidence
    -1.0: final says no-fault despite Fault/Warning/Abnormal tool evidence
    -0.5: final fault contradicts the tool-confirmed fault node
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
        for r in abnormal:
            r_node = _norm_text(r.get("node_id"))
            r_fault = _norm_text(r.get("fault_type"))
            if r_node == diag_node and (
                not r_fault
                or r_fault in ("none", "uncertain")
                or r_fault in diag_fault
                or diag_fault in r_fault
            ):
                return 1.0
        return 0.4

    if abnormal:
        return -0.5
    return 0.0


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

    # Gate efficiency and completeness by accuracy:
    # No reward for "fast but wrong" or "complete but wrong"
    if r_accuracy < 0.3:
        r_efficiency = 0.0
        r_completeness = 0.0
    if final_diagnosis is None:
        r_reasoning = min(r_reasoning, 0.3)
        r_topology = 0.0

    total = (
        w.get("accuracy", 0.45) * r_accuracy
        + w.get("efficiency", 0.10) * r_efficiency
        + w.get("format", 0.10) * r_format
        + w.get("reasoning", 0.10) * r_reasoning
        + w.get("completeness", 0.10) * r_completeness
        + w.get("topology", 0.15) * r_topology
        + w.get("consistency", 0.0) * r_consistency
    )

    # Explicit terminal penalties keep "valid but unfinished" trajectories from
    # competing with real diagnoses.
    if final_diagnosis is None:
        total -= 0.20
        if n_tool_calls >= max_steps:
            total -= 0.10

    if r_consistency < 0:
        total += 0.10 * r_consistency

    # Add micro-noise based on tool call count to break exact reward ties
    # between rollouts. This ensures leave-one-out advantages are non-zero
    # even when all rollouts achieve the same diagnosis outcome.
    # Scale: ±0.005 (negligible vs total reward but breaks ties)
    efficiency_noise = -n_tool_calls * 0.001
    total += efficiency_noise

    return {
        "total": round(total, 4),
        "accuracy": round(r_accuracy, 4),
        "efficiency": round(r_efficiency, 4),
        "format": round(r_format, 4),
        "reasoning": round(r_reasoning, 4),
        "completeness": round(r_completeness, 4),
        "topology": round(r_topology, 4),
        "consistency": round(r_consistency, 4),
        "n_tool_calls": n_tool_calls,
        "optimal_path_length": optimal_path_length,
    }
