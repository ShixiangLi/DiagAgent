"""
Reward Functions — Multi-signal reward computation for RL training.

Implements a composite reward function for diagnostic agent episodes:
  R_total = w1*R_accuracy + w2*R_efficiency + w3*R_format + w4*R_reasoning + w5*R_completeness

Each signal is computed from the episode trajectory and ground truth.
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

from src.utils.io_utils import setup_logger

logger = setup_logger(__name__)

# Default reward weights
DEFAULT_WEIGHTS = {
    "accuracy": 0.35,
    "efficiency": 0.20,
    "format": 0.15,
    "reasoning": 0.15,
    "completeness": 0.15,
}


def compute_accuracy_reward(
    final_diagnosis: Optional[Dict],
    ground_truth: Dict[str, Any],
) -> float:
    """
    Compute diagnostic accuracy reward.

    +1.0: Correct root cause node AND fault type
    +0.5: Correct system but wrong specific node/fault
    +0.3: Correct that there IS a fault (for fault scenarios) or no fault (for normal)
     0.0: Completely wrong
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
    if gt_fault == "normal" or gt_node == "none":
        if "normal" in diag_status or "none" in diag_fault or "no fault" in diag_status:
            return 1.0
        return 0.0

    # Exact match
    node_exact = gt_node in diag_node or diag_node in gt_node
    fault_match = gt_fault in diag_fault or diag_fault in gt_fault

    if node_exact and fault_match:
        return 1.0

    # System-level match
    system_match = gt_system in diag_node or gt_system in str(final_diagnosis)
    if system_match and fault_match:
        return 0.5

    if system_match:
        return 0.3

    # At least identified that something is faulty
    if "fault" in diag_status or diag_fault not in ("none", "normal", ""):
        return 0.1

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


def compute_total_reward(
    agent_outputs: List[str],
    final_diagnosis: Optional[Dict],
    ground_truth: Dict[str, Any],
    n_tool_calls: int,
    optimal_path_length: int = 3,
    max_steps: int = 15,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute the total composite reward for an episode.

    Args:
        agent_outputs: List of all assistant text outputs in the episode.
        final_diagnosis: Parsed final diagnosis dict.
        ground_truth: Ground truth dict with root_cause_node, fault_type, etc.
        n_tool_calls: Number of tool calls made in the episode.
        optimal_path_length: Optimal number of tool calls.
        max_steps: Maximum allowed steps.
        weights: Optional custom weights for each reward signal.

    Returns:
        Dict with individual rewards and total reward.
    """
    w = weights or DEFAULT_WEIGHTS

    r_accuracy = compute_accuracy_reward(final_diagnosis, ground_truth)
    r_efficiency = compute_efficiency_reward(n_tool_calls, optimal_path_length, max_steps)
    r_format = compute_format_reward(agent_outputs)
    r_reasoning = compute_reasoning_reward(agent_outputs)
    r_completeness = compute_completeness_reward(final_diagnosis)

    total = (
        w["accuracy"] * r_accuracy
        + w["efficiency"] * r_efficiency
        + w["format"] * r_format
        + w["reasoning"] * r_reasoning
        + w["completeness"] * r_completeness
    )

    return {
        "total": round(total, 4),
        "accuracy": round(r_accuracy, 4),
        "efficiency": round(r_efficiency, 4),
        "format": round(r_format, 4),
        "reasoning": round(r_reasoning, 4),
        "completeness": round(r_completeness, 4),
        "n_tool_calls": n_tool_calls,
        "optimal_path_length": optimal_path_length,
    }
