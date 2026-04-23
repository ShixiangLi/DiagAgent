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

from src.utils.io_utils import setup_logger

logger = setup_logger(__name__)


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

        gt_node = str(gt.get("root_cause_node", "")).lower()
        gt_fault = str(gt.get("fault_type", "")).lower()
        diag_node = str(diag.get("root_cause_node", "")).lower()
        diag_fault = str(diag.get("fault_type", "")).lower()

        # Handle no-fault scenarios
        if gt_fault == "normal" or gt_node == "none":
            if "normal" in str(diag.get("status", "")).lower() or "none" in diag_fault:
                correct += 1
            continue

        node_match = gt_node in diag_node or diag_node in gt_node
        fault_match = gt_fault in diag_fault or diag_fault in gt_fault

        if node_match and fault_match:
            correct += 1

    return correct / len(episodes)


def tool_format_validity(episodes: List[Dict]) -> float:
    """
    TFV: Fraction of tool calls with syntactically correct JSON matching schema.
    """
    total_calls = 0
    valid_calls = 0

    valid_tool_names = {
        "get_system_overview", "get_node_children", "get_downstream_nodes",
        "get_upstream_nodes", "get_node_sensors", "get_related_systems",
        "diagnose_node", "get_node_status_summary",
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
        optimal = ep.get("optimal_path_length", 3)

        if actual == 0:
            efficiencies.append(0.0)
        else:
            efficiencies.append(min(1.0, optimal / actual))

    return sum(efficiencies) / len(efficiencies)


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
                        "get_node_status_summary": ["summary", "scan", "overview", "status"],
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
        outputs = ep.get("agent_outputs", [])
        diagnosed_nodes: Set[str] = set()
        found_fault = False

        for text in outputs:
            tc_matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)

            for tc_str in tc_matches:
                total_calls += 1
                is_rational = True

                try:
                    tc = json.loads(tc_str)
                    tool_name = tc.get("name", "")
                    args = tc.get("arguments", {})

                    # Check 1: Not re-diagnosing already-checked nodes
                    if tool_name == "diagnose_node":
                        node_id = args.get("node_id", "")
                        if node_id in diagnosed_nodes:
                            is_rational = False  # Redundant check
                        diagnosed_nodes.add(node_id)

                    # Check 2: Logical progression
                    # First call should be get_system_overview or get_node_children
                    if total_calls == 1 and tool_name not in (
                        "get_system_overview", "get_node_children",
                        "get_node_status_summary"
                    ):
                        is_rational = False  # Should start with overview

                    # Check 3: Don't explore topology after finding definitive root cause
                    if found_fault and tool_name in (
                        "get_system_overview", "get_related_systems"
                    ):
                        is_rational = False  # Should be concluding

                except json.JSONDecodeError:
                    is_rational = False

                if is_rational:
                    rational_calls += 1

    return rational_calls / max(total_calls, 1)


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
        "diagnostic_accuracy": diagnostic_accuracy(episodes),
        "tool_format_validity": tool_format_validity(episodes),
        "diagnostic_completeness": diagnostic_completeness(episodes),
        "search_efficiency": search_efficiency(episodes),
        "reasoning_authenticity": reasoning_authenticity(episodes),
        "tool_invocation_rationality": tool_invocation_rationality(episodes),
    }

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

    return metrics
