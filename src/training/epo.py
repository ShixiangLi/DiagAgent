"""Evidence-Potential Optimization reward primitives.

EPO converts a multi-step tool trajectory into evidence states.  The code is
deterministic and uses only visible tool outputs, so it can run during local
validation, evaluation, and cloud RL rollouts without model-specific hooks.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.evaluation.evidence_fusion import observation_reliability
from src.evaluation.fault_taxonomy import (
    fault_exact_match,
    is_no_fault_label,
    is_no_fault_node,
    node_system,
    same_fault_family,
)


DEFAULT_EPO_CONFIG = {
    "enabled": True,
    "use_state_machine": True,
    "use_potential_reward": True,
    "weight": 0.20,
    # lambda_epo (eq:pf_evp_objective / sensitivity table): strength of the
    # evidence-potential + state-machine shaping. 0.0 ⇒ pure outcome reward
    # (Outcome-PPO-like); the paper uses 0.75.
    "lambda_epo": 0.75,
    "discount": 0.95,            # xi: step-reward discount in eq:epo_reward
    "lambda_chi": 0.45,          # closure conflict weight (eq:epo_closure_score)
    "lambda_entropy": 0.20,      # closure entropy weight (eq:epo_closure_score)
    "lambda_violation": 0.65,    # lambda_m: conflict penalty in the potential
    "tool_cost": 0.025,          # lambda_d: per-step tool cost
    "closed_threshold": 0.25,    # delta_c
    "open_threshold": 0.10,      # delta_o
    "violation_threshold": 0.35, # delta_v
    "recover_threshold": 0.15,   # delta_r
    "reliability_threshold": 0.55,  # delta_rho
    "terminal_correct": 1.0,     # R_c
    "terminal_wrong": 1.0,       # R_w
    "terminal_open": 0.35,       # R_o
    "smooth": 1e-3,
    "same_system_projection": 0.18,
    "one_hop_projection": 0.55,
    "two_hop_projection": 0.30,
    "normal_alternative_support": 0.15,
    "no_fault_review_threshold": 1.0,
    "no_fault_normal_reliability": 0.20,
}

# Evidence-state machine labels (eq:epo_state_space). ``conflict`` matches the
# paper; ``violated`` is accepted as a legacy alias on read.
EVIDENCE_OPEN = "open"
EVIDENCE_CLOSED = "closed"
EVIDENCE_CONFLICT = "conflict"

ABNORMAL_STATUSES = {"fault", "warning", "abnormal", "alert"}
NORMAL_STATUSES = {"normal", "success"}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _parse_json(raw: Any) -> Optional[Dict[str, Any]]:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_tool_calls(agent_outputs: Optional[List[str]]) -> List[Dict[str, Any]]:
    calls = []
    for text in agent_outputs or []:
        for match in re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL):
            parsed = _parse_json(match)
            if parsed is not None:
                calls.append(parsed)
    return calls


def _parse_tool_results(tool_results: Optional[Iterable[Any]]) -> List[Dict[str, Any]]:
    parsed = []
    for raw in tool_results or []:
        item = _parse_json(raw)
        if item is not None:
            parsed.append(item)
    return parsed


def _reference_review_nodes(ground_truth: Dict[str, Any]) -> List[str]:
    return [
        _candidate_key(node)
        for node in (
            ground_truth.get("reference_review_nodes")
            or ground_truth.get("diagnostic_path_nodes")
            or []
        )
        if _candidate_key(node)
    ]


def _normal_evidence_nodes(
    results: Iterable[Dict[str, Any]],
    reliability_threshold: float,
) -> set[str]:
    nodes: set[str] = set()
    for result in results:
        status = _status(result)
        if status not in NORMAL_STATUSES:
            continue
        node = _candidate_key(
            result.get("node_id")
            or result.get("component_node")
        )
        if not node:
            continue
        if status == "success" and not result.get("readings_available"):
            continue
        if observation_reliability(result) >= reliability_threshold:
            nodes.add(node)
    return nodes


def _no_fault_review_coverage(
    ground_truth: Dict[str, Any],
    results: Iterable[Dict[str, Any]],
    config: Dict[str, Any],
) -> float:
    refs = set(_reference_review_nodes(ground_truth))
    if not refs:
        return 1.0
    normal_nodes = _normal_evidence_nodes(
        results,
        float(config["no_fault_normal_reliability"]),
    )
    return len(refs & normal_nodes) / len(refs)


def _tool_arg(call: Dict[str, Any], name: str) -> str:
    args = call.get("arguments", {}) if isinstance(call, dict) else {}
    if not isinstance(args, dict):
        return ""
    return str(args.get(name) or "").strip()


def _candidate_key(node_id: str) -> str:
    return str(node_id or "").strip()


def _status(result: Dict[str, Any]) -> str:
    return _norm(result.get("status"))


def _confidence(result: Dict[str, Any]) -> float:
    try:
        return _clip(float(result.get("confidence") or 0.0))
    except (TypeError, ValueError):
        return 0.0


def _normalize(dist: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(v)) for v in dist.values())
    if total <= 0.0:
        n = max(len(dist), 1)
        return {k: 1.0 / n for k in dist}
    return {k: max(0.0, float(v)) / total for k, v in dist.items()}


def _entropy(dist: Dict[str, float]) -> float:
    if len(dist) <= 1:
        return 0.0
    h = 0.0
    for p in dist.values():
        if p > 0:
            h -= p * math.log(p)
    return h / math.log(len(dist))


def _js_divergence(p: Dict[str, float], q: Dict[str, float]) -> float:
    keys = set(p) | set(q)
    if not keys:
        return 0.0
    pp = _normalize({k: p.get(k, 0.0) for k in keys})
    qq = _normalize({k: q.get(k, 0.0) for k in keys})
    m = {k: 0.5 * (pp.get(k, 0.0) + qq.get(k, 0.0)) for k in keys}

    def _kl(a: Dict[str, float], b: Dict[str, float]) -> float:
        val = 0.0
        for k in keys:
            av = a.get(k, 0.0)
            bv = b.get(k, 0.0)
            if av > 0.0 and bv > 0.0:
                val += av * math.log(av / bv)
        return val

    return _clip(0.5 * _kl(pp, m) + 0.5 * _kl(qq, m))


def _align_belief(belief: Dict[str, float], candidates: Iterable[str], smooth: float) -> Dict[str, float]:
    aligned = {}
    for c in candidates:
        aligned[c] = belief.get(c, smooth)
    return _normalize(aligned)


def _belief_gap(belief: Dict[str, float]) -> float:
    if not belief:
        return 0.0
    vals = sorted(belief.values(), reverse=True)
    if len(vals) == 1:
        return 1.0
    return _clip(vals[0] - vals[1])


def _copy_links(links: Optional[Dict[str, set]]) -> Dict[str, set]:
    return {node: set(neighbors) for node, neighbors in (links or {}).items()}


def _add_topology_link(links: Dict[str, set], source: str, target: str) -> None:
    source = _candidate_key(source)
    target = _candidate_key(target)
    if not source or not target or source == target:
        return
    links.setdefault(source, set()).add(target)
    links.setdefault(target, set()).add(source)


def _topology_links_from_step(
    prev_links: Optional[Dict[str, set]],
    call: Dict[str, Any],
    result: Dict[str, Any],
) -> Dict[str, set]:
    links = _copy_links(prev_links)
    name = _norm(call.get("name"))
    if name == "get_node_children":
        parent = result.get("parent_node") or _tool_arg(call, "node_id")
        for item in result.get("children", []) or []:
            _add_topology_link(links, parent, item.get("node_id"))
    elif name in {"get_upstream_nodes", "get_downstream_nodes"}:
        anchor = (
            result.get("target_node")
            or result.get("source_node")
            or _tool_arg(call, "node_id")
        )
        key = "upstream" if name == "get_upstream_nodes" else "downstream"
        for item in result.get(key, []) or []:
            _add_topology_link(links, anchor, item.get("node_id"))
    elif name == "get_related_systems":
        for key in ("upstream_connections", "downstream_connections"):
            for item in result.get(key, []) or []:
                _add_topology_link(
                    links,
                    item.get("connected_via"),
                    item.get("target_component"),
                )
    return links


def _topology_distance(
    source: str,
    target: str,
    links: Dict[str, set],
    max_depth: int = 2,
) -> Optional[int]:
    source = _candidate_key(source)
    target = _candidate_key(target)
    if not source or not target:
        return None
    if source == target:
        return 0
    frontier = {source}
    seen = {source}
    for depth in range(1, max_depth + 1):
        next_frontier = set()
        for node in frontier:
            for neighbor in links.get(node, set()):
                if neighbor == target:
                    return depth
                if neighbor not in seen:
                    seen.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break
    return None


def _topology_compatibility(
    candidate: str,
    evidence_node: str,
    links: Dict[str, set],
    config: Dict[str, Any],
) -> float:
    if not candidate or not evidence_node:
        return 0.0
    if candidate == evidence_node:
        return 1.0
    distance = _topology_distance(candidate, evidence_node, links)
    if distance == 1:
        return float(config["one_hop_projection"])
    if distance == 2:
        return float(config["two_hop_projection"])
    cand_system = node_system(candidate)
    evidence_system = node_system(evidence_node)
    if cand_system and cand_system == evidence_system:
        return float(config["same_system_projection"])
    return 0.0


def _project_evidence(
    candidates: List[str],
    diagnosed_node: str,
    result: Dict[str, Any],
    smooth: float,
    links: Dict[str, set],
    config: Dict[str, Any],
) -> Dict[str, float]:
    """Project local node evidence to candidate roots with topology compatibility."""
    if not candidates:
        return {}
    status = _status(result)
    confidence = _confidence(result)
    support = {c: smooth for c in candidates}
    node = _candidate_key(diagnosed_node)
    if not node:
        return _normalize(support)

    if status in ABNORMAL_STATUSES:
        severity = {"fault": 1.0, "warning": 0.72, "abnormal": 0.55, "alert": 1.0}.get(status, 0.5)
        base = max(0.05, confidence) * severity
        for candidate in candidates:
            compatibility = _topology_compatibility(candidate, node, links, config)
            if compatibility > 0.0:
                support[candidate] += base * compatibility
    elif status in NORMAL_STATUSES:
        for candidate in candidates:
            if candidate == node:
                support[candidate] += smooth
                continue
            compatibility = _topology_compatibility(candidate, node, links, config)
            support[candidate] += float(config["normal_alternative_support"]) * (
                1.0 - 0.5 * compatibility
            )
    else:
        if node in support:
            support[node] += 0.05
    return _normalize(support)


def _is_no_fault_gt(ground_truth: Dict[str, Any]) -> bool:
    return (
        is_no_fault_label(ground_truth.get("fault_type"))
        or is_no_fault_node(ground_truth.get("root_cause_node"))
    )


def _diagnosis_says_no_fault(final_diagnosis: Optional[Dict[str, Any]]) -> bool:
    if not final_diagnosis:
        return False
    text = (
        _norm(final_diagnosis.get("status"))
        + " "
        + _norm(final_diagnosis.get("fault_type"))
        + " "
        + _norm(final_diagnosis.get("root_cause_node"))
    )
    return any(token in text for token in ("normal", "no_fault", "no fault", "none"))


def _final_correct(final_diagnosis: Optional[Dict[str, Any]], ground_truth: Dict[str, Any]) -> bool:
    if not final_diagnosis:
        return False
    if _is_no_fault_gt(ground_truth):
        return _diagnosis_says_no_fault(final_diagnosis)
    gt_node = _norm(ground_truth.get("root_cause_node"))
    gt_fault = ground_truth.get("fault_type", "")
    diag_node = _norm(final_diagnosis.get("root_cause_node"))
    diag_fault = final_diagnosis.get("fault_type", "")
    return bool(gt_node and gt_node == diag_node and fault_exact_match(gt_fault, diag_fault))


@dataclass
class EPOTraceState:
    candidates: List[str]
    belief: Dict[str, float]
    support: Dict[str, float]
    mode: str
    potential: float
    closure: float
    conflict: float
    topology_links: Dict[str, set]


def _initial_state(config: Dict[str, Any]) -> EPOTraceState:
    return EPOTraceState(
        candidates=[],
        belief={},
        support={},
        mode="open",
        potential=float(config["closed_threshold"]),
        closure=0.0,
        conflict=0.0,
        topology_links={},
    )


def _transition_mode(mode: str, reliability: float, conflict: float, closure: float, config: Dict[str, Any]) -> str:
    """Evidence state-machine transition Γ (eq:epo_transition_cases)."""
    if not config.get("use_state_machine", True):
        return EVIDENCE_OPEN
    if reliability >= config["reliability_threshold"] and conflict >= config["violation_threshold"]:
        return EVIDENCE_CONFLICT
    if mode != EVIDENCE_CONFLICT and closure >= config["closed_threshold"]:
        return EVIDENCE_CLOSED
    if mode == EVIDENCE_CLOSED and closure < config["open_threshold"]:
        return EVIDENCE_OPEN
    if mode == EVIDENCE_CONFLICT and conflict <= config["recover_threshold"]:
        return EVIDENCE_OPEN
    return mode


def _potential(mode: str, closure: float, config: Dict[str, Any]) -> float:
    """Evidence potential V (eq:epo_potential): [δ_c - κ]_+ + λ_m·1[conflict]."""
    return (
        max(0.0, config["closed_threshold"] - closure)
        + config["lambda_violation"] * (1.0 if mode == EVIDENCE_CONFLICT else 0.0)
    )


def _add_candidate(candidates: List[str], node_id: str) -> List[str]:
    node_id = _candidate_key(node_id)
    if not node_id:
        return candidates
    if node_id not in candidates:
        candidates.append(node_id)
    return candidates


def _local_support(
    call: Dict[str, Any],
    result: Dict[str, Any],
    diagnosed_node: str,
) -> Tuple[str, float]:
    """Return positive same-node diagnostic support contributed by a step."""
    name = _norm(call.get("name"))
    node = _candidate_key(
        diagnosed_node
        or result.get("node_id")
        or result.get("component_node")
        or _tool_arg(call, "node_id")
    )
    if not node:
        return "", 0.0

    if name == "diagnose_node":
        status = _status(result)
        if status in ABNORMAL_STATUSES:
            severity = {
                "fault": 1.0,
                "warning": 0.62,
                "abnormal": 0.45,
                "alert": 1.0,
            }.get(status, 0.4)
            return node, _clip(observation_reliability(result) * severity)
        return node, 0.0

    if name == "get_node_sensors" and result.get("readings_available") is True:
        return node, 0.25

    return node, 0.0


def _update_candidates(
    candidates: List[str],
    call: Dict[str, Any],
    result: Dict[str, Any],
) -> Tuple[List[str], str]:
    """Deterministic candidate update from visible action and observation."""
    updated = list(candidates)
    name = _norm(call.get("name"))
    diagnosed_node = ""

    if name == "diagnose_node":
        diagnosed_node = result.get("node_id") or _tool_arg(call, "node_id")
        status = _status(result)
        if status in ABNORMAL_STATUSES:
            updated = _add_candidate(updated, diagnosed_node)
        elif not updated and diagnosed_node:
            updated = _add_candidate(updated, diagnosed_node)

    elif name == "get_node_sensors":
        diagnosed_node = result.get("component_node") or _tool_arg(call, "node_id")
        updated = _add_candidate(updated, diagnosed_node) if not updated else updated

    elif name in {"get_upstream_nodes", "get_downstream_nodes"}:
        key = "upstream" if name == "get_upstream_nodes" else "downstream"
        for item in result.get(key, []) or []:
            updated = _add_candidate(updated, item.get("node_id"))

    elif name == "get_related_systems":
        for key in ("upstream_connections", "downstream_connections"):
            for item in result.get(key, []) or []:
                updated = _add_candidate(updated, item.get("connected_via"))
                updated = _add_candidate(updated, item.get("target_component"))

    elif name == "get_node_children":
        for item in result.get("children", []) or []:
            updated = _add_candidate(updated, item.get("node_id"))

    return updated[:24], diagnosed_node


def _step_epo(
    prev: EPOTraceState,
    call: Dict[str, Any],
    result: Dict[str, Any],
    config: Dict[str, Any],
) -> Tuple[EPOTraceState, float, Dict[str, float]]:
    candidates, diagnosed_node = _update_candidates(prev.candidates, call, result)
    topology_links = _topology_links_from_step(prev.topology_links, call, result)
    smooth = float(config["smooth"])
    aligned = _align_belief(prev.belief, candidates, smooth) if candidates else {}
    support = {
        c: prev.support.get(c, 0.0)
        for c in candidates
    }

    reliability = observation_reliability(result) if isinstance(result, dict) else 0.0
    if _norm(call.get("name")) != "diagnose_node":
        reliability *= 0.35
    evidence = _project_evidence(
        candidates,
        diagnosed_node,
        result,
        smooth,
        topology_links,
        config,
    )
    if candidates and evidence:
        belief = _normalize({
            c: (aligned.get(c, smooth) ** (1.0 - reliability))
            * (evidence.get(c, smooth) ** reliability)
            for c in candidates
        })
    else:
        belief = aligned

    gap = _belief_gap(belief)
    entropy = _entropy(belief)
    conflict = reliability * _js_divergence(evidence, aligned) if evidence and aligned else 0.0
    support_node, support_value = _local_support(call, result, diagnosed_node)
    if support_node and support_node in support:
        support[support_node] = _clip(support.get(support_node, 0.0) + support_value)
    top_support = 0.0
    if belief:
        top_candidate = max(belief.items(), key=lambda item: item[1])[0]
        top_support = support.get(top_candidate, 0.0)
    # Closure score κ_t (eq:epo_closure_score): candidate gap, penalized by
    # reliability-weighted conflict and normalized belief entropy. The belief
    # gap g_t already reflects evidence support through the multiplicative
    # belief update, so no extra support multiplier is applied here.
    closure = (
        gap
        - config["lambda_chi"] * conflict
        - config["lambda_entropy"] * entropy
    )
    mode = _transition_mode(prev.mode, reliability, conflict, closure, config)
    potential = _potential(mode, closure, config)
    # Step shaping reward: λ_epo · (V_t - V_{t+1}) - λ_d · d_t  (eq:epo_reward,
    # non-terminal part). The terminal correctness term is added by the
    # trajectory/step aggregator, which knows the stop action and ground truth.
    lambda_epo = float(config.get("lambda_epo", 0.75))
    if config.get("use_potential_reward", True):
        shaping = lambda_epo * (prev.potential - potential) - float(config["tool_cost"])
    else:
        shaping = -float(config["tool_cost"])
    info = {
        "reliability": reliability,
        "support": top_support,
        "gap": gap,
        "entropy": entropy,
        "conflict": conflict,
        "closure": closure,
        "potential": potential,
        "potential_delta": prev.potential - potential,
        "mode": mode,
    }
    return EPOTraceState(
        candidates,
        belief,
        support,
        mode,
        potential,
        closure,
        conflict,
        topology_links,
    ), shaping, info


def compute_epo_step_rewards(
    agent_outputs: List[str],
    final_diagnosis: Optional[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    n_tool_calls: int,
    optimal_path_length: int = 3,
    max_steps: int = 15,
    tool_results: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute the paper-faithful per-step EPO reward sequence (eq:epo_reward).

    Unlike :func:`compute_epo_reward` (which squashes the trajectory into a
    single scalar for reward-mixing/eval), this returns the *unnormalized*
    step-level reward sequence ``r_t^epo`` aligned to each tool-call step, plus a
    terminal reward attached to the stop step. The RL trainer uses these to form
    discounted returns and per-step advantages, giving genuine step-level credit
    assignment for potential-decreasing actions.

    Returns a dict with:
      - ``step_rewards``: list[float], one shaping reward per executed tool call
      - ``terminal_reward``: float, added at the final (stop) step
      - ``potentials``: list[float] V_{t+1} after each step (telescoping check)
      - ``modes``: list[str] evidence mode after each step
      - ``final_mode``: str
      - ``correct``: bool final-diagnosis correctness
      - ``discount``: float xi
    """
    cfg = dict(DEFAULT_EPO_CONFIG)
    if config:
        cfg.update(config)

    calls = _extract_tool_calls(agent_outputs)
    results = _parse_tool_results(tool_results)
    state = _initial_state(cfg)
    initial_potential = state.potential

    step_rewards: List[float] = []
    potentials: List[float] = []
    modes: List[str] = []
    for idx, call in enumerate(calls):
        result = results[idx] if idx < len(results) else {}
        state, shaping, _info = _step_epo(state, call, result, cfg)
        step_rewards.append(float(shaping))
        potentials.append(float(state.potential))
        modes.append(state.mode)

    # Terminal reward (eq:epo_reward stop term): correctness, wrong penalty,
    # and open/conflict penalty. Not scaled by lambda_epo so the outcome signal
    # survives even at lambda_epo=0 (Outcome-PPO limit).
    correct = _final_correct(final_diagnosis, ground_truth)
    is_no_fault = _is_no_fault_gt(ground_truth)
    no_fault_coverage = (
        _no_fault_review_coverage(ground_truth, results, cfg) if is_no_fault else 0.0
    )
    no_fault_closed = (
        is_no_fault and no_fault_coverage >= float(cfg["no_fault_review_threshold"])
    )
    closed = (state.mode == EVIDENCE_CLOSED) or no_fault_closed

    terminal = 0.0
    stopped = final_diagnosis is not None
    if stopped:
        if correct and closed:
            terminal += float(cfg["terminal_correct"])
        elif correct:
            terminal += float(cfg["terminal_correct"]) - float(cfg["terminal_open"])
        else:
            terminal -= float(cfg["terminal_wrong"])
        if (not closed) and (not is_no_fault):
            terminal -= float(cfg["terminal_open"])
    elif n_tool_calls >= max_steps:
        terminal -= float(cfg["terminal_open"])

    # Telescoping diagnostic: Σ(V_t - V_{t+1}) == V_1 - V_{T+1}.
    telescope_lhs = sum(
        (potentials[i - 1] if i > 0 else initial_potential) - potentials[i]
        for i in range(len(potentials))
    )
    telescope_rhs = initial_potential - (potentials[-1] if potentials else initial_potential)

    return {
        "step_rewards": step_rewards,
        "terminal_reward": float(terminal),
        "potentials": potentials,
        "initial_potential": float(initial_potential),
        "modes": modes,
        "final_mode": state.mode,
        "correct": bool(correct),
        "closed": bool(closed),
        "no_fault_coverage": float(no_fault_coverage),
        "discount": float(cfg.get("discount", 0.95)),
        "telescope_lhs": round(float(telescope_lhs), 6),
        "telescope_rhs": round(float(telescope_rhs), 6),
        "n_steps": len(step_rewards),
    }


def compute_epo_reward(
    agent_outputs: List[str],
    final_diagnosis: Optional[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    n_tool_calls: int,
    optimal_path_length: int = 3,
    max_steps: int = 15,
    tool_results: Optional[List[str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Compute EPO trajectory reward and diagnostic evidence diagnostics."""
    cfg = dict(DEFAULT_EPO_CONFIG)
    if config:
        cfg.update(config)

    if not cfg.get("enabled", True):
        return {
            "epo": 0.0,
            "epo_raw": 0.0,
            "epo_closure": 0.0,
            "epo_max_closure": 0.0,
            "epo_violation_rate": 0.0,
            "epo_closed_rate": 0.0,
            "epo_final_potential": 0.0,
            "epo_conflict": 0.0,
            "epo_last_reliability": 0.0,
            "epo_last_support": 0.0,
            "epo_belief_top": 0.0,
            "epo_mode_closed": 0.0,
            "epo_mode_violated": 0.0,
            "epo_candidate_count": 0.0,
            "epo_top_candidate_is_gt": 0.0,
            "epo_no_fault_review_coverage": 0.0,
            "epo_no_fault_closed": 0.0,
        }

    calls = _extract_tool_calls(agent_outputs)
    results = _parse_tool_results(tool_results)
    state = _initial_state(cfg)
    total = 0.0
    closed_steps = 0
    violated_steps = 0
    max_closure = state.closure
    last_info: Dict[str, float] = {}

    for idx, call in enumerate(calls):
        result = results[idx] if idx < len(results) else {}
        state, step_reward, info = _step_epo(state, call, result, cfg)
        total += step_reward
        last_info = info
        if state.mode == EVIDENCE_CLOSED:
            closed_steps += 1
        if state.mode == EVIDENCE_CONFLICT:
            violated_steps += 1
        max_closure = max(max_closure, state.closure)

    stopped = final_diagnosis is not None
    correct = _final_correct(final_diagnosis, ground_truth)
    is_no_fault = _is_no_fault_gt(ground_truth)
    no_fault_coverage = (
        _no_fault_review_coverage(ground_truth, results, cfg)
        if is_no_fault else 0.0
    )
    no_fault_closed = (
        is_no_fault
        and no_fault_coverage >= float(cfg["no_fault_review_threshold"])
    )

    if stopped:
        if correct and (state.mode == EVIDENCE_CLOSED or no_fault_closed):
            total += float(cfg["terminal_correct"])
        elif correct:
            total += float(cfg["terminal_correct"]) - float(cfg["terminal_open"])
        else:
            total -= float(cfg["terminal_wrong"])
        if state.mode != EVIDENCE_CLOSED and not is_no_fault:
            total -= float(cfg["terminal_open"])
    elif n_tool_calls >= max_steps:
        total -= float(cfg["terminal_open"])

    efficiency = min(1.0, max(1, int(optimal_path_length or 3)) / max(1, int(n_tool_calls or 0)))
    closure_score = _clip((state.closure + 1.0) / 2.0)
    violation_rate = violated_steps / max(len(calls), 1)
    closed_rate = closed_steps / max(len(calls), 1)

    # Normalize to a stable range for mixture with the existing reward.
    normalized = _clip(0.5 + 0.25 * total, 0.0, 1.0)
    if violation_rate > 0:
        normalized = max(0.0, normalized - 0.25 * violation_rate)
    if stopped and not correct:
        normalized = max(0.0, normalized - 0.20)
    if correct:
        normalized = min(1.0, normalized + 0.10 * efficiency)

    top_candidate = ""
    top_belief = 0.0
    if state.belief:
        top_candidate, top_belief = max(state.belief.items(), key=lambda item: item[1])

    return {
        "epo": round(normalized, 4),
        "epo_raw": round(total, 4),
        "epo_closure": round(closure_score, 4),
        "epo_max_closure": round(_clip((max_closure + 1.0) / 2.0), 4),
        "epo_violation_rate": round(violation_rate, 4),
        "epo_closed_rate": round(closed_rate, 4),
        "epo_final_potential": round(state.potential, 4),
        "epo_conflict": round(state.conflict, 4),
        "epo_last_reliability": round(last_info.get("reliability", 0.0), 4),
        "epo_last_support": round(last_info.get("support", 0.0), 4),
        "epo_belief_top": round(float(top_belief), 4),
        "epo_mode_closed": 1.0 if state.mode == "closed" else 0.0,
        "epo_mode_violated": 1.0 if state.mode == EVIDENCE_CONFLICT else 0.0,
        "epo_candidate_count": float(len(state.candidates)),
        "epo_top_candidate_is_gt": (
            1.0
            if top_candidate and _norm(top_candidate) == _norm(ground_truth.get("root_cause_node"))
            else 0.0
        ),
        "epo_no_fault_review_coverage": round(no_fault_coverage, 4),
        "epo_no_fault_closed": 1.0 if no_fault_closed else 0.0,
    }
