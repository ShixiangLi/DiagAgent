"""
Trajectory Generator — Path-guided diagnostic trajectory generation.

Generates multi-turn diagnostic trajectories where the agent follows a
pre-designed diagnostic path from symptom to root cause. The Oracle returns
different responses based on whether the queried node is on the path:

  - Off-path node  → "Normal"  (agent should switch direction)
  - Path symptom   → "Abnormal" + hint (agent should investigate upstream)
  - Path intermediate → "Abnormal" + hint (agent should continue)
  - Path root cause → "Fault" + type (agent confirms diagnosis)
"""

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.data_gen.reasoning_composer import (
    compose_initial_reasoning, compose_system_selection, compose_diagnose_request,
    compose_normal_response, compose_abnormal_response, compose_fault_found,
    compose_final_diagnosis, compose_no_fault_conclusion, compose_upstream_trace,
    compose_wrong_system, compose_wrong_system_ack, compose_upstream_ack,
    compose_cross_system_transition, _format_sensor_evidence,
    compose_anomaly_score_selection, compose_multi_system_ranking,
    compose_system_elimination, compose_system_pivot,
    compose_warning_response, compose_sensor_verification,
    compose_status_summary_request, compose_status_summary_ack,
    compose_related_systems_request, compose_related_systems_ack,
    compose_related_systems_impact_request, compose_related_systems_impact_ack,
)
from src.environment.diagnostic_path import DiagnosticPath, PathNode
from src.environment.fault_scenario import FaultScenario
from src.environment.tool_executor import UnifiedToolExecutor
from src.utils.io_utils import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class TrajectoryStep:
    """A single step in a diagnostic trajectory."""
    role: str                      # "user", "assistant", "tool"
    content: str                   # Full content (may include <think>, <tool_call>)
    tool_call: Optional[Dict] = None
    tool_name: Optional[str] = None
    thought: Optional[str] = None


@dataclass
class DiagnosticTrajectory:
    """A complete diagnostic trajectory."""
    scenario_id: str
    scenario_type: str
    steps: List[TrajectoryStep]
    ground_truth: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


# Reasoning templates now provided by src.data_gen.reasoning_composer
# for high-diversity compositional generation with evidence citation.


# ============================================================================
# Diagnosis result synthesis (path-guided)
# ============================================================================

def _synthesize_path_result(
    scenario: FaultScenario,
    node_id: str,
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Synthesize a diagnose_node result based on the diagnostic path.

    Returns different results depending on the node's role in the path:
      - Off-path → Normal
      - Symptom/Intermediate → Abnormal with hint
      - Root cause → Fault with type
    """
    path = scenario.diagnostic_path
    if path is None:
        # No path available — fallback to Normal
        return {
            "status": "Normal",
            "fault_type": "None",
            "confidence": round(rng.uniform(0.82, 0.95), 4),
            "node_id": node_id,
        }

    path_node = path.get_path_node(node_id)

    if path_node is None:
        # Off-path: return Normal
        return {
            "status": "Normal",
            "fault_type": "None",
            "confidence": round(rng.uniform(0.82, 0.95), 4),
            "node_id": node_id,
            "node_name": node_id.split("::")[-1] if "::" in node_id else node_id,
        }

    if path_node.role == "root_cause":
        # Root cause: return definitive Fault
        return {
            "status": "Fault",
            "fault_type": scenario.fault_type,
            "confidence": round(rng.uniform(0.88, 0.97), 4),
            "node_id": node_id,
            "node_name": path_node.component_name,
            "system_id": path_node.system_id,
        }

    # Symptom or intermediate: return Abnormal with hint
    return {
        "status": "Abnormal",
        "fault_type": "None",
        "confidence": round(rng.uniform(0.70, 0.90), 4),
        "node_id": node_id,
        "node_name": path_node.component_name,
        "system_id": path_node.system_id,
        "abnormal_indicators": path_node.abnormal_hint,
        "suggested_direction": path_node.direction,
    }


def _synthesize_sensor_readings(rng: random.Random, abnormal: bool = False) -> Dict[str, float]:
    """Synthesize plausible HVAC sensor readings."""
    sensors = {
        "supply_temp": (52.0, 58.0),
        "return_temp": (68.0, 76.0),
        "differential_pressure": (1.5, 3.5),
        "flow_rate": (150.0, 350.0),
        "power": (10.0, 80.0),
    }
    readings = {}
    for name, (lo, hi) in sensors.items():
        if abnormal and rng.random() < 0.4:
            shift = rng.choice([-1, 1]) * rng.uniform(0.15, 0.40) * (hi - lo)
            readings[name] = round(rng.uniform(lo, hi) + shift, 2)
        else:
            readings[name] = round(rng.uniform(lo, hi), 2)
    return readings


# ============================================================================
# System display names
# ============================================================================

SYSTEM_DISPLAY_NAMES = {
    "chiller_plant": "Chiller Plant",
    "boiler_plant": "Boiler Plant",
    "sdahu": "Single-Duct AHU",
    "ddahu": "Dual-Duct AHU",
    "rtu": "Rooftop Unit",
    "fcu": "Fan Coil Unit",
    "pfpu": "Parallel Fan Powered Unit",
    "sfpu": "Series Fan Powered Unit",
}

_DISPLAY_NAME_TO_SYSTEM = {v: k for k, v in SYSTEM_DISPLAY_NAMES.items()}


def _explicit_system_from_prompt(description: str) -> Optional[str]:
    """Extract a leading ``[System Name]`` prompt hint when present."""
    text = str(description or "").strip()
    if not text.startswith("[") or "]" not in text:
        return None
    display_name = text[1:text.index("]")].strip()
    return _DISPLAY_NAME_TO_SYSTEM.get(display_name)


def _ensure_node_discovered(
    node_id: str,
    discovered_nodes: set,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
    _queried_parents: set = None,
) -> List[TrajectoryStep]:
    """Generate get_node_children steps needed to reveal a target node."""
    if node_id in discovered_nodes:
        return []

    parts = node_id.split("::")
    if len(parts) < 2:
        return []

    system_id = parts[0]
    if _queried_parents is None:
        _queried_parents = set()

    steps = []
    parent_chain = _get_has_part_parent_chain(node_id, tool_executor)
    if parent_chain:
        for parent_id in parent_chain:
            if node_id in discovered_nodes:
                break
            if parent_id in _queried_parents:
                continue
            steps.extend(_query_children_step(
                parent_id, discovered_nodes, _queried_parents,
                tool_executor, rng,
            ))
        return steps

    # Fallback frontier: start from system root, drill down shallowly.
    system_root = f"system::{system_id}"
    if system_root in _queried_parents:
        # System root already queried — start from already-discovered nodes
        frontier = [n for n in discovered_nodes if n.startswith(f"{system_id}::")]
    else:
        frontier = [system_root]
    max_depth = 2

    for depth in range(max_depth):
        if node_id in discovered_nodes:
            break
        next_frontier = []
        for parent_id in frontier:
            if parent_id in _queried_parents:
                # Already queried this parent; skip
                continue
            before = set(discovered_nodes)
            steps.extend(_query_children_step(
                parent_id, discovered_nodes, _queried_parents,
                tool_executor, rng,
            ))
            next_frontier.extend(n for n in discovered_nodes - before)

            if node_id in discovered_nodes:
                break
        frontier = next_frontier

    return steps


def _query_children_step(
    parent_id: str,
    discovered_nodes: set,
    queried_parents: set,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
) -> List[TrajectoryStep]:
    """Generate one get_node_children call and update discovery bookkeeping."""
    queried_parents.add(parent_id)
    parent_name = parent_id.split("::")[-1]
    thought = compose_diagnose_request(
        f"sub-components of {parent_name}", rng,
    )
    tc = {"name": "get_node_children", "arguments": {"node_id": parent_id}}
    steps = [TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tc)}</tool_call>",
        tool_call=tc,
        thought=thought,
    )]
    children_result = tool_executor.execute("get_node_children", {"node_id": parent_id})
    steps.append(TrajectoryStep(
        role="tool",
        content=json.dumps(children_result, indent=2),
        tool_name="get_node_children",
    ))
    for child in children_result.get("children", []):
        discovered_nodes.add(child["node_id"])
    return steps


def _get_has_part_parent_chain(
    node_id: str,
    tool_executor: UnifiedToolExecutor,
) -> List[str]:
    """Return has_part ancestors that need querying to reveal node_id."""
    tb = getattr(getattr(tool_executor, "topo_executor", None), "tb", None)
    graph = getattr(tb, "graph", None)
    if graph is None or node_id not in graph:
        return []

    parents = []
    current = node_id
    seen = set()
    while current in graph and current not in seen:
        seen.add(current)
        parent = None
        for source, _, data in graph.in_edges(current, data=True):
            if data.get("relation") == "has_part":
                parent = source
                break
        if not parent or parent == "building":
            break
        parents.append(parent)
        if parent.startswith("system::"):
            break
        current = parent
    parents.reverse()
    return parents


# ============================================================================
# Trajectory generation — main entry point
# ============================================================================

def generate_trajectory(
    scenario: FaultScenario,
    tool_executor: UnifiedToolExecutor,
    rng: Optional[random.Random] = None,
    scenario_state=None,
) -> DiagnosticTrajectory:
    """
    Generate a complete path-guided diagnostic trajectory.

    When scenario_state is provided, diagnose_node calls use the real Oracle
    model with actual sensor data. Otherwise, falls back to synthesis.

    Complexity tiers (by tool_call count):
      - easy/no_fault:  3-4 tool calls
      - single_system:  6-8 tool calls
      - cross_system:  9-10 tool calls
    """
    if rng is None:
        rng = random.Random()

    # Set scenario state on the prediction executor for real Oracle predictions
    if scenario_state is not None:
        tool_executor.set_scenario_state(scenario_state)

    steps: List[TrajectoryStep] = []
    path = scenario.diagnostic_path
    difficulty = scenario.difficulty
    stype = scenario.scenario_type

    # ---- Type-specific trajectory dispatch (2×4 matrix) ----
    # Ambiguous types
    if stype in ("a_single_system", "ambiguous"):
        return _generate_ambiguous_trajectory(
            scenario, tool_executor, rng, scenario_state,
        )
    if stype == "a_cross_system":
        return _generate_ambiguous_trajectory(
            scenario, tool_executor, rng, scenario_state,
        )
    if stype in ("a_low_confidence", "na_low_confidence", "low_confidence"):
        return _generate_low_confidence_trajectory(
            scenario, tool_executor, rng, scenario_state,
        )
    # No-fault (Non-Ambiguous only)
    if stype in ("na_no_fault", "no_fault"):
        return _generate_no_fault_trajectory(
            scenario, tool_executor, rng, scenario_state,
        )

    explicit_system = _explicit_system_from_prompt(scenario.description)

    # Standard fault types: na_single_system, na_cross_system, single_system, cross_system
    is_cross = stype in ("na_cross_system", "cross_system")
    if is_cross:
        n_elimination = rng.randint(2, 3)
        do_wrong_system = True
        do_upstream_trace = True
        do_downstream_impact = True
    else:  # na_single_system, single_system
        n_elimination = rng.randint(2, 4)
        do_wrong_system = rng.random() < 0.4
        do_upstream_trace = rng.random() < 0.5
        do_downstream_impact = rng.random() < 0.7

    # Non-ambiguous prompts that name a system should begin there.  A wrong
    # pre-sweep teaches the model to ignore explicit user context.
    if explicit_system and stype.startswith("na_"):
        do_wrong_system = False

    # ================================================================
    # Phase 1: User describes symptoms
    # ================================================================
    steps.append(TrajectoryStep(role="user", content=scenario.description))

    # ================================================================
    # Phase 2: System overview (1 tool call)
    # ================================================================
    thought = compose_initial_reasoning(rng)
    tool_call = {"name": "get_system_overview", "arguments": {}}
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call, thought=thought,
    ))
    overview_result = tool_executor.execute("get_system_overview", {})
    steps.append(TrajectoryStep(
        role="tool", content=json.dumps(overview_result, indent=2),
        tool_name="get_system_overview",
    ))

    # ================================================================
    # Phase 3: Determine starting system
    # ================================================================
    if explicit_system:
        start_system = explicit_system
    elif path and path.nodes:
        start_system = path.nodes[0].system_id
    else:
        start_system = scenario.root_cause_system

    correct_sys = scenario.root_cause_system or start_system
    start_system_name = SYSTEM_DISPLAY_NAMES.get(start_system, start_system)

    # Use anomaly_score to justify system selection (matches overview data)
    overview_systems = overview_result.get("systems", [])
    sys_score = next(
        (s.get("anomaly_score", 0.0) for s in overview_systems if s.get("system_id") == start_system),
        0.0,
    )

    # ================================================================
    # Phase 3b: Optional wrong-system exploration (+2 tool calls)
    # ================================================================
    if do_wrong_system and path and path.nodes:
        root_system = scenario.root_cause_system
        all_systems = list(SYSTEM_DISPLAY_NAMES.keys())
        wrong_candidates = [s for s in all_systems if s != root_system and s != start_system]

        # Avoid exploring the thermally opposite plant first:
        # heating faults should not check chiller_plant first (confusing signal)
        # cooling faults should not check boiler_plant first
        ft_lower = scenario.fault_type.lower()
        is_heating = any(
            k in ft_lower
            for k in ("reheat", "heating", "hot_water", "boiler", "hwc", "hwl")
        )
        is_cooling = any(
            k in ft_lower
            for k in (
                "chiller", "cooling", "bypass", "coolingtower", "cond",
                "evap", "chwc", "overcharge", "undercharge",
                "filterrestriction",
            )
        )
        if is_heating:
            wrong_candidates = [s for s in wrong_candidates if s != "chiller_plant"] or wrong_candidates
        elif is_cooling:
            wrong_candidates = [s for s in wrong_candidates if s != "boiler_plant"] or wrong_candidates

        if wrong_candidates:
            wrong_sys = rng.choice(wrong_candidates)
            steps.extend(_generate_wrong_system_check(
                wrong_sys, tool_executor, rng
            ))

    # ================================================================
    # Phase 4: Enter starting system (+1 tool call)
    # Use anomaly_score reasoning for NA types too (consistent with overview)
    # ================================================================
    if sys_score > 0:
        thought = compose_anomaly_score_selection(start_system_name, start_system, sys_score, rng)
    else:
        thought = compose_system_selection(start_system_name, start_system, rng)
    tool_call = {"name": "get_node_children", "arguments": {"node_id": f"system::{start_system}"}}
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call, thought=thought,
    ))
    children_result = tool_executor.execute("get_node_children", {"node_id": f"system::{start_system}"})
    steps.append(TrajectoryStep(
        role="tool", content=json.dumps(children_result, indent=2),
        tool_name="get_node_children",
    ))

    # Track which nodes have been discovered via get_node_children
    discovered_nodes = set()
    queried_parents = {f"system::{start_system}"}  # Already queried
    related_systems_queried = set()
    for child in children_result.get("children", []):
        discovered_nodes.add(child["node_id"])

    summary_steps, _summary_result = _generate_status_summary_step(
        start_system, tool_executor, rng,
    )
    steps.extend(summary_steps)
    # Once a summary has narrowed candidates, keep only a small amount of
    # off-path verification. Long blind sweeps are reserved for older baselines.
    n_elimination = min(n_elimination, 1)

    # ================================================================
    # Phase 5: No-fault scenarios
    # ================================================================
    if scenario.fault_type in ("Normal", "None") or path is None:
        steps.extend(_generate_no_fault_investigation(
            scenario, start_system, tool_executor, rng
        ))
        return _build_trajectory(scenario, steps, is_correct=True)

    # ================================================================
    # Phase 6: Elimination steps — check wrong nodes first (+N tool calls)
    # ================================================================
    steps.extend(_generate_elimination_steps(
        scenario, start_system, path, tool_executor, rng, n_elimination
    ))

    # ================================================================
    # Phase 7: Follow diagnostic path (core investigation)
    # ================================================================
    current_system = start_system
    path_nodes_to_follow = path.nodes
    if explicit_system == correct_sys:
        root_system_nodes = [p for p in path.nodes if p.system_id == correct_sys]
        if root_system_nodes:
            path_nodes_to_follow = root_system_nodes

    for i, path_node in enumerate(path_nodes_to_follow):
        node_id = path_node.node_id
        node_name = path_node.component_name
        node_system = path_node.system_id

        # Cross-system transition with explicit upstream trace
        if node_system != current_system:
            steps.extend(_generate_related_systems_step(
                current_system, node_system, tool_executor, rng,
            ))
            related_systems_queried.add(current_system)
            if do_upstream_trace:
                steps.extend(_generate_upstream_trace(
                    current_system, node_system, tool_executor, rng
                ))
            steps.extend(_generate_system_transition(
                current_system, node_system, node_id, tool_executor, rng,
                discovered_nodes=discovered_nodes,
                queried_parents=queried_parents,
            ))
            current_system = node_system

            summary_steps, _summary_result = _generate_status_summary_step(
                current_system, tool_executor, rng,
            )
            steps.extend(summary_steps)

            # Add 1 elimination step in the new system
            if rng.random() < 0.5:
                steps.extend(_generate_elimination_steps(
                    scenario, current_system, path, tool_executor, rng, 1
                ))

        # Ensure node is discovered before diagnosing it
        discovery_steps = _ensure_node_discovered(
            node_id, discovered_nodes, tool_executor, rng,
            _queried_parents=queried_parents,
        )
        steps.extend(discovery_steps)

        # Diagnose this path node (+1 tool call)
        thought = compose_diagnose_request(node_name, rng)
        tool_call = {"name": "diagnose_node", "arguments": {"node_id": node_id}}
        steps.append(TrajectoryStep(
            role="assistant",
            content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
            tool_call=tool_call, thought=thought,
        ))

        # Use real Oracle when scenario_state is available, else synthesize
        if scenario_state is not None:
            diag_result = tool_executor.execute("diagnose_node", {"node_id": node_id})
        else:
            diag_result = _synthesize_path_result(scenario, node_id, rng)
            diag_result["sensor_readings"] = _synthesize_sensor_readings(
                rng, abnormal=(path_node.role != "root_cause")
            )
        steps.append(TrajectoryStep(
            role="tool", content=json.dumps(diag_result, indent=2),
            tool_name="diagnose_node",
        ))

        status = diag_result["status"]

        if status == "Abnormal":
            hint = diag_result.get("abnormal_indicators", "")
            direction = diag_result.get("suggested_direction", "upstream")
            sensors = diag_result.get("sensor_readings", {})
            thought = compose_abnormal_response(node_name, hint, direction, sensors, rng)
            if direction == "upstream" and i < len(path.nodes) - 1:
                next_node = path.nodes[i + 1]
                if next_node.system_id != current_system:
                    trace_thought = compose_upstream_trace(node_name, rng)
                    thought = thought + "\n" + trace_thought
            steps.append(TrajectoryStep(
                role="assistant", content=f"<think>{thought}</think>", thought=thought,
            ))

        elif status == "Fault":
            fault_type = diag_result.get("fault_type", "unknown")
            confidence = diag_result.get("confidence", 0.90)
            sensors = diag_result.get("sensor_readings", {})
            thought = compose_fault_found(node_name, fault_type, confidence, sensors, rng)

            # Cross-system faults need an explicit system-relationship query so
            # the agent learns to justify affected-system scope with topology.
            if is_cross and current_system not in related_systems_queried:
                steps.extend(_generate_related_systems_impact_step(
                    current_system, tool_executor, rng,
                ))
                related_systems_queried.add(current_system)

            # Phase 8: Downstream impact verification (+1 tool call)
            if do_downstream_impact:
                ds_call = {"name": "get_downstream_nodes", "arguments": {"node_id": node_id}}
                ds_thought = f"Let me verify the downstream impact of this fault at {node_name}."
                steps.append(TrajectoryStep(
                    role="assistant",
                    content=f"<think>{thought}\n{ds_thought}</think>\n<tool_call>{json.dumps(ds_call)}</tool_call>",
                    tool_call=ds_call, thought=thought + "\n" + ds_thought,
                ))
                ds_result = tool_executor.execute("get_downstream_nodes", {"node_id": node_id})
                steps.append(TrajectoryStep(
                    role="tool", content=json.dumps(ds_result, indent=2),
                    tool_name="get_downstream_nodes",
                ))

            # Phase 9: Final diagnosis with sensor evidence
            affected = ", ".join(scenario.affected_systems)
            sensor_evidence = _format_sensor_evidence(sensors, rng, max_sensors=3, fault_type=fault_type)
            final_thought = compose_final_diagnosis(
                node_id, fault_type, affected, sensor_evidence, rng,
            )

            # Deduplicate: avoid repeating sensor evidence in both think + summary
            if sensor_evidence and sensor_evidence in thought and sensor_evidence in final_thought:
                final_thought = final_thought.replace(f" Critical evidence: {sensor_evidence}.", "")
                final_thought = final_thought.replace(f" Key diagnostic indicators: {sensor_evidence}.", "")
                final_thought = final_thought.replace(f" The diagnosis is grounded in: {sensor_evidence}.", "")

            diagnosis = {
                "root_cause_node": node_id,
                "fault_type": fault_type,
                "confidence": confidence,
                "affected_systems": scenario.affected_systems,
            }
            diag_json = json.dumps(diagnosis)
            steps.append(TrajectoryStep(
                role="assistant",
                content=f"<think>{thought}</think>\n\n{final_thought}\n\n<diagnosis>{diag_json}</diagnosis>",
                thought=thought + "\n" + final_thought,
            ))
            break

    return _build_trajectory(scenario, steps, is_correct=True)


# ============================================================================
# Helper generators
# ============================================================================

def _generate_no_fault_trajectory(
    scenario: FaultScenario,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
    scenario_state=None,
) -> DiagnosticTrajectory:
    """
    Generate no-fault trajectory: full scan of ALL nodes in the system.

    Two variants:
      A) Single-system full scan: check every node in the target system
      B) Multi-system scan: check 2 related systems completely
    """
    if scenario_state is not None:
        tool_executor.set_scenario_state(scenario_state)

    steps: List[TrajectoryStep] = []
    system_id = scenario.root_cause_system

    # Determine if this is a multi-system check
    multi_sys = getattr(scenario, '_no_fault_second_system', None)
    systems_to_check = [system_id]
    if multi_sys:
        systems_to_check.append(multi_sys)

    # Phase 1: User prompt
    steps.append(TrajectoryStep(role="user", content=scenario.description))

    # Phase 2: System overview
    thought = compose_initial_reasoning(rng)
    tool_call = {"name": "get_system_overview", "arguments": {}}
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call, thought=thought,
    ))
    overview_result = tool_executor.execute("get_system_overview", {})
    steps.append(TrajectoryStep(
        role="tool", content=json.dumps(overview_result, indent=2),
        tool_name="get_system_overview",
    ))

    all_checked = []

    for sys_idx, sys_id in enumerate(systems_to_check):
        sys_name = SYSTEM_DISPLAY_NAMES.get(sys_id, sys_id)

        # Extract real anomaly score from overview
        overview_systems = overview_result.get("systems", [])
        real_score = next(
            (s.get("anomaly_score", 0.0) for s in overview_systems if s.get("system_id") == sys_id),
            0.0,
        )

        # Phase 3: Select system to check — use real anomaly score in reasoning
        if sys_idx == 0:
            thought = compose_anomaly_score_selection(sys_name, sys_id, real_score, rng)
        else:
            thought = compose_system_pivot(
                SYSTEM_DISPLAY_NAMES.get(systems_to_check[0], systems_to_check[0]),
                sys_name, real_score, rng,
            )

        tool_call = {"name": "get_node_children", "arguments": {"node_id": f"system::{sys_id}"}}
        steps.append(TrajectoryStep(
            role="assistant",
            content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
            tool_call=tool_call, thought=thought,
        ))
        children_result = tool_executor.execute("get_node_children", {"node_id": f"system::{sys_id}"})
        steps.append(TrajectoryStep(
            role="tool", content=json.dumps(children_result, indent=2),
            tool_name="get_node_children",
        ))

        # Phase 4: Use a calibrated summary, then verify a small
        # representative subset instead of teaching blind full-system sweeps.
        summary_steps, summary_result = _generate_status_summary_step(
            sys_id, tool_executor, rng,
        )
        steps.extend(summary_steps)

        children = children_result.get("children", [])
        candidate_ids = {
            n.get("node_id")
            for n in summary_result.get("candidate_nodes", [])
            if n.get("node_id")
        }
        representative = [c for c in children if c.get("node_id") in candidate_ids]
        remaining = [c for c in children if c.get("node_id") not in candidate_ids]
        if remaining:
            representative.extend(rng.sample(remaining, min(2, len(remaining))))

        for comp in representative:
            nid = comp["node_id"]
            nname = comp.get("name", nid.split("::")[-1])

            thought = compose_diagnose_request(nname, rng)
            tool_call = {"name": "diagnose_node", "arguments": {"node_id": nid}}
            steps.append(TrajectoryStep(
                role="assistant",
                content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
                tool_call=tool_call, thought=thought,
            ))

            result = tool_executor.execute("diagnose_node", {"node_id": nid})

            steps.append(TrajectoryStep(
                role="tool", content=json.dumps(result, indent=2),
                tool_name="diagnose_node",
            ))

            sensors = result.get("sensor_readings", {})
            ack = _compose_no_fault_normal_ack(
                nname, result.get("confidence", 0.90), sensors, rng,
            )
            steps.append(TrajectoryStep(
                role="assistant", content=f"<think>{ack}</think>", thought=ack,
            ))
            all_checked.append(nname)

        if not representative:
            all_checked.extend(
                c.get("name", c.get("node_id", ""))
                for c in children[:3]
            )

        # System elimination reasoning between multi-system checks
        if len(systems_to_check) > 1 and sys_idx < len(systems_to_check) - 1:
            thought = compose_system_elimination(sys_name, len(representative), rng)
            steps.append(TrajectoryStep(
                role="assistant", content=f"<think>{thought}</think>", thought=thought,
            ))

    # Phase 5: Final no-fault conclusion
    final = compose_no_fault_conclusion(all_checked, rng)
    diagnosis = {
        "status": "Normal",
        "root_cause_node": "none",
        "fault_type": "no_fault",
        "confidence": 0.92,
        "systems_checked": systems_to_check,
    }
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{final}</think>\n\n<diagnosis>{json.dumps(diagnosis)}</diagnosis>",
        thought=final,
    ))

    return _build_trajectory(scenario, steps, is_correct=True)


def _compose_no_fault_normal_ack(
    node_name: str,
    confidence: float,
    sensors: Dict[str, Any],
    rng: random.Random,
) -> str:
    """A normal-node acknowledgement that does not imply a hidden fault."""
    evidence = _format_sensor_evidence(sensors, rng, max_sensors=2)
    base_options = [
        f"{node_name} reports Normal status with {confidence:.1%} confidence.",
        f"{node_name} is operating within expected limits ({confidence:.1%} confidence).",
        f"The diagnostic check for {node_name} is clean at {confidence:.1%} confidence.",
    ]
    base = rng.choice(base_options)
    if evidence:
        return f"{base} Supporting readings: {evidence}. This supports the no-fault assessment."
    return f"{base} This supports the no-fault assessment."


def _generate_wrong_detour(
    scenario: FaultScenario,
    current_system: str,
    path: DiagnosticPath,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
) -> List[TrajectoryStep]:
    """
    Generate a wrong-direction detour for trajectory diversity.

    The agent checks a node NOT on the path, gets Normal, then corrects course.
    """
    steps = []

    # Pick a random component not on the path
    components = tool_executor.execute("get_node_children", {"node_id": f"system::{current_system}"})
    children = components.get("children", [])
    path_ids = set(path.node_ids)
    off_path = [c for c in children if c.get("node_id") not in path_ids]

    if not off_path:
        return steps

    wrong_node = rng.choice(off_path)
    node_id = wrong_node["node_id"]
    node_name = wrong_node.get("name", node_id.split("::")[-1])

    thought = compose_diagnose_request(node_name, rng)
    tool_call = {"name": "diagnose_node", "arguments": {"node_id": node_id}}
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call,
        thought=thought,
    ))

    result = {
        "status": "Normal",
        "fault_type": "None",
        "confidence": round(rng.uniform(0.85, 0.95), 4),
        "node_id": node_id,
        "node_name": node_name,
        "sensor_readings": _synthesize_sensor_readings(rng, abnormal=False),
    }
    steps.append(TrajectoryStep(
        role="tool",
        content=json.dumps(result, indent=2),
        tool_name="diagnose_node",
    ))

    sensors = result.get("sensor_readings", {})
    ack = compose_normal_response(node_name, result.get("confidence", 0.90), sensors, rng)
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{ack}</think>",
        thought=ack,
    ))
    return steps


def _generate_elimination_steps(
    scenario: FaultScenario,
    system_id: str,
    path: DiagnosticPath,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
    n_steps: int = 2,
) -> List[TrajectoryStep]:
    """
    Generate elimination steps: diagnose N off-path nodes (all return Normal).

    This forces the agent to rule out healthy components before finding the
    correct path, increasing trajectory realism and length.
    """
    steps = []

    components = tool_executor.execute("get_node_children", {"node_id": f"system::{system_id}"})
    children = components.get("children", [])
    path_ids = set(path.node_ids)
    off_path = [c for c in children if c.get("node_id") not in path_ids]

    if not off_path:
        return steps

    # Sample up to n_steps off-path nodes
    selected = rng.sample(off_path, min(n_steps, len(off_path)))

    for comp in selected:
        node_id = comp["node_id"]
        node_name = comp.get("name", node_id.split("::")[-1])

        thought = compose_diagnose_request(node_name, rng)
        tool_call = {"name": "diagnose_node", "arguments": {"node_id": node_id}}
        steps.append(TrajectoryStep(
            role="assistant",
            content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
            tool_call=tool_call, thought=thought,
        ))

        result = tool_executor.execute("diagnose_node", {"node_id": node_id})
        steps.append(TrajectoryStep(
            role="tool", content=json.dumps(result, indent=2),
            tool_name="diagnose_node",
        ))

        sensors = result.get("sensor_readings", {})
        ack = compose_normal_response(node_name, result.get("confidence", 0.90), sensors, rng)
        steps.append(TrajectoryStep(
            role="assistant", content=f"<think>{ack}</think>", thought=ack,
        ))

    return steps


def _generate_status_summary_step(
    system_id: str,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
) -> Tuple[List[TrajectoryStep], Dict[str, Any]]:
    """Generate a get_node_status_summary step and a short acknowledgement."""
    steps: List[TrajectoryStep] = []
    system_name = SYSTEM_DISPLAY_NAMES.get(system_id, system_id)
    thought = compose_status_summary_request(system_name, rng)
    tool_call = {
        "name": "get_node_status_summary",
        "arguments": {"system_id": system_id},
    }
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call,
        thought=thought,
    ))
    result = tool_executor.execute("get_node_status_summary", {"system_id": system_id})
    steps.append(TrajectoryStep(
        role="tool",
        content=json.dumps(result, indent=2),
        tool_name="get_node_status_summary",
    ))
    candidate_count = len(result.get("candidate_nodes", []))
    ack = compose_status_summary_ack(system_name, candidate_count, rng)
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{ack}</think>",
        thought=ack,
    ))
    return steps, result


def _generate_related_systems_step(
    from_system: str,
    to_system: str,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
) -> List[TrajectoryStep]:
    """Generate a get_related_systems step before a cross-system transition."""
    steps: List[TrajectoryStep] = []
    from_name = SYSTEM_DISPLAY_NAMES.get(from_system, from_system)
    to_name = SYSTEM_DISPLAY_NAMES.get(to_system, to_system)

    thought = compose_related_systems_request(from_name, rng)
    tool_call = {
        "name": "get_related_systems",
        "arguments": {"system_id": from_system},
    }
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call,
        thought=thought,
    ))
    result = tool_executor.execute("get_related_systems", {"system_id": from_system})
    steps.append(TrajectoryStep(
        role="tool",
        content=json.dumps(result, indent=2),
        tool_name="get_related_systems",
    ))
    ack = compose_related_systems_ack(from_name, to_name, rng)
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{ack}</think>",
        thought=ack,
    ))
    return steps


def _generate_related_systems_impact_step(
    system_id: str,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
) -> List[TrajectoryStep]:
    """Generate get_related_systems after a root fault to verify impact scope."""
    steps: List[TrajectoryStep] = []
    system_name = SYSTEM_DISPLAY_NAMES.get(system_id, system_id)

    thought = compose_related_systems_impact_request(system_name, rng)
    tool_call = {
        "name": "get_related_systems",
        "arguments": {"system_id": system_id},
    }
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call,
        thought=thought,
    ))
    result = tool_executor.execute("get_related_systems", {"system_id": system_id})
    steps.append(TrajectoryStep(
        role="tool",
        content=json.dumps(result, indent=2),
        tool_name="get_related_systems",
    ))

    upstream_count = len(result.get("upstream_connections", []))
    downstream_count = len(result.get("downstream_connections", []))
    ack = compose_related_systems_impact_ack(
        system_name, upstream_count, downstream_count, rng,
    )
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{ack}</think>",
        thought=ack,
    ))
    return steps


def _generate_wrong_system_check(
    wrong_system_id: str,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
) -> List[TrajectoryStep]:
    """
    Explore an unrelated system and diagnose one component (returns Normal).

    Adds +2 tool calls: get_node_children + diagnose_node.
    """
    steps = []
    sys_name = SYSTEM_DISPLAY_NAMES.get(wrong_system_id, wrong_system_id)

    # Get children of the wrong system
    thought = compose_wrong_system(sys_name, rng)
    tool_call = {"name": "get_node_children", "arguments": {"node_id": f"system::{wrong_system_id}"}}
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call, thought=thought,
    ))

    children_result = tool_executor.execute("get_node_children", {"node_id": f"system::{wrong_system_id}"})
    steps.append(TrajectoryStep(
        role="tool", content=json.dumps(children_result, indent=2),
        tool_name="get_node_children",
    ))

    # Diagnose one component
    children = children_result.get("children", [])
    if children:
        comp = rng.choice(children)
        node_id = comp["node_id"]
        node_name = comp.get("name", node_id.split("::")[-1])

        thought = compose_diagnose_request(node_name, rng)
        tool_call = {"name": "diagnose_node", "arguments": {"node_id": node_id}}
        steps.append(TrajectoryStep(
            role="assistant",
            content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
            tool_call=tool_call, thought=thought,
        ))

        result = tool_executor.execute("diagnose_node", {"node_id": node_id})
        steps.append(TrajectoryStep(
            role="tool", content=json.dumps(result, indent=2),
            tool_name="diagnose_node",
        ))

        actual_status = result.get("status", "Normal")
        confidence = result.get("confidence", 0.90)
        ack = compose_wrong_system_ack(node_name, sys_name, actual_status, confidence, rng)
        steps.append(TrajectoryStep(
            role="assistant", content=f"<think>{ack}</think>", thought=ack,
        ))

    return steps


def _generate_upstream_trace(
    from_system: str,
    to_system: str,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
) -> List[TrajectoryStep]:
    """
    Generate an explicit get_upstream_nodes call to trace cross-system links.

    Adds +1 tool call before transitioning to the upstream system.
    """
    steps = []
    from_name = SYSTEM_DISPLAY_NAMES.get(from_system, from_system)

    # Find a node in the from_system to trace upstream from
    components = tool_executor.execute("get_node_children", {"node_id": f"system::{from_system}"})
    children = components.get("children", [])
    if not children:
        return steps

    # Pick the first component (typically the main unit)
    trace_node = children[0]
    node_id = trace_node["node_id"]
    node_name = trace_node.get("name", node_id.split("::")[-1])

    thought = compose_upstream_trace(node_name, rng)
    tool_call = {"name": "get_upstream_nodes", "arguments": {"node_id": node_id}}
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call, thought=thought,
    ))

    upstream_result = tool_executor.execute("get_upstream_nodes", {"node_id": node_id})
    steps.append(TrajectoryStep(
        role="tool", content=json.dumps(upstream_result, indent=2),
        tool_name="get_upstream_nodes",
    ))

    upstream_list = upstream_result.get("upstream", [])
    to_name = SYSTEM_DISPLAY_NAMES.get(to_system, to_system)
    if upstream_list:
        # Upstream nodes found — acknowledge the connection
        ack = compose_upstream_ack(from_name, to_name, rng)
    else:
        # No direct upstream link returned — fall back to anomaly-score reasoning
        ack = rng.choice([
            f"No direct upstream link was returned for {node_name}, but the system anomaly scores strongly suggest {to_name} as the upstream source. Switching focus.",
            f"The topology query returned no explicit upstream for {node_name}. However, given the cross-system fault propagation pattern and high anomaly score on {to_name}, I should investigate there next.",
            f"While get_upstream_nodes returned empty for {node_name}, the building's thermal architecture indicates {to_name} feeds this system. Investigating.",
        ])
    steps.append(TrajectoryStep(
        role="assistant", content=f"<think>{ack}</think>", thought=ack,
    ))

    return steps


def _generate_system_transition(
    from_system: str,
    to_system: str,
    target_node: str,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
    discovered_nodes: set = None,
    queried_parents: set = None,
) -> List[TrajectoryStep]:
    """Generate steps for transitioning between systems (upstream trace).

    Updates discovered_nodes and queried_parents to prevent duplicate
    get_node_children calls downstream.
    """
    steps = []
    from_name = SYSTEM_DISPLAY_NAMES.get(from_system, from_system)
    to_name = SYSTEM_DISPLAY_NAMES.get(to_system, to_system)

    system_root = f"system::{to_system}"

    # Upstream trace reasoning
    thought = compose_cross_system_transition(from_name, to_name, rng)
    tool_call = {"name": "get_node_children", "arguments": {"node_id": system_root}}
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call,
        thought=thought,
    ))

    result = tool_executor.execute("get_node_children", {"node_id": system_root})
    steps.append(TrajectoryStep(
        role="tool",
        content=json.dumps(result, indent=2),
        tool_name="get_node_children",
    ))

    # Mark as queried so _ensure_node_discovered won't repeat this call
    if queried_parents is not None:
        queried_parents.add(system_root)
    if discovered_nodes is not None:
        for child in result.get("children", []):
            discovered_nodes.add(child["node_id"])

    return steps


# ============================================================================
# Type-specific trajectory generators
# ============================================================================

def _generate_ambiguous_trajectory(
    scenario: FaultScenario,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
    scenario_state=None,
) -> DiagnosticTrajectory:
    """
    Generate trajectory for ambiguous scenarios: multi-system exploration.

    Three variants (randomly selected):
      A (40%): Agent picks highest-scoring system directly → finds fault
      B (40%): Agent tries wrong system first → eliminates → pivots → finds fault
      C (20%): Agent tries two wrong systems → eliminates both → finds fault
    """
    if scenario_state is not None:
        tool_executor.set_scenario_state(scenario_state)

    steps: List[TrajectoryStep] = []
    path = scenario.diagnostic_path

    # Phase 1: User prompt
    steps.append(TrajectoryStep(role="user", content=scenario.description))

    # Phase 2: System overview
    thought = compose_initial_reasoning(rng)
    tool_call = {"name": "get_system_overview", "arguments": {}}
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call, thought=thought,
    ))
    overview_result = tool_executor.execute("get_system_overview", {})
    steps.append(TrajectoryStep(
        role="tool", content=json.dumps(overview_result, indent=2),
        tool_name="get_system_overview",
    ))

    # Parse anomaly scores from overview
    systems_info = overview_result.get("systems", [])
    scored = [
        (s.get("name", ""), s.get("system_id", ""), s.get("anomaly_score", 0.0))
        for s in systems_info
    ]
    scored.sort(key=lambda x: x[2], reverse=True)

    correct_sys = scenario.root_cause_system
    correct_name = SYSTEM_DISPLAY_NAMES.get(correct_sys, correct_sys)

    # Pick wrong systems (exclude correct), ordered by symptom relevance
    wrong_candidates = [(n, sid, sc) for n, sid, sc in scored if sid != correct_sys]

    # Reorder: put symptom-irrelevant systems last so they aren't explored first
    # e.g., for heating faults, chiller_plant and rtu are least relevant
    _HEATING_IRRELEVANT = {"chiller_plant", "rtu"}
    _COOLING_IRRELEVANT = {"boiler_plant"}
    fault_lower = scenario.fault_type.lower()
    prompt_lower = scenario.description.lower() if scenario.description else ""
    fault_is_heating = any(
        k in fault_lower
        for k in ("reheat", "heating", "hot_water", "boiler", "hwc", "hwl")
    )
    fault_is_cooling = any(
        k in fault_lower
        for k in (
            "cooling", "chiller", "chilled", "evap", "cond", "compressor",
            "bypass", "coolingtower", "chwc", "overcharge", "undercharge",
            "filterrestriction",
        )
    )
    prompt_is_heating = any(
        k in prompt_lower
        for k in ("reheat", "heating", "hot_water", "boiler")
    )
    prompt_is_cooling = any(
        k in prompt_lower
        for k in ("cooling", "chiller", "chilled", "evap", "cond", "compressor")
    )
    is_cooling = fault_is_cooling or (not fault_is_heating and prompt_is_cooling)
    is_heating = fault_is_heating or (not fault_is_cooling and prompt_is_heating)

    if is_cooling:
        # Push boiler_plant to the end for cooling-dominant faults.
        wrong_candidates.sort(
            key=lambda x: (1 if x[1] in _COOLING_IRRELEVANT else 0, -x[2])
        )
    elif is_heating:
        # Push chiller_plant/rtu to end
        wrong_candidates.sort(
            key=lambda x: (1 if x[1] in _HEATING_IRRELEVANT else 0, -x[2])
        )

    # Select variant
    roll = rng.random()
    if roll < 0.4:
        variant = "direct"
    elif roll < 0.8:
        variant = "one_wrong"
    else:
        variant = "two_wrong"

    # Phase 3: Anomaly-score ranking + system selection (merged into one think)
    # This avoids contradictory double-think blocks
    if variant == "direct":
        # Go directly to correct system — use anomaly score selection
        correct_score = next(
            (sc for _, sid, sc in scored if sid == correct_sys), 0.5
        )
        thought = compose_anomaly_score_selection(
            correct_name, correct_sys, correct_score, rng,
        )
    else:
        # Go to first wrong system — use simple system selection
        # (don't claim it has the 'highest anomaly' — that would be incorrect)
        wrong_sys_info = wrong_candidates[0] if wrong_candidates else (correct_name, correct_sys, 0.5)
        thought = compose_system_selection(
            wrong_sys_info[0], wrong_sys_info[1], rng,
        )

    first_sys = correct_sys if variant == "direct" else (wrong_candidates[0][1] if wrong_candidates else correct_sys)
    first_name = SYSTEM_DISPLAY_NAMES.get(first_sys, first_sys)

    # Enter first system
    tool_call = {"name": "get_node_children", "arguments": {"node_id": f"system::{first_sys}"}}
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call, thought=thought,
    ))
    children_result = tool_executor.execute("get_node_children", {"node_id": f"system::{first_sys}"})
    steps.append(TrajectoryStep(
        role="tool", content=json.dumps(children_result, indent=2),
        tool_name="get_node_children",
    ))

    if variant != "direct":
        # Check 1-2 components in wrong system → all Normal
        children = children_result.get("children", [])
        n_check = rng.randint(1, min(2, len(children))) if children else 0
        checked = rng.sample(children, n_check) if children else []

        for comp in checked:
            nid = comp["node_id"]
            nname = comp.get("name", nid.split("::")[-1])
            t = compose_diagnose_request(nname, rng)
            tc = {"name": "diagnose_node", "arguments": {"node_id": nid}}
            steps.append(TrajectoryStep(
                role="assistant",
                content=f"<think>{t}</think>\n<tool_call>{json.dumps(tc)}</tool_call>",
                tool_call=tc, thought=t,
            ))
            res = tool_executor.execute("diagnose_node", {"node_id": nid})
            steps.append(TrajectoryStep(
                role="tool", content=json.dumps(res, indent=2), tool_name="diagnose_node",
            ))
            sensors = res.get("sensor_readings", {})
            actual_status = res.get("status", "Normal")
            if actual_status == "Abnormal":
                hint = res.get("abnormal_indicators", "Operating parameters are outside normal ranges.")
                ack = compose_abnormal_response(nname, hint, "upstream", sensors, rng)
            else:
                ack = compose_normal_response(nname, res.get("confidence", 0.9), sensors, rng)
            # For the last checked component, merge ack + elimination reasoning
            if comp == checked[-1]:
                elim = compose_system_elimination(first_name, n_check, rng)
                combined = f"{ack}\n{elim}"
                steps.append(TrajectoryStep(
                    role="assistant", content=f"<think>{combined}</think>", thought=combined,
                ))
            else:
                steps.append(TrajectoryStep(
                    role="assistant", content=f"<think>{ack}</think>", thought=ack,
                ))

        # System elimination already merged above for last component
        if not checked:
            # Edge case: no components were checked
            thought = compose_system_elimination(first_name, n_check, rng)
            steps.append(TrajectoryStep(
                role="assistant", content=f"<think>{thought}</think>", thought=thought,
            ))

        # Variant C: try second wrong system too
        if variant == "two_wrong" and len(wrong_candidates) >= 2:
            second_wrong = wrong_candidates[1]
            thought = compose_system_pivot(
                first_name, second_wrong[0], second_wrong[2], rng,
            )
            tc = {"name": "get_node_children", "arguments": {"node_id": f"system::{second_wrong[1]}"}}
            steps.append(TrajectoryStep(
                role="assistant",
                content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tc)}</tool_call>",
                tool_call=tc, thought=thought,
            ))
            ch2 = tool_executor.execute("get_node_children", {"node_id": f"system::{second_wrong[1]}"})
            steps.append(TrajectoryStep(
                role="tool", content=json.dumps(ch2, indent=2), tool_name="get_node_children",
            ))
            ch2_list = ch2.get("children", [])
            if ch2_list:
                comp2 = rng.choice(ch2_list)
                nid2 = comp2["node_id"]
                nname2 = comp2.get("name", nid2.split("::")[-1])
                t2 = compose_diagnose_request(nname2, rng)
                tc2 = {"name": "diagnose_node", "arguments": {"node_id": nid2}}
                steps.append(TrajectoryStep(
                    role="assistant",
                    content=f"<think>{t2}</think>\n<tool_call>{json.dumps(tc2)}</tool_call>",
                    tool_call=tc2, thought=t2,
                ))
                res2 = tool_executor.execute("diagnose_node", {"node_id": nid2})
                steps.append(TrajectoryStep(
                    role="tool", content=json.dumps(res2, indent=2), tool_name="diagnose_node",
                ))
                actual_status2 = res2.get("status", "Normal")
                sensors2 = res2.get("sensor_readings", {})
                if actual_status2 == "Abnormal":
                    hint2 = res2.get("abnormal_indicators", "Operating parameters are outside normal ranges.")
                    ack2 = compose_abnormal_response(nname2, hint2, "upstream", sensors2, rng)
                else:
                    ack2 = compose_normal_response(nname2, res2.get("confidence", 0.9), sensors2, rng)
                # Merge ack + elimination for second wrong system
                elim2 = compose_system_elimination(second_wrong[0], 1, rng)
                combined2 = f"{ack2}\n{elim2}"
                steps.append(TrajectoryStep(
                    role="assistant", content=f"<think>{combined2}</think>", thought=combined2,
                ))
            else:
                thought = compose_system_elimination(second_wrong[0], 0, rng)
                steps.append(TrajectoryStep(
                    role="assistant", content=f"<think>{thought}</think>", thought=thought,
                ))

        # Pivot to correct system
        correct_score = next(
            (sc for _, sid, sc in scored if sid == correct_sys), 0.5
        )
        thought = compose_system_pivot(
            first_name if variant == "one_wrong" else wrong_candidates[1][0] if len(wrong_candidates) >= 2 else first_name,
            correct_name, correct_score, rng,
        )
        tc = {"name": "get_node_children", "arguments": {"node_id": f"system::{correct_sys}"}}
        steps.append(TrajectoryStep(
            role="assistant",
            content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tc)}</tool_call>",
            tool_call=tc, thought=thought,
        ))
        children_result = tool_executor.execute("get_node_children", {"node_id": f"system::{correct_sys}"})
        steps.append(TrajectoryStep(
            role="tool", content=json.dumps(children_result, indent=2),
            tool_name="get_node_children",
        ))

    # Track which nodes have been discovered via get_node_children
    discovered_nodes = set()
    queried_parents = {f"system::{correct_sys}"}  # Already queried
    for child in children_result.get("children", []):
        discovered_nodes.add(child["node_id"])

    # Now in correct system — follow path to root cause
    if path and path.nodes:
        for pnode in path.nodes:
            if pnode.system_id != correct_sys:
                continue
            nid = pnode.node_id
            nname = pnode.component_name

            # Ensure node is discovered before diagnosing it
            discovery_steps = _ensure_node_discovered(
                nid, discovered_nodes, tool_executor, rng,
                _queried_parents=queried_parents,
            )
            steps.extend(discovery_steps)

            t = compose_diagnose_request(nname, rng)
            tc = {"name": "diagnose_node", "arguments": {"node_id": nid}}
            steps.append(TrajectoryStep(
                role="assistant",
                content=f"<think>{t}</think>\n<tool_call>{json.dumps(tc)}</tool_call>",
                tool_call=tc, thought=t,
            ))
            if scenario_state is not None:
                res = tool_executor.execute("diagnose_node", {"node_id": nid})
            else:
                res = _synthesize_path_result(scenario, nid, rng)
                res["sensor_readings"] = _synthesize_sensor_readings(rng)
            steps.append(TrajectoryStep(
                role="tool", content=json.dumps(res, indent=2), tool_name="diagnose_node",
            ))

            status = res.get("status", "Normal")
            if status == "Fault":
                ft = res.get("fault_type", "unknown")
                conf = res.get("confidence", 0.9)
                sensors = res.get("sensor_readings", {})
                thought = compose_fault_found(nname, ft, conf, sensors, rng)
                if scenario.scenario_type == "a_cross_system":
                    steps.extend(_generate_related_systems_impact_step(
                        correct_sys, tool_executor, rng,
                    ))
                affected = ", ".join(scenario.affected_systems)
                se = _format_sensor_evidence(sensors, rng, max_sensors=3, fault_type=ft)
                final = compose_final_diagnosis(nid, ft, affected, se, rng)
                diag = {
                    "root_cause_node": nid, "fault_type": ft,
                    "confidence": conf, "affected_systems": scenario.affected_systems,
                }
                steps.append(TrajectoryStep(
                    role="assistant",
                    content=f"<think>{thought}</think>\n\n{final}\n\n<diagnosis>{json.dumps(diag)}</diagnosis>",
                    thought=thought + "\n" + final,
                ))
                break
            elif status == "Abnormal":
                hint = res.get("abnormal_indicators", "")
                sensors = res.get("sensor_readings", {})
                thought = compose_abnormal_response(nname, hint, "upstream", sensors, rng)
                steps.append(TrajectoryStep(
                    role="assistant", content=f"<think>{thought}</think>", thought=thought,
                ))
            elif status == "Normal":
                sensors = res.get("sensor_readings", {})
                ack = compose_normal_response(nname, res.get("confidence", 0.9), sensors, rng)
                steps.append(TrajectoryStep(
                    role="assistant", content=f"<think>{ack}</think>", thought=ack,
                ))

    return _build_trajectory(scenario, steps, is_correct=True)


def _generate_low_confidence_trajectory(
    scenario: FaultScenario,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
    scenario_state=None,
) -> DiagnosticTrajectory:
    """
    Generate trajectory for low-confidence scenarios: sensor cross-validation.

    Oracle returns Warning (not Fault) for root cause. Agent must:
    1. Navigate to correct system (prompt names it)
    2. Get Warning from Oracle
    3. Call get_node_sensors to check raw data
    4. Reason from sensor evidence to confirm diagnosis
    """
    if scenario_state is not None:
        tool_executor.set_scenario_state(scenario_state)

    steps: List[TrajectoryStep] = []
    path = scenario.diagnostic_path

    # Phase 1: User prompt
    steps.append(TrajectoryStep(role="user", content=scenario.description))

    # Phase 2: System overview
    thought = compose_initial_reasoning(rng)
    tool_call = {"name": "get_system_overview", "arguments": {}}
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call, thought=thought,
    ))
    overview_result = tool_executor.execute("get_system_overview", {})
    steps.append(TrajectoryStep(
        role="tool", content=json.dumps(overview_result, indent=2),
        tool_name="get_system_overview",
    ))

    # Phase 3: Select system (prompt already names it, anomaly score confirms)
    start_system = scenario.root_cause_system
    start_name = SYSTEM_DISPLAY_NAMES.get(start_system, start_system)

    systems_info = overview_result.get("systems", [])
    sys_score = next(
        (s.get("anomaly_score", 0.5) for s in systems_info if s.get("system_id") == start_system),
        0.5,
    )
    thought = compose_anomaly_score_selection(start_name, start_system, sys_score, rng)
    tool_call = {"name": "get_node_children", "arguments": {"node_id": f"system::{start_system}"}}
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call, thought=thought,
    ))
    children_result = tool_executor.execute("get_node_children", {"node_id": f"system::{start_system}"})
    steps.append(TrajectoryStep(
        role="tool", content=json.dumps(children_result, indent=2),
        tool_name="get_node_children",
    ))

    summary_steps, _summary_result = _generate_status_summary_step(
        start_system, tool_executor, rng,
    )
    steps.extend(summary_steps)

    # Phase 4: Optional elimination of 1-2 off-path nodes
    if path:
        steps.extend(_generate_elimination_steps(
            scenario, start_system, path, tool_executor, rng,
            n_steps=rng.randint(0, 1),
        ))

    # Phase 5: Diagnose root cause → get Warning (not Fault)
    root_node = scenario.root_cause_node
    root_name = root_node.split("::")[-1] if "::" in root_node else root_node

    # Ensure root node is discovered via get_node_children before diagnosing
    discovered_nodes = set()
    queried_parents = {f"system::{start_system}"}
    for child in children_result.get("children", []):
        discovered_nodes.add(child["node_id"])
    discovery_steps = _ensure_node_discovered(
        root_node, discovered_nodes, tool_executor, rng,
        _queried_parents=queried_parents,
    )
    steps.extend(discovery_steps)

    thought = compose_diagnose_request(root_name, rng)
    tool_call = {"name": "diagnose_node", "arguments": {"node_id": root_node}}
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call, thought=thought,
    ))

    if scenario_state is not None:
        diag_result = tool_executor.execute("diagnose_node", {"node_id": root_node})
    else:
        # Synthesize Warning response for SFT
        diag_result = {
            "status": "Warning",
            "fault_type": "uncertain",
            "confidence": round(rng.uniform(0.45, 0.60), 4),
            "node_id": root_node,
            "node_name": root_name,
            "message": "Borderline anomaly detected. Recommend sensor-level verification.",
            "sensor_readings": _synthesize_sensor_readings(rng, abnormal=True),
        }
    steps.append(TrajectoryStep(
        role="tool", content=json.dumps(diag_result, indent=2), tool_name="diagnose_node",
    ))

    # Phase 6: Warning reasoning → decide to check sensors
    sensors = diag_result.get("sensor_readings", {})
    thought = compose_warning_response(root_name, diag_result.get("confidence", 0.55), sensors, rng)
    # Call get_node_sensors
    tool_call = {"name": "get_node_sensors", "arguments": {"node_id": root_node}}
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call, thought=thought,
    ))
    sensors_result = tool_executor.execute("get_node_sensors", {"node_id": root_node})
    steps.append(TrajectoryStep(
        role="tool", content=json.dumps(sensors_result, indent=2),
        tool_name="get_node_sensors",
    ))

    # Phase 7: Sensor verification reasoning → confirm diagnosis
    sensor_evidence = _format_sensor_evidence(sensors, rng, max_sensors=3, fault_type=scenario.fault_type)
    if not sensor_evidence:
        sensor_evidence = "temperature and pressure readings deviate from normal operating baseline"

    thought = compose_sensor_verification(root_name, sensor_evidence, rng)
    ft = scenario.fault_type
    affected = ", ".join(scenario.affected_systems)
    final = compose_final_diagnosis(root_node, ft, affected, sensor_evidence, rng)

    # Deduplicate: if the final summary is too similar to the think block, shorten it
    if sensor_evidence in thought and sensor_evidence in final:
        # Avoid repeating the exact same evidence string in both think + summary
        final = final.replace(f" Critical evidence: {sensor_evidence}.", "")
        final = final.replace(f" Key diagnostic indicators: {sensor_evidence}.", "")
        final = final.replace(f" The diagnosis is grounded in: {sensor_evidence}.", "")

    diagnosis = {
        "root_cause_node": root_node,
        "fault_type": ft,
        # Sensor verification raises confidence above the Warning but keeps it
        # moderate; low-confidence scenarios should not jump to high certainty.
        "confidence": round(rng.uniform(0.64, 0.72), 4),
        "affected_systems": scenario.affected_systems,
    }
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n\n{final}\n\n<diagnosis>{json.dumps(diagnosis)}</diagnosis>",
        thought=thought + "\n" + final,
    ))

    return _build_trajectory(scenario, steps, is_correct=True)


# ============================================================================
# Trajectory finalization
# ============================================================================

def _build_trajectory(
    scenario: FaultScenario,
    steps: List[TrajectoryStep],
    is_correct: bool,
) -> DiagnosticTrajectory:
    """Build the final trajectory with metadata."""
    path = scenario.diagnostic_path
    final_diagnosis = _extract_final_diagnosis(steps)

    is_no_fault = (
        "no_fault" in str(scenario.scenario_type).lower()
        or str(scenario.root_cause_node).lower() in ("none", "")
        or str(scenario.fault_type).lower() in ("normal", "no_fault")
    )
    gt_root_node = "none" if is_no_fault else scenario.root_cause_node
    gt_fault_type = "no_fault" if is_no_fault else scenario.fault_type
    label_consistent = _diagnosis_matches_ground_truth(
        final_diagnosis, gt_root_node, gt_fault_type, is_no_fault,
    )

    return DiagnosticTrajectory(
        scenario_id=scenario.scenario_id,
        scenario_type=scenario.scenario_type,
        steps=steps,
        ground_truth={
            "root_cause_system": scenario.root_cause_system,
            "root_cause_node": gt_root_node,
            "fault_type": gt_fault_type,
            "fault_intensity": "none" if is_no_fault else scenario.fault_intensity,
        },
        metadata={
            "n_tool_calls": sum(1 for s in steps if s.tool_call is not None),
            "difficulty": scenario.difficulty,
            "path_length": path.path_length if path else 0,
            "trajectory_quality": "correct" if is_correct else "incorrect",
            "tool_faithful": True,
            "label_consistent": label_consistent,
            "final_diagnosis": final_diagnosis,
        },
    )


def _extract_final_diagnosis(steps: List[TrajectoryStep]) -> Dict[str, Any]:
    """Extract the final <diagnosis> JSON, if present."""
    import re

    for step in reversed(steps):
        if step.role != "assistant" or not step.content or "<diagnosis>" not in step.content:
            continue
        m = re.search(r"<diagnosis>(.*?)</diagnosis>", step.content, re.DOTALL)
        if not m:
            continue
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            return {}
    return {}


def _norm_label(value: Any) -> str:
    import re

    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _diagnosis_matches_ground_truth(
    diagnosis: Dict[str, Any],
    gt_root_node: str,
    gt_fault_type: str,
    is_no_fault: bool,
) -> bool:
    """Check whether the final diagnosis is semantically aligned with GT."""
    if not diagnosis:
        return False
    diag_root = str(diagnosis.get("root_cause_node", ""))
    diag_fault = _norm_label(diagnosis.get("fault_type", ""))
    gt_fault = _norm_label(gt_fault_type)

    if is_no_fault:
        status = _norm_label(diagnosis.get("status", ""))
        return (
            diag_root.lower() in ("none", "")
            and diag_fault in ("no_fault", "normal", "none")
            and status in ("normal", "")
        )

    return diag_root == gt_root_node and diag_fault == gt_fault


# ============================================================================
# Batch generation
# ============================================================================

def generate_batch_trajectories(
    scenarios: List[FaultScenario],
    tool_executor: UnifiedToolExecutor,
    scenario_states: Optional[Dict] = None,
    seed: int = 42,
) -> List[DiagnosticTrajectory]:
    """
    Generate trajectories for a batch of scenarios.

    Args:
        scenarios: List of fault scenarios.
        tool_executor: Unified tool executor.
        scenario_states: Optional dict of {scenario_id: FaultScenarioState}.
            When provided, real Oracle model predictions are used.
        seed: Random seed for reproducibility.

    Returns:
        List of generated trajectories.
    """
    rng = random.Random(seed)
    trajectories = []
    failed = 0
    skipped_inconsistent = 0

    for i, scenario in enumerate(scenarios):
        try:
            state = None
            if scenario_states:
                state = scenario_states.get(scenario.scenario_id)
            traj = generate_trajectory(
                scenario, tool_executor, rng, scenario_state=state,
            )
            if not traj.metadata.get("label_consistent", False):
                skipped_inconsistent += 1
                logger.warning(
                    "[%s] Skipping label-inconsistent trajectory: gt=%s final=%s",
                    scenario.scenario_id,
                    traj.ground_truth,
                    traj.metadata.get("final_diagnosis", {}),
                )
                continue
            trajectories.append(traj)
        except Exception as e:
            failed += 1
            logger.warning(f"[{scenario.scenario_id}] Generation failed: {e}")
            if failed > len(scenarios) * 0.1:
                logger.error("Too many failures, stopping batch generation")
                break

        if (i + 1) % 100 == 0 or (i + 1) == len(scenarios):
            logger.info(f"Generated {i + 1}/{len(scenarios)} trajectories")

    logger.info(
        "Generated %s trajectories total (failed: %s, skipped_inconsistent: %s)",
        len(trajectories), failed, skipped_inconsistent,
    )
    return trajectories
