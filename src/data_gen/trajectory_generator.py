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

    # Determine complexity parameters
    if scenario.scenario_type == "cross_system":
        n_elimination = rng.randint(2, 3)  # Wrong nodes to check
        do_wrong_system = True             # Explore a wrong system first
        do_upstream_trace = True           # Explicit upstream trace call
        do_downstream_impact = True        # Always verify downstream
    elif scenario.scenario_type in ("single_system", "ambiguous", "low_confidence"):
        n_elimination = rng.randint(2, 4)  # More elimination for depth
        do_wrong_system = rng.random() < 0.4
        do_upstream_trace = rng.random() < 0.5
        do_downstream_impact = rng.random() < 0.7
    else:  # no_fault
        n_elimination = rng.randint(2, 4)
        do_wrong_system = False
        do_upstream_trace = False
        do_downstream_impact = False

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
    if path and path.nodes:
        start_system = path.nodes[0].system_id
    else:
        start_system = scenario.root_cause_system

    start_system_name = SYSTEM_DISPLAY_NAMES.get(start_system, start_system)

    # ================================================================
    # Phase 3b: Optional wrong-system exploration (+2 tool calls)
    # ================================================================
    if do_wrong_system and path and path.nodes:
        root_system = scenario.root_cause_system
        all_systems = list(SYSTEM_DISPLAY_NAMES.keys())
        wrong_candidates = [s for s in all_systems if s != root_system and s != start_system]
        if wrong_candidates:
            wrong_sys = rng.choice(wrong_candidates)
            steps.extend(_generate_wrong_system_check(
                wrong_sys, tool_executor, rng
            ))

    # ================================================================
    # Phase 4: Enter starting system (+1 tool call)
    # ================================================================
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
    for i, path_node in enumerate(path.nodes):
        node_id = path_node.node_id
        node_name = path_node.component_name
        node_system = path_node.system_id

        # Cross-system transition with explicit upstream trace
        if node_system != current_system:
            if do_upstream_trace:
                steps.extend(_generate_upstream_trace(
                    current_system, node_system, tool_executor, rng
                ))
            steps.extend(_generate_system_transition(
                current_system, node_system, node_id, tool_executor, rng
            ))
            current_system = node_system

            # Add 1 elimination step in the new system
            if rng.random() < 0.5:
                steps.extend(_generate_elimination_steps(
                    scenario, current_system, path, tool_executor, rng, 1
                ))

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
            sensor_evidence = _format_sensor_evidence(sensors, rng, max_sensors=3)
            final_thought = compose_final_diagnosis(
                node_id, fault_type, affected, sensor_evidence, rng,
            )
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

def _generate_no_fault_investigation(
    scenario: FaultScenario,
    system_id: str,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
) -> List[TrajectoryStep]:
    """Generate investigation steps for a no-fault scenario."""
    steps = []
    path = scenario.diagnostic_path

    # Check 2-3 nodes, all return Normal
    nodes_to_check = []
    if path and path.nodes:
        nodes_to_check = [(n.node_id, n.component_name) for n in path.nodes[:3]]
    else:
        components = tool_executor.execute("get_node_children", {"node_id": f"system::{system_id}"})
        children = components.get("children", [])
        if children:
            selected = rng.sample(children, min(2, len(children)))
            nodes_to_check = [(c["node_id"], c.get("name", c["node_id"])) for c in selected]

    checked_names = []
    for node_id, node_name in nodes_to_check:
        thought = compose_diagnose_request(node_name, rng)
        tool_call = {"name": "diagnose_node", "arguments": {"node_id": node_id}}
        steps.append(TrajectoryStep(
            role="assistant",
            content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
            tool_call=tool_call,
            thought=thought,
        ))

        result = tool_executor.execute("diagnose_node", {"node_id": node_id})
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
        checked_names.append(node_name)

    # Final conclusion with evidence of what was checked
    final = compose_no_fault_conclusion(checked_names, rng)
    diagnosis = {
        "status": "Normal",
        "root_cause_node": "none",
        "fault_type": "None",
        "confidence": 0.90,
    }
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{final}</think>\n\n<diagnosis>{json.dumps(diagnosis)}</diagnosis>",
        thought=final,
    ))
    return steps


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

    to_name = SYSTEM_DISPLAY_NAMES.get(to_system, to_system)
    ack = compose_upstream_ack(from_name, to_name, rng)
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
) -> List[TrajectoryStep]:
    """Generate steps for transitioning between systems (upstream trace)."""
    steps = []
    from_name = SYSTEM_DISPLAY_NAMES.get(from_system, from_system)
    to_name = SYSTEM_DISPLAY_NAMES.get(to_system, to_system)

    # Upstream trace reasoning
    thought = compose_cross_system_transition(from_name, to_name, rng)
    tool_call = {"name": "get_node_children", "arguments": {"node_id": f"system::{to_system}"}}
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call,
        thought=thought,
    ))

    result = tool_executor.execute("get_node_children", {"node_id": f"system::{to_system}"})
    steps.append(TrajectoryStep(
        role="tool",
        content=json.dumps(result, indent=2),
        tool_name="get_node_children",
    ))

    return steps


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
    return DiagnosticTrajectory(
        scenario_id=scenario.scenario_id,
        scenario_type=scenario.scenario_type,
        steps=steps,
        ground_truth={
            "root_cause_system": scenario.root_cause_system,
            "root_cause_node": scenario.root_cause_node,
            "fault_type": scenario.fault_type,
            "fault_intensity": scenario.fault_intensity,
        },
        metadata={
            "n_tool_calls": sum(1 for s in steps if s.tool_call is not None),
            "difficulty": scenario.difficulty,
            "path_length": path.path_length if path else 0,
            "trajectory_quality": "correct" if is_correct else "incorrect",
            "tool_faithful": True,
        },
    )


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

    for i, scenario in enumerate(scenarios):
        try:
            state = None
            if scenario_states:
                state = scenario_states.get(scenario.scenario_id)
            traj = generate_trajectory(
                scenario, tool_executor, rng, scenario_state=state,
            )
            trajectories.append(traj)
        except Exception as e:
            failed += 1
            logger.warning(f"[{scenario.scenario_id}] Generation failed: {e}")
            if failed > len(scenarios) * 0.1:
                logger.error("Too many failures, stopping batch generation")
                break

        if (i + 1) % 100 == 0 or (i + 1) == len(scenarios):
            logger.info(f"Generated {i + 1}/{len(scenarios)} trajectories")

    logger.info(f"Generated {len(trajectories)} trajectories total (failed: {failed})")
    return trajectories
