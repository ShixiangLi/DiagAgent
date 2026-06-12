"""
Trajectory Generator 鈥?Path-guided diagnostic trajectory generation.

Generates multi-turn diagnostic trajectories where the agent follows a
pre-designed diagnostic path from symptom to root cause. The Oracle returns
different responses based on whether the queried node is on the path:

  - Off-path node  鈫?"Normal"  (agent should switch direction)
  - Path symptom   鈫?"Abnormal" + hint (agent should investigate upstream)
  - Path intermediate 鈫?"Abnormal" + hint (agent should continue)
  - Path root cause 鈫?"Fault" + type (agent confirms diagnosis)
"""

import json
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.data_gen.reasoning_composer import (
    compose_initial_reasoning, compose_system_selection, compose_diagnose_request,
    compose_normal_response, compose_uncertain_response,
    compose_abnormal_response, compose_fault_found,
    compose_final_diagnosis, compose_no_fault_conclusion, compose_upstream_trace,
    compose_wrong_system, compose_wrong_system_ack, compose_upstream_ack,
    compose_cross_system_transition, _format_sensor_evidence,
    compose_anomaly_score_selection, compose_multi_system_ranking,
    compose_system_elimination, compose_system_inconclusive, compose_system_pivot,
    compose_warning_response, compose_sensor_verification,
    compose_related_systems_request, compose_related_systems_ack,
    compose_related_systems_impact_request, compose_related_systems_impact_ack,
    compose_children_query, compose_observable_system_start,
    compose_explicit_system_start, compose_cross_downstream_start,
    compose_upstream_anchor_trace,
)
from src.environment.diagnostic_path import DiagnosticPath, PathNode
from src.environment.fault_scenario import FaultScenario
from src.environment.tool_executor import UnifiedToolExecutor
from src.evaluation.fault_taxonomy import fault_exact_match
from src.utils.io_utils import setup_logger

logger = setup_logger(__name__)

NORMAL_CLEAR_CONFIDENCE_THRESHOLD = 0.70
NO_FAULT_CLEAR_CONFIDENCE_THRESHOLD = 0.85
MIN_NO_FAULT_STRONG_NORMALS = 2
MIN_ORDINARY_FAULT_FINAL_CONFIDENCE = 0.60


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
      - Off-path 鈫?Normal
      - Symptom/Intermediate 鈫?Abnormal with hint
      - Root cause 鈫?Fault with type
    """
    path = scenario.diagnostic_path
    if path is None:
        # No path available 鈥?fallback to Normal
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

_OBSERVABLE_AREA_TO_SYSTEM = {
    "central chilled-water plant": "chiller_plant",
    "chilled-water loop": "chiller_plant",
    "condenser-water loop": "chiller_plant",
    "cooling tower area": "chiller_plant",
    "central hot-water plant": "boiler_plant",
    "hot-water loop": "boiler_plant",
    "boiler room": "boiler_plant",
    "heating-water distribution loop": "boiler_plant",
    "single-duct air-handling unit": "sdahu",
    "supply-air duct serving the single-duct ahu zones": "sdahu",
    "outdoor-air section of an air handler": "sdahu",
    "dual-duct air-handling unit": "ddahu",
    "hot-deck and cold-deck air handler": "ddahu",
    "dual-duct mixing box area": "ddahu",
    "rooftop packaged unit": "rtu",
    "dx rooftop unit": "rtu",
    "roof-mounted cooling unit": "rtu",
    "fan-coil zone unit": "fcu",
    "terminal fan-coil unit": "fcu",
    "local zone coil unit": "fcu",
    "parallel fan-powered terminal unit": "pfpu",
    "terminal reheat box with parallel fan": "pfpu",
    "zone terminal unit with local fan": "pfpu",
    "series fan-powered terminal unit": "sfpu",
    "series-flow terminal reheat box": "sfpu",
    "zone terminal unit in series fan mode": "sfpu",
}


def _explicit_system_from_prompt(description: str) -> Optional[str]:
    """Extract a leading ``[System Name]`` prompt hint when present."""
    text = str(description or "").strip()
    if not text.startswith("[") or "]" not in text:
        return None
    display_name = text[1:text.index("]")].strip()
    return _DISPLAY_NAME_TO_SYSTEM.get(display_name)


def _observable_system_from_prompt(description: str) -> Optional[str]:
    """Infer a visible starting system from non-ID equipment/location wording."""
    text = str(description or "").lower()
    for marker, system_id in _OBSERVABLE_AREA_TO_SYSTEM.items():
        if marker in text:
            return system_id
    return None


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
        # System root already queried 鈥?start from already-discovered nodes
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


def _node_has_children(node_id: str, tool_executor: UnifiedToolExecutor) -> bool:
    """Return True when a topology node has child components."""
    try:
        result = tool_executor.execute("get_node_children", {"node_id": node_id})
    except Exception:
        return False
    return bool(result.get("children"))


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
    thought = compose_children_query(parent_name, rng)
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


def _sensor_readings_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Extract runtime readings from a get_node_sensors result."""
    readings = result.get("sensor_readings")
    if isinstance(readings, dict) and readings:
        return readings
    extracted: Dict[str, Any] = {}
    for sensor in result.get("sensors", []) or []:
        if not isinstance(sensor, dict):
            continue
        name = sensor.get("name")
        if name and "current_value" in sensor:
            extracted[str(name)] = sensor.get("current_value")
    return extracted


def _status_lower(result: Dict[str, Any]) -> str:
    return str((result or {}).get("status", "")).strip().lower()


def _confidence_value(result: Dict[str, Any]) -> float:
    try:
        return float((result or {}).get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _needs_sensor_verification(result: Dict[str, Any]) -> bool:
    """Whether a diagnose_node observation should trigger same-node sensors."""
    if not isinstance(result, dict):
        return False
    status = _status_lower(result)
    fault_type = str(result.get("fault_type", "")).strip().lower()
    if status in {"warning", "indeterminate"}:
        return True
    if fault_type == "uncertain":
        return status not in {"normal", "unknown", "error", "data_unavailable"}
    return False


def _is_weak_normal_result(result: Dict[str, Any]) -> bool:
    if not isinstance(result, dict):
        return False
    status = _status_lower(result)
    if status == "normal":
        confidence = _confidence_value(result)
        return 0.0 < confidence < NORMAL_CLEAR_CONFIDENCE_THRESHOLD
    return False


def _is_unresolved_diagnostic_result(result: Dict[str, Any]) -> bool:
    """Evidence that cannot be used as a clean confirmation/elimination."""
    if not isinstance(result, dict):
        return True
    status = _status_lower(result)
    fault_type = str(result.get("fault_type", "")).strip().lower()
    return (
        _needs_sensor_verification(result)
        or _is_weak_normal_result(result)
        or status in {"unknown", "error", "data_unavailable"}
        or fault_type in {"unknown", "uncertain"}
    )


def _has_runtime_sensor_readings(
    tool_executor: UnifiedToolExecutor,
    node_id: str,
) -> bool:
    """Return whether get_node_sensors can provide current readings."""
    provider = getattr(getattr(tool_executor, "topo_executor", None), "_sensor_provider", None)
    scenario_state = getattr(provider, "scenario_state", None)
    if scenario_state is None:
        return False
    try:
        readings = scenario_state.get_sensor_readings(node_id)
    except Exception:
        return False
    if provider is not None and hasattr(provider, "_sanitize_sensor_readings"):
        try:
            readings = provider._sanitize_sensor_readings(readings)
        except Exception:
            pass
    return bool(readings)


def _visible_sensor_evidence_status(
    steps: List[TrajectoryStep],
    node_id: str,
) -> str:
    """Summarize explicit sensor evidence already gathered for a node."""
    for step in reversed(steps):
        if step.role != "tool" or step.tool_name != "get_node_sensors":
            continue
        try:
            result = json.loads(step.content)
        except (json.JSONDecodeError, TypeError):
            continue
        if str(result.get("component_node", "")) != node_id:
            continue
        if result.get("readings_available") is True:
            return "available"
        readings = result.get("sensor_readings")
        if isinstance(readings, dict) and readings:
            return "available"
        sensors = result.get("sensors")
        if isinstance(sensors, list) and any(
            isinstance(sensor, dict) and "current_value" in sensor
            for sensor in sensors
        ):
            return "available"
        return "unavailable"
    return "not_queried"


def _fault_result_matches_scenario(
    result: Dict[str, Any],
    scenario: FaultScenario,
) -> bool:
    return (
        _status_lower(result) == "fault"
        and str(result.get("node_id", "")) == str(scenario.root_cause_node)
        and fault_exact_match(result.get("fault_type", ""), scenario.fault_type)
    )


def _append_sensor_verification_steps(
    steps: List[TrajectoryStep],
    node_id: str,
    node_name: str,
    fault_type: str,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
    lead_thought: Optional[str] = None,
) -> str:
    """Call get_node_sensors on node_id and append grounded reasoning.

    Returns the final sensor-evidence text to cite in the diagnosis.  If the
    tool has no runtime readings, the generated reasoning states that
    limitation rather than fabricating sensor confirmation.
    """
    thought = lead_thought or (
        f"The diagnostic evidence for {node_name} is uncertain, so I need a "
        f"same-node sensor check before using it as diagnosis evidence."
    )
    tool_call = {"name": "get_node_sensors", "arguments": {"node_id": node_id}}
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call,
        thought=thought,
    ))
    sensors_result = tool_executor.execute("get_node_sensors", {"node_id": node_id})
    steps.append(TrajectoryStep(
        role="tool",
        content=json.dumps(sensors_result, indent=2),
        tool_name="get_node_sensors",
    ))

    sensors = _sensor_readings_from_result(sensors_result)
    sensor_evidence = _format_sensor_evidence(
        sensors, rng, max_sensors=3, fault_type=fault_type,
    )
    if sensor_evidence:
        verify_thought = compose_sensor_verification(
            node_name, sensor_evidence, rng,
        )
    else:
        sensor_evidence = (
            "sensor-level verification was requested, but no current readings "
            "were available from the tool response"
        )
        verify_thought = (
            f"The same-node sensor query for {node_name} did not return current "
            f"readings. I will keep this evidence as uncertain and avoid claiming "
            f"direct measurement confirmation."
        )
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{verify_thought}</think>",
        thought=verify_thought,
    ))
    return sensor_evidence


def _append_post_diagnosis_reasoning(
    steps: List[TrajectoryStep],
    node_id: str,
    node_name: str,
    result: Dict[str, Any],
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
    fault_type: str = "",
    extra_thought: Optional[str] = None,
    uncertain_lead: Optional[str] = None,
) -> bool:
    """Append faithful reasoning after diagnose_node.

    Returns True when the result was uncertain and same-node sensors were
    queried.  Callers can use that to avoid treating weak Normal/unknown
    evidence as a clean elimination.
    """
    if _needs_sensor_verification(result):
        if not _has_runtime_sensor_readings(tool_executor, node_id):
            unresolved = _compose_diagnosis_followup(node_name, result, rng)
            unresolved = (
                f"{unresolved}\nA same-node sensor check is not useful here "
                f"because the runtime tool has no current readings for "
                f"{node_name}. I will keep this evidence unresolved and rely "
                f"on topology plus other component observations."
            )
            if extra_thought:
                unresolved = f"{unresolved}\n{extra_thought}"
            steps.append(TrajectoryStep(
                role="assistant",
                content=f"<think>{unresolved}</think>",
                thought=unresolved,
            ))
            return False
        lead = uncertain_lead or _compose_diagnosis_followup(node_name, result, rng)
        lead = (
            f"{lead}\nThis is not strong enough to eliminate or confirm "
            f"{node_name}; I need same-node sensor evidence before moving on."
        )
        _append_sensor_verification_steps(
            steps,
            node_id,
            node_name,
            fault_type,
            tool_executor,
            rng,
            lead_thought=lead,
        )
        if extra_thought:
            cautious = (
                f"{extra_thought}\nBecause the previous diagnostic evidence was "
                f"uncertain, I will keep {node_name} as unresolved rather than "
                f"treating it as a clean exclusion."
            )
            steps.append(TrajectoryStep(
                role="assistant",
                content=f"<think>{cautious}</think>",
                thought=cautious,
            ))
        return True

    ack = _compose_diagnosis_followup(node_name, result, rng)
    if extra_thought:
        ack = f"{ack}\n{extra_thought}"
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{ack}</think>",
        thought=ack,
    ))
    return False


def _diag_node_name(result: Dict[str, Any], fallback_node_id: str) -> str:
    name = str(result.get("node_name") or "").strip() if isinstance(result, dict) else ""
    if name:
        return name
    return fallback_node_id.split("::")[-1] if "::" in fallback_node_id else fallback_node_id


def _system_id_from_node(node_id: str) -> str:
    return node_id.split("::", 1)[0] if "::" in str(node_id) else ""


def _topology_builder_from_executor(tool_executor: UnifiedToolExecutor):
    return getattr(getattr(tool_executor, "topo_executor", None), "tb", None)


def _topology_downstream_nodes(
    tool_executor: UnifiedToolExecutor,
    node_id: str,
) -> Optional[List[Dict[str, Any]]]:
    """Return topology downstream nodes when available without emitting a tool turn."""
    tb = _topology_builder_from_executor(tool_executor)
    if tb is None:
        return None
    try:
        return tb.get_downstream_nodes(node_id)
    except Exception:
        return None


def _has_topology_downstream(
    tool_executor: UnifiedToolExecutor,
    node_id: str,
) -> bool:
    downstream = _topology_downstream_nodes(tool_executor, node_id)
    if downstream is None:
        return True
    return bool(downstream)


def _node_display_name(node_id: str, fallback: str = "") -> str:
    if fallback:
        return fallback
    return node_id.split("::")[-1] if "::" in str(node_id) else str(node_id)


def _select_cross_impact_node(
    root_node: str,
    root_system: str,
    upstream_result: Dict[str, Any],
    tool_executor: UnifiedToolExecutor,
) -> Dict[str, str]:
    """Choose the visible upstream node that can expose downstream scope.

    Some plant-level classifiers diagnose a simulated plant aggregate while
    cross-system edges are attached to a distribution node such as
    Chilled_Water_System.  The trajectory may use that already-visible
    upstream node for impact tracing, but it must keep the root diagnosis on
    the same node that returned Fault.
    """
    candidates: List[Dict[str, str]] = [{
        "node_id": root_node,
        "name": _node_display_name(root_node),
    }]
    related_result = upstream_result.get("_related_systems")
    if isinstance(related_result, dict):
        for item in related_result.get("downstream_connections", []) or []:
            connected_via = str(item.get("connected_via") or "")
            if not connected_via or _system_id_from_node(connected_via) != root_system:
                continue
            candidates.append({
                "node_id": connected_via,
                "name": _node_display_name(connected_via),
                "medium": str(item.get("medium") or ""),
            })
    for item in upstream_result.get("upstream", []) or []:
        if item.get("system_id") != root_system or not item.get("node_id"):
            continue
        candidates.append({
            "node_id": str(item.get("node_id")),
            "name": str(item.get("name") or _node_display_name(item.get("node_id", ""))),
            "medium": str(item.get("medium") or ""),
        })

    seen = set()
    unique = []
    for candidate in candidates:
        node_id = candidate["node_id"]
        if node_id in seen:
            continue
        seen.add(node_id)
        unique.append(candidate)

    for candidate in unique:
        downstream = _topology_downstream_nodes(tool_executor, candidate["node_id"])
        if downstream:
            return candidate
    return unique[0]


def _affected_systems_from_visible_scope(
    scenario: FaultScenario,
    downstream_result: Optional[Dict[str, Any]] = None,
    observed_systems: Optional[List[str]] = None,
) -> List[str]:
    supported = set(observed_systems or [])
    if scenario.root_cause_system:
        supported.add(scenario.root_cause_system)
    if isinstance(downstream_result, dict):
        for item in downstream_result.get("downstream", []) or []:
            system_id = item.get("system_id")
            if system_id:
                supported.add(str(system_id))

    ordered = [
        system_id for system_id in scenario.affected_systems
        if system_id in supported
    ]
    for system_id in sorted(supported):
        if system_id and system_id not in ordered:
            ordered.append(system_id)
    return ordered or list(scenario.affected_systems)


def _impact_scope_step(
    steps: List[TrajectoryStep],
    scenario: FaultScenario,
    root_node: str,
    root_name: str,
    root_system: str,
    upstream_result: Dict[str, Any],
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
    fault_thought: str,
) -> Dict[str, Any]:
    """Append a downstream-scope query using a topology-visible impact node."""
    impact_node = _select_cross_impact_node(
        root_node, root_system, upstream_result, tool_executor,
    )
    impact_node_id = impact_node["node_id"]
    impact_node_name = impact_node.get("name") or _node_display_name(impact_node_id)
    ds_call = {"name": "get_downstream_nodes", "arguments": {"node_id": impact_node_id}}
    if impact_node_id == root_node:
        impact_thought = (
            f"I will verify downstream impact from {root_name} before "
            f"finalizing the affected-system scope."
        )
    else:
        impact_thought = (
            f"The Fault was identified at {root_name}; the topology also "
            f"exposes {impact_node_name} as the visible cross-system "
            f"connection. I will use that connection node to verify downstream "
            f"impact scope."
        )
    steps.append(TrajectoryStep(
        role="assistant",
        content=(
            f"<think>{fault_thought}\n{impact_thought}</think>\n"
            f"<tool_call>{json.dumps(ds_call)}</tool_call>"
        ),
        tool_call=ds_call,
        thought=fault_thought + "\n" + impact_thought,
    ))
    ds_result = tool_executor.execute("get_downstream_nodes", {"node_id": impact_node_id})
    steps.append(TrajectoryStep(
        role="tool",
        content=json.dumps(ds_result, indent=2),
        tool_name="get_downstream_nodes",
    ))
    ds_result["_impact_node_name"] = impact_node_name
    return ds_result


def _append_cross_final(
    steps: List[TrajectoryStep],
    scenario: FaultScenario,
    root_node: str,
    root_name: str,
    root_system_name: str,
    ft: str,
    confidence: float,
    ds_result: Dict[str, Any],
    observed_systems: List[str],
    rng: random.Random,
    start_context: str,
) -> DiagnosticTrajectory:
    visible_affected = _affected_systems_from_visible_scope(
        scenario,
        downstream_result=ds_result,
        observed_systems=observed_systems,
    )
    affected = ", ".join(visible_affected)
    final = compose_final_diagnosis(root_node, ft, affected, "", rng)
    diagnosis = {
        "root_cause_node": root_node,
        "fault_type": ft,
        "confidence": confidence,
        "affected_systems": visible_affected,
    }
    impact_node_name = ds_result.get("_impact_node_name", root_name)
    downstream_items = ds_result.get("downstream", []) if isinstance(ds_result, dict) else []
    fast_closure = (
        isinstance(ds_result, dict)
        and ds_result.get("status") == "not_queried_after_root_confirmation"
    )
    if fast_closure:
        closing = (
            f"The causal chain is now explicit: {start_context} and same-node "
            f"Fault at {root_name}. This is sufficient root-cause evidence, "
            f"so I will close the cross-system diagnosis now rather than spend "
            f"extra turns on impact-scope queries."
        )
    elif downstream_items:
        closing = (
            f"The causal chain is now explicit: {start_context}, "
            f"same-node Fault at {root_name}, and visible downstream links "
            f"from {impact_node_name} to the affected systems."
        )
    else:
        closing = (
            f"The causal chain is now explicit: {start_context} and same-node "
            f"Fault at {root_name}. The downstream query returned no child "
            f"links, so I will report only systems supported by the observed trace."
        )
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{closing}</think>\n\n{final}\n\n<diagnosis>{json.dumps(diagnosis)}</diagnosis>",
        thought=closing + "\n" + final,
    ))
    return _build_trajectory(scenario, steps, is_correct=True)


def _observed_cross_scope_result(
    *,
    root_name: str,
    observed_systems: List[str],
) -> Dict[str, Any]:
    """Synthetic scope marker for fast cross-system closure.

    The diagnosis remains grounded in real same-node root evidence.  This marker
    prevents the SFT target from teaching an extra downstream-impact tool call
    after the root Fault has already been confirmed.
    """
    return {
        "status": "not_queried_after_root_confirmation",
        "downstream": [],
        "_impact_node_name": root_name,
        "_fast_closure_observed_systems": list(observed_systems),
    }


def _select_cross_trace_node(
    scenario: FaultScenario,
    tool_executor: UnifiedToolExecutor,
    preferred_system: Optional[str] = None,
) -> PathNode:
    """Select a downstream component that has a real upstream cross edge."""
    tb = _topology_builder_from_executor(tool_executor)
    path = scenario.diagnostic_path
    root_system = scenario.root_cause_system
    root_node = scenario.root_cause_node

    downstream_nodes = []
    if path and path.nodes:
        downstream_nodes = [
            pnode for pnode in path.nodes
            if pnode.system_id and pnode.system_id != root_system
        ]

    if tb is not None:
        if preferred_system and preferred_system != root_system:
            for pnode in downstream_nodes:
                if pnode.system_id != preferred_system:
                    continue
                upstream = tb.get_upstream_nodes(pnode.node_id)
                if any(item.get("node_id") == root_node for item in upstream):
                    return pnode
                if any(item.get("system_id") == root_system for item in upstream):
                    return pnode
            candidate = _connected_target_node(
                preferred_system, root_system, root_node, tool_executor,
            )
            if candidate:
                return candidate
            raise ValueError(
                "Prompt-named downstream system does not expose an upstream "
                f"cross trace to root: scenario={scenario.scenario_id}, "
                f"preferred={preferred_system}, root={root_system}"
            )

        for pnode in downstream_nodes:
            upstream = tb.get_upstream_nodes(pnode.node_id)
            if any(item.get("node_id") == root_node for item in upstream):
                return pnode
        for pnode in downstream_nodes:
            upstream = tb.get_upstream_nodes(pnode.node_id)
            if any(item.get("system_id") == root_system for item in upstream):
                return pnode

        downstream_systems = [pnode.system_id for pnode in downstream_nodes]
        for downstream_system in downstream_systems:
            candidate = _connected_target_node(
                downstream_system, root_system, root_node, tool_executor,
            )
            if candidate:
                return candidate

        for affected_system in scenario.affected_systems:
            if affected_system == root_system:
                continue
            candidate = _connected_target_node(
                affected_system, root_system, root_node, tool_executor,
            )
            if candidate:
                return candidate

    if downstream_nodes:
        return downstream_nodes[0]
    raise ValueError(f"Cross-system scenario has no downstream trace node: {scenario.scenario_id}")


def _connected_target_node(
    downstream_system: str,
    root_system: str,
    root_node: str,
    tool_executor: Optional[UnifiedToolExecutor] = None,
) -> Optional[PathNode]:
    if tool_executor is None:
        return None
    tb = _topology_builder_from_executor(tool_executor)
    if tb is None:
        return None
    fallback: Optional[PathNode] = None
    for comp in tb.get_system_components(downstream_system):
        node_id = comp.get("node_id", "")
        upstream = tb.get_upstream_nodes(node_id)
        matches = [
            item for item in upstream
            if item.get("system_id") == root_system
        ]
        if not matches:
            continue
        candidate = PathNode(
            node_id=node_id,
            role="symptom",
            system_id=downstream_system,
            component_name=comp.get("name", node_id.split("::")[-1]),
            abnormal_hint=(
                "Downstream component is connected to an upstream plant; "
                "trace the supply relationship before deciding root cause."
            ),
            direction="upstream",
        )
        if any(item.get("node_id") == root_node for item in matches):
            return candidate
        if fallback is None:
            fallback = candidate
    return fallback


def _cross_anchor_for_transition(
    related_result: Dict[str, Any],
    from_system: str,
    to_system: str,
    preferred_target_component: Optional[str] = None,
    preferred_connected_via: Optional[str] = None,
) -> Dict[str, Any]:
    """Return the best node-level anchor from get_related_systems output."""
    if not isinstance(related_result, dict):
        return {}

    scored: List[tuple] = []
    for key in ("upstream_connections", "downstream_connections"):
        direction_bonus = 0
        if key == "upstream_connections":
            direction_bonus = 3
        for item in related_result.get(key, []) or []:
            if str(item.get("system_id") or "") != to_system:
                continue
            connected_via = str(item.get("connected_via") or "")
            target_component = str(item.get("target_component") or "")
            score = direction_bonus
            if preferred_target_component and target_component == preferred_target_component:
                score += 10
            if preferred_connected_via and connected_via == preferred_connected_via:
                score += 10
            if _system_id_from_node(target_component) == from_system:
                score += 3
            if _system_id_from_node(connected_via) == to_system:
                score += 3
            scored.append((score, dict(item, _connection_key=key)))
    if not scored:
        return {}
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[0][1]


def _query_system_children(
    steps: List[TrajectoryStep],
    system_id: str,
    thought: str,
    tool_executor: UnifiedToolExecutor,
) -> Dict[str, Any]:
    tool_call = {
        "name": "get_node_children",
        "arguments": {"node_id": f"system::{system_id}"},
    }
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call,
        thought=thought,
    ))
    result = tool_executor.execute("get_node_children", {"node_id": f"system::{system_id}"})
    steps.append(TrajectoryStep(
        role="tool",
        content=json.dumps(result, indent=2),
        tool_name="get_node_children",
    ))
    return result


def _append_cross_drill_correction(
    steps: List[TrajectoryStep],
    root_node: str,
    root_system: str,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
) -> bool:
    """Teach Normal-then-drill: diagnose a sibling/aggregate node first.

    In cross-system cases the policy tends to stop at the first plant node the
    topology exposes (e.g. the plant aggregate) instead of the exact responsible
    component. With the soft node gate, diagnosing such a sibling returns
    ``Abnormal`` with a ``within_system`` directional hint (the exact responsible
    node is kept internal so the agent cannot shortcut to the answer). This
    helper emits that step plus a short "drill to the root component" rationale
    that names the KNOWN ground-truth root (legitimate expert demonstration),
    so SFT sees the corrective pattern that fixes cross-system DA. The
    subsequent same-node Fault diagnosis on ``root_node`` is appended by the
    caller.

    Returns True if a correction step was emitted, False otherwise (e.g. no
    visible sibling, or the probe did not yield the expected Abnormal signal).
    """
    # Only consider siblings ALREADY discovered in the visible trajectory, so
    # the corrective diagnose_node never introduces a step-skip (querying an
    # undiscovered node). Discovered nodes are recovered from prior tool
    # observations in ``steps``.
    discovered: set = set()
    for st in steps:
        if st.role != "tool":
            continue
        try:
            obs = json.loads(st.content)
        except (json.JSONDecodeError, TypeError):
            continue
        for child in obs.get("children", []) or []:
            if isinstance(child, dict) and child.get("node_id"):
                discovered.add(str(child["node_id"]))
        for key in ("upstream", "downstream"):
            for item in obs.get(key, []) or []:
                if isinstance(item, dict) and item.get("node_id"):
                    discovered.add(str(item["node_id"]))
        for key in ("upstream_connections", "downstream_connections"):
            for item in obs.get(key, []) or []:
                if not isinstance(item, dict):
                    continue
                for nk in ("connected_via", "target_component"):
                    if item.get(nk):
                        discovered.add(str(item[nk]))

    sibling_ids = [
        nid for nid in discovered
        if nid != root_node
        and nid.split("::")[0] == root_system
        and not nid.startswith("system::")
    ]
    # Prefer an aggregate-looking node (plant aggregate) when present, since
    # that is exactly the node the policy wrongly concludes on at inference.
    sibling_ids.sort(
        key=lambda n: (
            0 if any(t in n for t in ("Simulated_", "Plant", "System")) else 1
        )
    )
    if not sibling_ids:
        return False
    sibling = sibling_ids[0]
    sibling_name = sibling.split("::")[-1]

    probe = tool_executor.execute("diagnose_node", {"node_id": sibling})
    status = str(probe.get("status", "")).strip().lower()
    # Only use this as a teaching step when the soft gate flags the sibling as
    # implicated (Abnormal within-system). The exact responsible node is not in
    # the agent-visible probe (stripped); we drill to the KNOWN GT root instead.
    if status != "abnormal":
        return False

    think = compose_diagnose_request(sibling_name, rng)
    tc = {"name": "diagnose_node", "arguments": {"node_id": sibling}}
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{think}</think>\n<tool_call>{json.dumps(tc)}</tool_call>",
        tool_call=tc,
        thought=think,
    ))
    steps.append(TrajectoryStep(
        role="tool",
        content=json.dumps(probe, indent=2),
        tool_name="diagnose_node",
    ))
    root_name = root_node.split("::")[-1]
    drill = (
        f"{sibling_name} is implicated by a system-level fault but is not the "
        f"root cause; the result directs me to keep localizing within "
        f"{root_system}. I will diagnose {root_name} directly instead of "
        f"concluding here."
    )
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{drill}</think>",
        thought=drill,
    ))
    return True


def _append_diagnose_node(
    steps: List[TrajectoryStep],
    node_id: str,
    node_name: str,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
    scenario: Optional[FaultScenario] = None,
) -> Dict[str, Any]:
    thought = compose_diagnose_request(node_name, rng)
    tool_call = {"name": "diagnose_node", "arguments": {"node_id": node_id}}
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call,
        thought=thought,
    ))
    if scenario is not None:
        result = tool_executor.execute("diagnose_node", {"node_id": node_id})
    else:
        result = tool_executor.execute("diagnose_node", {"node_id": node_id})
    steps.append(TrajectoryStep(
        role="tool",
        content=json.dumps(result, indent=2),
        tool_name="diagnose_node",
    ))
    return result


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


def _append_cross_root_discovery_steps(
    steps: List[TrajectoryStep],
    *,
    root_node: str,
    root_system: str,
    root_system_name: str,
    anchor_connected: str,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
) -> None:
    """Reveal a cross-system root node with the shortest visible topology path."""
    root_discovered: set = set()
    root_queried: set = set()
    parent_chain = _get_has_part_parent_chain(root_node, tool_executor)
    if (
        anchor_connected
        and anchor_connected != root_node
        and anchor_connected in parent_chain
        and _system_id_from_node(anchor_connected) == root_system
    ):
        steps.extend(_query_children_step(
            anchor_connected,
            root_discovered,
            root_queried,
            tool_executor,
            rng,
        ))
        if root_node in root_discovered:
            return

    root_thought = compose_children_query(
        root_system_name, rng, purpose="root_reveal",
    )
    root_children = _query_system_children(
        steps, root_system, root_thought, tool_executor,
    )
    root_discovered.update(
        child.get("node_id")
        for child in root_children.get("children", [])
        if child.get("node_id")
    )
    root_queried.add(f"system::{root_system}")
    steps.extend(_ensure_node_discovered(
        root_node,
        root_discovered,
        tool_executor,
        rng,
        _queried_parents=root_queried,
    ))


# ============================================================================
# Trajectory generation 鈥?main entry point
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

    # ---- Type-specific trajectory dispatch (2脳4 matrix) ----
    # Ambiguous types
    if stype in ("a_single_system", "ambiguous"):
        return _generate_ambiguous_trajectory(
            scenario, tool_executor, rng, scenario_state,
        )
    if stype == "a_cross_system":
        return _generate_cross_system_trace_trajectory(
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
        return _generate_cross_system_trace_trajectory(
            scenario, tool_executor, rng, scenario_state,
        )
    else:  # na_single_system, single_system
        n_elimination = rng.randint(1, 2)
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

    # Internal system-ranking prior only; generated reasoning must stay grounded
    # in visible topology and symptom cues, not hidden health fields.
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
    # Compose visible topology/symptom reasoning from the internal candidate rank.
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

    # ================================================================
    # Phase 5: No-fault scenarios
    # ================================================================
    if scenario.fault_type in ("Normal", "None") or path is None:
        steps.extend(_generate_no_fault_investigation(
            scenario, start_system, tool_executor, rng
        ))
        return _build_trajectory(scenario, steps, is_correct=True)

    # ================================================================
    # Phase 6: Elimination steps 鈥?check wrong nodes first (+N tool calls)
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
        is_expand_only = (
            path_node.role != "root_cause"
            and _node_has_children(node_id, tool_executor)
        )

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

        if is_expand_only:
            steps.extend(_query_children_step(
                node_id, discovered_nodes, queried_parents, tool_executor, rng,
            ))
            continue

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
        steps.append(TrajectoryStep(
            role="tool", content=json.dumps(diag_result, indent=2),
            tool_name="diagnose_node",
        ))

        status = diag_result["status"]

        if status == "Abnormal":
            hint = diag_result.get("abnormal_indicators", "")
            direction = diag_result.get("suggested_direction", "upstream")
            thought = compose_abnormal_response(node_name, hint, direction, {}, rng)
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
            thought = compose_fault_found(node_name, fault_type, confidence, {}, rng)

            # Cross-system faults need an explicit system-relationship query so
            # the agent learns to justify affected-system scope with topology.
            if is_cross and current_system not in related_systems_queried:
                steps.extend(_generate_related_systems_impact_step(
                    current_system, tool_executor, rng,
                ))
                related_systems_queried.add(current_system)

            # Phase 8: Downstream impact verification (+1 tool call)
            if do_downstream_impact:
                if _has_topology_downstream(tool_executor, node_id):
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
                else:
                    scope_note = (
                        f"{thought}\n{node_name} does not expose downstream links "
                        f"in the visible topology, so I will not claim a separate "
                        f"downstream-impact verification from that node."
                    )
                    steps.append(TrajectoryStep(
                        role="assistant",
                        content=f"<think>{scope_note}</think>",
                        thought=scope_note,
                    ))

            # Phase 9: Final diagnosis. Sensor evidence is only cited when it
            # came from an explicit get_node_sensors call; diagnose_node is a
            # model verdict, not the raw-evidence tool in the main experiment.
            affected_systems = _affected_systems_from_visible_scope(
                scenario,
                observed_systems=[current_system],
            )
            affected = ", ".join(affected_systems)
            sensor_evidence = ""
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
                "affected_systems": affected_systems,
            }
            diag_json = json.dumps(diagnosis)
            steps.append(TrajectoryStep(
                role="assistant",
                content=f"<think>{thought}</think>\n\n{final_thought}\n\n<diagnosis>{diag_json}</diagnosis>",
                thought=thought + "\n" + final_thought,
            ))
            break
        else:
            _append_post_diagnosis_reasoning(
                steps,
                node_id,
                node_name,
                diag_result,
                tool_executor,
                rng,
                fault_type=scenario.fault_type,
            )

    return _build_trajectory(scenario, steps, is_correct=True)


# ============================================================================
# Helper generators
# ============================================================================


def _select_representative_components(
    children: List[Dict[str, Any]],
    rng: random.Random,
    max_count: int = 3,
    tool_executor: Optional[UnifiedToolExecutor] = None,
) -> List[Dict[str, Any]]:
    """Select a compact, topology-visible set of components to verify."""
    visible = [
        c for c in children
        if isinstance(c, dict) and c.get("node_id") and "::" in c.get("node_id", "")
    ]
    if tool_executor is not None:
        leaves = [
            c for c in visible
            if not _node_has_children(str(c.get("node_id", "")), tool_executor)
        ]
        if leaves:
            visible = leaves
    if len(visible) <= max_count:
        return visible

    # Cover both early and late positions in the topology listing, then add one
    # random component to avoid teaching a rigid first-N scanning pattern.
    selected = [visible[0], visible[-1]]
    remaining = [
        c for c in visible
        if c.get("node_id") not in {s.get("node_id") for s in selected}
    ]
    if remaining and len(selected) < max_count:
        selected.extend(rng.sample(remaining, min(max_count - len(selected), len(remaining))))
    return selected[:max_count]


def _select_no_fault_review_components(
    system_id: str,
    first_children: List[Dict[str, Any]],
    steps: List[TrajectoryStep],
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
    min_count: int = MIN_NO_FAULT_STRONG_NORMALS,
    max_count: int = 3,
) -> List[Dict[str, Any]]:
    """Select visible leaf components for no-fault closure.

    If the system root exposes only aggregate components, this function emits
    topology-expansion turns until enough leaf candidates are visible.  That
    prevents SFT from teaching that a single normal aggregate check closes a
    broad no-fault request.
    """
    leaves = _leaf_components(first_children, tool_executor)
    expandable = [
        child for child in first_children
        if isinstance(child, dict)
        and child.get("node_id")
        and child.get("node_id") not in {leaf.get("node_id") for leaf in leaves}
    ]
    queried = {f"system::{system_id}"}

    for child in expandable:
        if len(leaves) >= min_count:
            break
        parent_id = str(child.get("node_id"))
        if parent_id in queried:
            continue
        queried.add(parent_id)
        parent_name = child.get("name") or _node_display_name(parent_id)
        thought = compose_children_query(parent_name, rng, purpose="no_fault")
        tool_call = {"name": "get_node_children", "arguments": {"node_id": parent_id}}
        steps.append(TrajectoryStep(
            role="assistant",
            content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
            tool_call=tool_call,
            thought=thought,
        ))
        result = tool_executor.execute("get_node_children", {"node_id": parent_id})
        steps.append(TrajectoryStep(
            role="tool",
            content=json.dumps(result, indent=2),
            tool_name="get_node_children",
        ))
        leaves.extend(_leaf_components(result.get("children", []) or [], tool_executor))

    unique = []
    seen = set()
    for comp in leaves:
        node_id = comp.get("node_id")
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)
        unique.append(comp)

    if len(unique) <= max_count:
        return unique
    selected = [unique[0], unique[-1]]
    remaining = [
        comp for comp in unique
        if comp.get("node_id") not in {item.get("node_id") for item in selected}
    ]
    if remaining and len(selected) < max_count:
        selected.extend(rng.sample(remaining, min(max_count - len(selected), len(remaining))))
    return selected[:max_count]


def _leaf_components(
    children: List[Dict[str, Any]],
    tool_executor: UnifiedToolExecutor,
    exclude_ids: Optional[set] = None,
) -> List[Dict[str, Any]]:
    """Return already-visible components that are safe diagnose_node targets."""
    exclude_ids = exclude_ids or set()
    leaves: List[Dict[str, Any]] = []
    for child in children:
        if not isinstance(child, dict):
            continue
        node_id = str(child.get("node_id", ""))
        if not node_id or node_id in exclude_ids:
            continue
        if _node_has_children(node_id, tool_executor):
            continue
        leaves.append(child)
    return leaves


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

        # Hidden health scores are internal sampling priors only.
        overview_systems = overview_result.get("systems", [])
        real_score = next(
            (s.get("anomaly_score", 0.0) for s in overview_systems if s.get("system_id") == sys_id),
            0.0,
        )

        # Phase 3: select system using visible topology and symptom cues.
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

        # Phase 4: Verify a small topology-visible subset.  The agent must
        # choose from components revealed by get_node_children; no hidden
        # all-node status summary is used in the main experiment.
        children = children_result.get("children", [])
        representative = _select_no_fault_review_components(
            sys_id,
            children,
            steps,
            tool_executor,
            rng,
            min_count=MIN_NO_FAULT_STRONG_NORMALS,
            max_count=3,
        )
        strong_normals = 0
        required_normals = min(MIN_NO_FAULT_STRONG_NORMALS, max(1, len(representative)))

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
            all_checked.append(nname)

            if (
                str(result.get("status", "")).strip().lower() == "normal"
                and _confidence_value(result) >= NO_FAULT_CLEAR_CONFIDENCE_THRESHOLD
            ):
                strong_normals += 1
                ack = _compose_no_fault_normal_ack(
                    nname,
                    _confidence_value(result),
                    {},
                    rng,
                )
                if strong_normals >= required_normals:
                    ack = (
                        f"{ack} The no-fault review now has enough "
                        f"high-confidence Normal evidence in {sys_name}. "
                        f"I should close with root_cause_node=\"none\" and "
                        f"fault_type=\"no_fault\" instead of tracing upstream "
                        f"or checking unrelated systems."
                    )
                else:
                    remaining = required_normals - strong_normals
                    ack = (
                        f"{ack} I still need {remaining} more visible "
                        f"high-confidence Normal check(s) in {sys_name} "
                        f"before closing no-fault; I should stay inside the "
                        f"current system rather than query related systems."
                    )
                steps.append(TrajectoryStep(
                    role="assistant",
                    content=f"<think>{ack}</think>",
                    thought=ack,
                ))
                if strong_normals >= required_normals:
                    break
            else:
                _append_post_diagnosis_reasoning(
                    steps,
                    nid,
                    nname,
                    result,
                    tool_executor,
                    rng,
                )

        if not representative:
            all_checked.extend(
                c.get("name", c.get("node_id", ""))
                for c in children[:3]
            )
        if strong_normals < required_normals:
            raise ValueError(
                "No-fault trajectory lacks enough high-confidence Normal "
                f"evidence: scenario={scenario.scenario_id}, system={sys_id}, "
                f"strong_normals={strong_normals}, required={required_normals}"
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


def _compose_diagnosis_followup(
    node_name: str,
    result: Dict[str, Any],
    rng: random.Random,
    abnormal_direction: str = "upstream",
) -> str:
    """Explain a diagnose_node result without normalizing unknown evidence."""
    status = result.get("status", "unknown")
    confidence = float(result.get("confidence", 0.0) or 0.0)

    if status == "Normal":
        if _is_weak_normal_result(result):
            return compose_uncertain_response(
                node_name,
                status,
                confidence,
                "normal-status confidence is too low to use as clean evidence",
                {},
                rng,
            )
        return compose_normal_response(node_name, confidence, {}, rng)

    if status == "Abnormal":
        hint = result.get(
            "abnormal_indicators",
            "Operating parameters are outside normal ranges.",
        )
        return compose_abnormal_response(
            node_name, hint, abnormal_direction, {}, rng,
        )

    if status == "Warning":
        return compose_warning_response(node_name, confidence, {}, rng)

    message = (
        result.get("message")
        or result.get("reason")
        or result.get("error")
        or result.get("model_source")
        or "diagnostic evidence unavailable"
    )
    return compose_uncertain_response(
        node_name, status, confidence, str(message), {}, rng,
    )


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
    off_path = _leaf_components(off_path, tool_executor)

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
    }
    steps.append(TrajectoryStep(
        role="tool",
        content=json.dumps(result, indent=2),
        tool_name="diagnose_node",
    ))

    ack = _compose_diagnosis_followup(node_name, result, rng)
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

    def is_leaf_candidate(component: Dict[str, Any]) -> bool:
        node_id = str(component.get("node_id", ""))
        if not node_id or node_id in path_ids:
            return False
        child_result = tool_executor.execute("get_node_children", {"node_id": node_id})
        return not child_result.get("children")

    off_path = [c for c in children if is_leaf_candidate(c)]

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

        _append_post_diagnosis_reasoning(
            steps,
            node_id,
            node_name,
            result,
            tool_executor,
            rng,
            fault_type=scenario.fault_type,
        )

    return steps


def _generate_related_systems_step(
    from_system: str,
    to_system: str,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
    target_component: Optional[str] = None,
    connected_via: Optional[str] = None,
) -> List[TrajectoryStep]:
    """Generate a get_related_systems step before a cross-system transition."""
    steps, _ = _generate_related_systems_step_with_result(
        from_system,
        to_system,
        tool_executor,
        rng,
        target_component=target_component,
        connected_via=connected_via,
    )
    return steps


def _generate_related_systems_step_with_result(
    from_system: str,
    to_system: str,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
    target_component: Optional[str] = None,
    connected_via: Optional[str] = None,
) -> Tuple[List[TrajectoryStep], Dict[str, Any]]:
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
    anchor = _cross_anchor_for_transition(
        result,
        from_system,
        to_system,
        preferred_target_component=target_component,
        preferred_connected_via=connected_via,
    )
    ack = compose_related_systems_ack(
        from_name,
        to_name,
        rng,
        connected_via=anchor.get("connected_via") or connected_via,
        target_component=anchor.get("target_component") or target_component,
        medium=anchor.get("medium"),
    )
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{ack}</think>",
        thought=ack,
    ))
    return steps, result


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


def _append_related_systems_impact_step(
    steps: List[TrajectoryStep],
    system_id: str,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
) -> Dict[str, Any]:
    """Append get_related_systems after a root fault and return its result."""
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
    return result


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
    leaf_children = _leaf_components(children, tool_executor)
    if leaf_children:
        comp = rng.choice(leaf_children)
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

        if _needs_sensor_verification(result):
            _append_post_diagnosis_reasoning(
                steps,
                node_id,
                node_name,
                result,
                tool_executor,
                rng,
            )
        else:
            actual_status = result.get("status", "Normal")
            confidence = result.get("confidence", 0.90)
            ack = compose_wrong_system_ack(
                node_name, sys_name, actual_status, confidence, rng,
            )
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
        # Upstream nodes found 鈥?acknowledge the connection
        ack = compose_upstream_ack(from_name, to_name, rng)
    else:
        # No direct upstream link returned; fall back to topology reasoning.
        ack = rng.choice([
            f"No direct upstream link was returned for {node_name}, but the building's thermal architecture still makes {to_name} a plausible upstream source. Switching focus.",
            f"The topology query returned no explicit upstream for {node_name}. However, the cross-system fault propagation pattern points toward {to_name}, so I should investigate there next.",
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
      A (40%): Agent picks the best topology/symptom candidate directly 鈫?finds fault
      B (40%): Agent tries wrong system first 鈫?eliminates 鈫?pivots 鈫?finds fault
      C (20%): Agent tries two wrong systems 鈫?eliminates both 鈫?finds fault
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

    # Parse hidden health scores only as internal sampling priors.
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
    observable_system = _observable_system_from_prompt(scenario.description)
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

    # Select variant. If the prompt contains a visible equipment/location cue,
    # the expert should start from that observable system instead of using a
    # hidden root-system shortcut or an arbitrary wrong-system sweep.
    roll = rng.random()
    if observable_system:
        variant = "direct"
        correct_sys = observable_system
        correct_name = SYSTEM_DISPLAY_NAMES.get(correct_sys, correct_sys)
        wrong_candidates = [
            item for item in wrong_candidates if item[1] != observable_system
        ]
    elif roll < 0.4:
        variant = "direct"
    elif roll < 0.8:
        variant = "one_wrong"
    else:
        variant = "two_wrong"

    # Phase 3: topology/symptom ranking + system selection.
    # This avoids contradictory double-think blocks
    if variant == "direct":
        # Go directly to the visible best candidate using visible reasoning.
        correct_score = next(
            (sc for _, sid, sc in scored if sid == correct_sys), 0.5
        )
        if observable_system:
            thought = compose_observable_system_start(correct_name, rng)
        else:
            thought = compose_anomaly_score_selection(
                correct_name, correct_sys, correct_score, rng,
            )
    else:
        # Go to first wrong system 鈥?use simple system selection
        # Do not claim hidden health-score evidence.
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
        # Check 1-2 components in wrong system 鈫?all Normal
        children = children_result.get("children", [])
        children = _leaf_components(children, tool_executor)
        n_check = rng.randint(1, min(2, len(children))) if children else 0
        checked = rng.sample(children, n_check) if children else []
        first_system_all_normal = True
        last_wrong_name = first_name
        last_wrong_clear = True

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
            is_uncertain = _is_unresolved_diagnostic_result(res)
            actual_status = res.get("status", "Normal")
            if actual_status != "Normal" or is_uncertain:
                first_system_all_normal = False
            # For the last checked component, merge ack + elimination reasoning
            if first_system_all_normal:
                elim = compose_system_elimination(first_name, n_check, rng)
            else:
                elim = compose_system_inconclusive(first_name, n_check, rng)
            if comp == checked[-1]:
                _append_post_diagnosis_reasoning(
                    steps,
                    nid,
                    nname,
                    res,
                    tool_executor,
                    rng,
                    fault_type=scenario.fault_type,
                    extra_thought=elim,
                )
            else:
                _append_post_diagnosis_reasoning(
                    steps,
                    nid,
                    nname,
                    res,
                    tool_executor,
                    rng,
                    fault_type=scenario.fault_type,
                )

        # System elimination already merged above for last component
        if not checked:
            # Edge case: no components were checked
            thought = compose_system_inconclusive(first_name, n_check, rng)
            steps.append(TrajectoryStep(
                role="assistant", content=f"<think>{thought}</think>", thought=thought,
            ))
            first_system_all_normal = False

        last_wrong_clear = first_system_all_normal

        # Variant C: try second wrong system too
        if variant == "two_wrong" and len(wrong_candidates) >= 2:
            second_wrong = wrong_candidates[1]
            if last_wrong_clear:
                thought = compose_system_pivot(
                    first_name, second_wrong[0], second_wrong[2], rng,
                )
            else:
                thought = rng.choice([
                    (
                        f"The previous probe in {first_name} was inconclusive, "
                        f"so I will not clear it. I will inspect {second_wrong[0]} "
                        f"as another visible candidate while keeping the case open."
                    ),
                    (
                        f"{first_name} did not provide clean Normal evidence. "
                        f"I need another topology-guided check, so I will examine "
                        f"{second_wrong[0]} next."
                    ),
                    (
                        f"The evidence from {first_name} remains unresolved. "
                        f"I will broaden the search to {second_wrong[0]} without "
                        f"treating the earlier system as eliminated."
                    ),
                ])
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
            ch2_list = _leaf_components(ch2_list, tool_executor)
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
                is_uncertain2 = _is_unresolved_diagnostic_result(res2)
                actual_status2 = res2.get("status", "Normal")
                if actual_status2 == "Normal" and not is_uncertain2:
                    elim2 = compose_system_elimination(second_wrong[0], 1, rng)
                    last_wrong_clear = True
                else:
                    elim2 = compose_system_inconclusive(second_wrong[0], 1, rng)
                    last_wrong_clear = False
                # Merge ack + elimination for second wrong system
                _append_post_diagnosis_reasoning(
                    steps,
                    nid2,
                    nname2,
                    res2,
                    tool_executor,
                    rng,
                    fault_type=scenario.fault_type,
                    extra_thought=elim2,
                )
            else:
                thought = compose_system_inconclusive(second_wrong[0], 0, rng)
                steps.append(TrajectoryStep(
                    role="assistant", content=f"<think>{thought}</think>", thought=thought,
                ))
                last_wrong_clear = False
            last_wrong_name = second_wrong[0]

        # Pivot to correct system
        correct_score = next(
            (sc for _, sid, sc in scored if sid == correct_sys), 0.5
        )
        if last_wrong_clear:
            thought = compose_system_pivot(last_wrong_name, correct_name, correct_score, rng)
        else:
            thought = rng.choice([
                (
                    f"The prior system probe was unresolved rather than clean. "
                    f"I will now inspect {correct_name}, which better matches "
                    f"the symptom location and topology."
                ),
                (
                    f"I cannot close the earlier system as Normal. The next "
                    f"best topology match is {correct_name}, so I will expose "
                    f"its components."
                ),
                (
                    f"Because the previous evidence did not settle the case, "
                    f"I will pivot to {correct_name} and continue with visible "
                    f"component checks."
                ),
            ])
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

    # Now in correct system 鈥?follow path to root cause
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

            if pnode.role != "root_cause" and _node_has_children(nid, tool_executor):
                steps.extend(_query_children_step(
                    nid, discovered_nodes, queried_parents, tool_executor, rng,
                ))
                continue

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
            steps.append(TrajectoryStep(
                role="tool", content=json.dumps(res, indent=2), tool_name="diagnose_node",
            ))

            status = res.get("status", "Normal")
            if status == "Fault":
                ft = res.get("fault_type", "unknown")
                conf = res.get("confidence", 0.9)
                thought = compose_fault_found(nname, ft, conf, {}, rng)
                if scenario.scenario_type == "a_cross_system":
                    steps.extend(_generate_related_systems_impact_step(
                        correct_sys, tool_executor, rng,
                    ))
                affected = ", ".join(scenario.affected_systems)
                se = ""
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
                thought = compose_abnormal_response(nname, hint, "upstream", {}, rng)
                steps.append(TrajectoryStep(
                    role="assistant", content=f"<think>{thought}</think>", thought=thought,
                ))
            elif status == "Normal":
                _append_post_diagnosis_reasoning(
                    steps,
                    nid,
                    nname,
                    res,
                    tool_executor,
                    rng,
                    fault_type=scenario.fault_type,
                )

    return _build_trajectory(scenario, steps, is_correct=True)


def _generate_cross_system_trace_trajectory(
    scenario: FaultScenario,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
    scenario_state=None,
) -> DiagnosticTrajectory:
    """Generate a strict downstream-to-upstream cross-system SFT trajectory."""
    if scenario_state is not None:
        tool_executor.set_scenario_state(scenario_state)

    steps: List[TrajectoryStep] = []
    steps.append(TrajectoryStep(role="user", content=scenario.description))

    thought = compose_initial_reasoning(rng)
    tool_call = {"name": "get_system_overview", "arguments": {}}
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call,
        thought=thought,
    ))
    overview_result = tool_executor.execute("get_system_overview", {})
    steps.append(TrajectoryStep(
        role="tool",
        content=json.dumps(overview_result, indent=2),
        tool_name="get_system_overview",
    ))

    root_system = scenario.root_cause_system
    root_system_name = SYSTEM_DISPLAY_NAMES.get(root_system, root_system)
    root_node = scenario.root_cause_node
    root_name = root_node.split("::")[-1] if "::" in root_node else root_node
    explicit_system = _explicit_system_from_prompt(scenario.description)
    observable_system = _observable_system_from_prompt(scenario.description)
    preferred_trace_system = (
        explicit_system
        if explicit_system and explicit_system != root_system
        else observable_system
        if observable_system and observable_system != root_system
        else None
    )
    trace_node = _select_cross_trace_node(
        scenario,
        tool_executor,
        preferred_system=preferred_trace_system,
    )

    downstream_system = trace_node.system_id
    downstream_name = SYSTEM_DISPLAY_NAMES.get(downstream_system, downstream_system)

    if explicit_system == root_system:
        start_thought = compose_explicit_system_start(root_system_name, rng)
        root_children = _query_system_children(
            steps, root_system, start_thought, tool_executor,
        )
        root_discovered = {
            child.get("node_id")
            for child in root_children.get("children", [])
            if child.get("node_id")
        }
        root_queried = {f"system::{root_system}"}
        steps.extend(_ensure_node_discovered(
            root_node,
            root_discovered,
            tool_executor,
            rng,
            _queried_parents=root_queried,
        ))
        if rng.random() < 0.5:
            _append_cross_drill_correction(
                steps, root_node, root_system, tool_executor, rng,
            )
        root_result = _append_diagnose_node(
            steps, root_node, root_name, tool_executor, rng, scenario,
        )
        if not _fault_result_matches_scenario(root_result, scenario):
            raise ValueError(
                "Explicit-system cross root diagnosis is not supported by "
                f"same-node Fault tool evidence: scenario={scenario.scenario_id}, "
                f"result={root_result}"
            )

        ft = root_result.get("fault_type", scenario.fault_type)
        confidence = _confidence_value(root_result) or round(rng.uniform(0.82, 0.94), 4)
        fault_thought = (
            f"The request named {root_system_name}, and the same-node diagnostic "
            f"result confirms {root_name} has {ft} at {confidence:.1%} confidence."
        )
        direct_upstream = {"upstream": [{
            "node_id": root_node,
            "name": root_name,
            "system_id": root_system,
        }]}
        ds_result = _observed_cross_scope_result(
            root_name=root_name,
            observed_systems=[root_system],
        )
        return _append_cross_final(
            steps,
            scenario,
            root_node,
            root_name,
            root_system_name,
            ft,
            confidence,
            ds_result,
            observed_systems=[root_system],
            rng=rng,
            start_context=f"request-named {root_system_name} investigation",
        )

    start_thought = compose_cross_downstream_start(
        downstream_name,
        rng,
        visible_cue=(observable_system == downstream_system),
    )
    children_result = _query_system_children(
        steps, downstream_system, start_thought, tool_executor,
    )
    discovered_nodes = {
        child.get("node_id")
        for child in children_result.get("children", [])
        if child.get("node_id")
    }
    queried_parents = {f"system::{downstream_system}"}
    discovery_steps = _ensure_node_discovered(
        trace_node.node_id,
        discovered_nodes,
        tool_executor,
        rng,
        _queried_parents=queried_parents,
    )
    steps.extend(discovery_steps)

    local_result = _append_diagnose_node(
        steps, trace_node.node_id, trace_node.component_name,
        tool_executor, rng, scenario,
    )
    local_name = _diag_node_name(local_result, trace_node.node_id)
    if _needs_sensor_verification(local_result):
        if _has_runtime_sensor_readings(tool_executor, trace_node.node_id):
            lead = _compose_diagnosis_followup(local_name, local_result, rng)
            lead = (
                f"{lead}\nA weak local result cannot close a cross-system case; "
                f"I need same-node sensor evidence before tracing away from it."
            )
            _append_sensor_verification_steps(
                steps,
                trace_node.node_id,
                local_name,
                scenario.fault_type,
                tool_executor,
                rng,
                lead_thought=lead,
            )
        else:
            ack = _compose_diagnosis_followup(local_name, local_result, rng)
            ack = (
                f"{ack}\nThis local evidence is unresolved and same-node "
                f"runtime readings are unavailable, so I will not claim it as "
                f"sensor-confirmed. I will use cross-system topology to trace "
                f"the likely upstream source."
            )
            steps.append(TrajectoryStep(
                role="assistant",
                content=f"<think>{ack}</think>",
                thought=ack,
            ))
    else:
        ack = _compose_diagnosis_followup(local_name, local_result, rng)
        ack = (
            f"{ack}\nThis downstream evidence does not by itself prove the "
            f"root cause. I need explicit cross-system topology before changing systems."
        )
        steps.append(TrajectoryStep(
            role="assistant",
            content=f"<think>{ack}</think>",
            thought=ack,
        ))

    related_steps, related_result = _generate_related_systems_step_with_result(
        downstream_system,
        root_system,
        tool_executor,
        rng,
        target_component=trace_node.node_id,
        connected_via=root_node,
    )
    steps.extend(related_steps)
    related_anchor = _cross_anchor_for_transition(
        related_result,
        downstream_system,
        root_system,
        preferred_target_component=trace_node.node_id,
        preferred_connected_via=root_node,
    )
    anchor_target = str(related_anchor.get("target_component") or trace_node.node_id)
    anchor_connected = str(related_anchor.get("connected_via") or "")
    anchor_medium = str(related_anchor.get("medium") or "")
    trace_node_for_upstream = anchor_target or trace_node.node_id

    trace_thought = compose_upstream_anchor_trace(
        local_name, trace_node_for_upstream, rng,
    )
    tool_call = {
        "name": "get_upstream_nodes",
        "arguments": {"node_id": trace_node_for_upstream},
    }
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{trace_thought}</think>\n<tool_call>{json.dumps(tool_call)}</tool_call>",
        tool_call=tool_call,
        thought=trace_thought,
    ))
    upstream_result = tool_executor.execute(
        "get_upstream_nodes", {"node_id": trace_node_for_upstream},
    )
    steps.append(TrajectoryStep(
        role="tool",
        content=json.dumps(upstream_result, indent=2),
        tool_name="get_upstream_nodes",
    ))
    upstream_result["_related_systems"] = related_result
    upstream_result["_anchor_connected_via"] = anchor_connected
    upstream_result["_anchor_target_component"] = trace_node_for_upstream
    upstream_result["_anchor_medium"] = anchor_medium
    if anchor_connected:
        upstream_items = upstream_result.setdefault("upstream", [])
        if not any(item.get("node_id") == anchor_connected for item in upstream_items):
            upstream_items.append({
                "node_id": anchor_connected,
                "name": _node_display_name(anchor_connected),
                "system_id": _system_id_from_node(anchor_connected),
                "relation": "cross_system",
                "medium": anchor_medium,
                "anchor_source": "get_related_systems.connected_via",
            })

    upstream_systems = {
        item.get("system_id")
        for item in upstream_result.get("upstream", [])
        if item.get("system_id")
    }
    if root_system not in upstream_systems:
        raise ValueError(
            f"Cross-system upstream trace does not expose root system: "
            f"scenario={scenario.scenario_id}, root={root_system}, result={upstream_result}"
        )

    ack = compose_upstream_ack(
        downstream_name,
        root_system_name,
        rng,
        connected_via=anchor_connected or None,
        target_component=trace_node_for_upstream or None,
    )
    steps.append(TrajectoryStep(
        role="assistant",
        content=f"<think>{ack}</think>",
        thought=ack,
    ))

    upstream_node_ids = {
        item.get("node_id")
        for item in upstream_result.get("upstream", [])
        if item.get("node_id")
    }
    root_visible_from_trace = root_node in upstream_node_ids
    if anchor_connected and anchor_connected == root_node:
        root_ack = (
            f"The related-system anchor connected_via={anchor_connected} is the "
            f"plant-side candidate for this cross-system link. I should diagnose "
            f"that exact node now rather than inventing a {root_system_name} "
            f"component name."
        )
        root_visible_from_trace = True
    elif root_visible_from_trace:
        root_ack = (
            f"The upstream trace directly exposes {root_name} as a "
            f"{root_system_name} candidate. Since this node is already visible "
            f"from the causal trace, I should diagnose it now instead of "
            f"expanding another hierarchy."
        )
    elif anchor_connected and _system_id_from_node(anchor_connected) == root_system:
        anchor_ack = (
            f"The related-system anchor gives plant-side connected_via="
            f"{anchor_connected}. It is not the final root node label, so I will "
            f"use this visible topology evidence to reveal the candidate root "
            f"without sweeping unrelated siblings."
        )
        steps.append(TrajectoryStep(
            role="assistant",
            content=f"<think>{anchor_ack}</think>",
            thought=anchor_ack,
        ))
        _append_cross_root_discovery_steps(
            steps,
            root_node=root_node,
            root_system=root_system,
            root_system_name=root_system_name,
            anchor_connected=anchor_connected,
            tool_executor=tool_executor,
            rng=rng,
        )
    else:
        _append_cross_root_discovery_steps(
            steps,
            root_node=root_node,
            root_system=root_system,
            root_system_name=root_system_name,
            anchor_connected=anchor_connected,
            tool_executor=tool_executor,
            rng=rng,
        )

    if root_visible_from_trace:
        steps.append(TrajectoryStep(
            role="assistant",
            content=f"<think>{root_ack}</think>",
            thought=root_ack,
        ))

    # ~50% of cross-system trajectories include a Normal-then-drill correction:
    # diagnose a plant sibling/aggregate first (soft gate -> Abnormal+suggested),
    # then drill to the exact root. This teaches the policy to not conclude at
    # the first plant node the topology exposes. The helper self-guards (returns
    # False when no useful sibling signal exists), so this is always safe.
    if rng.random() < 0.5:
        _append_cross_drill_correction(
            steps, root_node, root_system, tool_executor, rng,
        )

    root_result = _append_diagnose_node(
        steps, root_node, root_name, tool_executor, rng, scenario,
    )
    if not _fault_result_matches_scenario(root_result, scenario):
        raise ValueError(
            "Cross-system root diagnosis is not supported by same-node Fault "
            f"tool evidence: scenario={scenario.scenario_id}, result={root_result}"
        )

    ft = root_result.get("fault_type", scenario.fault_type)
    confidence = _confidence_value(root_result) or round(rng.uniform(0.82, 0.94), 4)
    fault_thought = (
        f"The downstream symptom was traced through explicit topology to "
        f"{root_system_name}. The same-node diagnostic result confirms "
        f"{root_name} has {ft} at {confidence:.1%} confidence."
    )

    ds_result = _observed_cross_scope_result(
        root_name=root_name,
        observed_systems=[downstream_system, root_system],
    )
    return _append_cross_final(
        steps,
        scenario,
        root_node,
        root_name,
        root_system_name,
        ft,
        confidence,
        ds_result,
        observed_systems=[downstream_system, root_system],
        rng=rng,
        start_context=(
            f"candidate symptoms in {downstream_name} traced upstream to "
            f"{root_system_name}"
        ),
    )


def _generate_low_confidence_trajectory(
    scenario: FaultScenario,
    tool_executor: UnifiedToolExecutor,
    rng: random.Random,
    scenario_state=None,
) -> DiagnosticTrajectory:
    """
    Generate trajectory for low-confidence scenarios: sensor cross-validation.

    diagnose_node returns Warning (not Fault) for root cause. Agent must:
    1. Navigate to correct system (prompt names it)
    2. Get Warning from diagnose_node
    3. Call get_node_sensors to check raw data
    4. Fuse Warning and sensor evidence to support, not over-confirm, diagnosis
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

    # Phase 3: select the system named or implied by the prompt.
    start_system = (
        _explicit_system_from_prompt(scenario.description)
        or _observable_system_from_prompt(scenario.description)
        or scenario.root_cause_system
    )
    start_name = SYSTEM_DISPLAY_NAMES.get(start_system, start_system)

    systems_info = overview_result.get("systems", [])
    sys_score = next(
        (s.get("anomaly_score", 0.5) for s in systems_info if s.get("system_id") == start_system),
        0.5,
    )
    if _observable_system_from_prompt(scenario.description) == start_system:
        thought = compose_observable_system_start(
            start_name, rng, context="low_confidence",
        )
    else:
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

    # Phase 4: Optional elimination of 1-2 off-path nodes
    if path:
        steps.extend(_generate_elimination_steps(
            scenario, start_system, path, tool_executor, rng,
            n_steps=rng.randint(0, 1),
        ))

    # Phase 5: Diagnose root cause 鈫?get Warning (not Fault)
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
        }
    steps.append(TrajectoryStep(
        role="tool", content=json.dumps(diag_result, indent=2), tool_name="diagnose_node",
    ))

    # Phase 6: conditionally acquire same-node sensor evidence.
    sensor_evidence = ""
    if _needs_sensor_verification(diag_result):
        if not _has_runtime_sensor_readings(tool_executor, root_node):
            raise ValueError(
                "Low-confidence root candidate has no runtime sensor readings "
                f"for same-node verification: scenario={scenario.scenario_id}, "
                f"node={root_node}"
            )
        if _status_lower(diag_result) == "warning":
            lead_thought = compose_warning_response(
                root_name, _confidence_value(diag_result), {}, rng,
            )
        else:
            lead_thought = _compose_diagnosis_followup(root_name, diag_result, rng)
            lead_thought = rng.choice([
                (
                    f"{lead_thought}\nThis root-candidate evidence is uncertain, "
                    f"so I need same-node sensor readings before concluding."
                ),
                (
                    f"{lead_thought}\nI will not close a low-confidence root "
                    f"candidate until a same-node sensor check supports it."
                ),
                (
                    f"{lead_thought}\nThe candidate remains open, but it needs "
                    f"sensor-level support on the same node before diagnosis."
                ),
            ])
        sensor_evidence = _append_sensor_verification_steps(
            steps,
            root_node,
            root_name,
            scenario.fault_type,
            tool_executor,
            rng,
            lead_thought=lead_thought,
        )
        thought_options = [
            (
                f"{root_name} remains the best-supported candidate because the "
                f"Warning and same-node sensor readings point to the same node."
            ),
            (
                f"The evidence is still borderline, but the same-node sensor "
                f"check supports {root_name} as the responsible component."
            ),
            (
                f"I have same-node evidence for {root_name}: a Warning result "
                f"plus available readings that support the candidate fault."
            ),
            (
                f"The sensor check does not turn the Warning into a definitive "
                f"Fault, but it is enough to support {root_name} as the likely "
                f"root candidate."
            ),
        ]
        thought = rng.choice(thought_options)
        confidence = round(rng.uniform(0.64, 0.72), 4)
        ft = scenario.fault_type
    elif _fault_result_matches_scenario(diag_result, scenario):
        ft = diag_result.get("fault_type", scenario.fault_type)
        confidence = _confidence_value(diag_result) or round(rng.uniform(0.82, 0.92), 4)
        thought = (
            f"The diagnostic tool returned a definitive Fault for {root_name}: "
            f"{ft} at {confidence:.1%} confidence. Since this is already "
            f"same-node fault evidence, a post-hoc sensor call is unnecessary."
        )
    else:
        raise ValueError(
            "Low-confidence trajectory root evidence is neither uncertain nor "
            f"GT-matching Fault: scenario={scenario.scenario_id}, result={diag_result}"
        )
    affected = ", ".join(scenario.affected_systems)
    final = compose_final_diagnosis(root_node, ft, affected, sensor_evidence, rng)

    # Deduplicate: if the final summary is too similar to the think block, shorten it
    if sensor_evidence and sensor_evidence in thought and sensor_evidence in final:
        # Avoid repeating the exact same evidence string in both think + summary
        final = final.replace(f" Critical evidence: {sensor_evidence}.", "")
        final = final.replace(f" Key diagnostic indicators: {sensor_evidence}.", "")
        final = final.replace(f" The diagnosis is grounded in: {sensor_evidence}.", "")

    diagnosis = {
        "root_cause_node": root_node,
        "fault_type": ft,
        "confidence": confidence,
        "affected_systems": scenario.affected_systems,
    }
    sensor_status = _visible_sensor_evidence_status(steps, root_node)
    if _status_lower(diag_result) == "warning" and sensor_status != "available":
        raise ValueError(
            "Low-confidence final diagnosis lacks explicit available "
            f"same-node sensor evidence: scenario={scenario.scenario_id}, "
            f"node={root_node}, sensor_status={sensor_status}"
        )
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
    diagnostic_path_nodes = path.node_ids if path else []
    symptom_node = path.symptom_node if path else ""
    optimal_path_length = path.path_length if path else len([p for p in scenario.optimal_path if p])

    # Reference propagation edges (TR metric, eq:exp_tr).
    # The diagnostic path is ordered symptom -> ... -> root_cause, so each
    # consecutive node pair is a key propagation/topology relation the agent is
    # expected to traverse. Storing edges (not just nodes) lets TR measure
    # relation coverage instead of mere endpoint visitation.
    reference_propagation_edges = [
        [diagnostic_path_nodes[i], diagnostic_path_nodes[i + 1]]
        for i in range(len(diagnostic_path_nodes) - 1)
    ]

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
    tool_faithful = _diagnosis_matches_tool_evidence(
        final_diagnosis, steps, is_no_fault,
    )
    final_confidence_supported = _final_confidence_supported(
        final_diagnosis, scenario.scenario_type, is_no_fault,
    )
    label_consistent = label_consistent and final_confidence_supported

    return DiagnosticTrajectory(
        scenario_id=scenario.scenario_id,
        scenario_type=scenario.scenario_type,
        steps=steps,
        ground_truth={
            "root_cause_system": scenario.root_cause_system,
            "root_cause_node": gt_root_node,
            "fault_type": gt_fault_type,
            "fault_intensity": "none" if is_no_fault else scenario.fault_intensity,
            "optimal_path_length": optimal_path_length or 3,
            "diagnostic_path_nodes": diagnostic_path_nodes,
            "symptom_node": symptom_node,
            "reference_review_nodes": diagnostic_path_nodes if is_no_fault else [],
            "reference_propagation_edges": reference_propagation_edges,
        },
        metadata={
            "n_tool_calls": sum(1 for s in steps if s.tool_call is not None),
            "difficulty": scenario.difficulty,
            "path_length": optimal_path_length or 0,
            "diagnostic_path_nodes": diagnostic_path_nodes,
            "reference_propagation_edges": reference_propagation_edges,
            "symptom_node": symptom_node,
            "affected_systems": list(scenario.affected_systems),
            "source_file": scenario.source_file,
            "time_window_start": scenario.time_window_start,
            "time_window_end": scenario.time_window_end,
            "trajectory_quality": "correct" if is_correct else "incorrect",
            "tool_faithful": tool_faithful,
            "label_consistent": label_consistent,
            "final_confidence_supported": final_confidence_supported,
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

    text = str(value or "").strip().lower()
    text = re.sub(r"(^|[_\s])\+(?=\d)", r"\1pos_", text)
    text = re.sub(r"(^|[_\s])-(?=\d)", r"\1neg_", text)
    text = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return re.sub(r"_+", "_", text)


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
    diag_fault = diagnosis.get("fault_type", "")
    gt_fault = gt_fault_type

    if is_no_fault:
        diag_fault_norm = _norm_label(diag_fault)
        status = _norm_label(diagnosis.get("status", ""))
        return (
            diag_root.lower() in ("none", "")
            and diag_fault_norm in ("no_fault", "normal", "none")
            and status in ("normal", "")
        )

    return diag_root == gt_root_node and fault_exact_match(gt_fault, diag_fault)


def _final_confidence_value(diagnosis: Dict[str, Any]) -> float:
    try:
        return float(diagnosis.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _final_confidence_supported(
    diagnosis: Dict[str, Any],
    scenario_type: str,
    is_no_fault: bool,
) -> bool:
    """Reject ordinary fault trajectories with weak final-confidence labels."""
    if not diagnosis:
        return False
    if is_no_fault:
        return True
    if "low_confidence" in str(scenario_type).lower():
        return True
    return _final_confidence_value(diagnosis) >= MIN_ORDINARY_FAULT_FINAL_CONFIDENCE


def _diagnosis_matches_tool_evidence(
    diagnosis: Dict[str, Any],
    steps: List[TrajectoryStep],
    is_no_fault: bool,
) -> bool:
    """Check that final diagnosis does not contradict definitive tool outputs.

    Real-Oracle data generation can encounter cases where the model predicts a
    different signed intensity than the scenario label. Those samples are not
    useful SFT supervision: the final answer may match GT, but it teaches the
    agent to ignore the tool result. Keep uncertain/Warning + sensor-verification
    trajectories, but reject definitive Fault observations that disagree with
    the final diagnosis.
    """
    if not diagnosis:
        return False

    diag_root = str(diagnosis.get("root_cause_node", ""))
    diag_fault = diagnosis.get("fault_type", "")

    saw_fault_observation = False
    saw_matching_root_fault = False

    for step in steps:
        if step.role != "tool" or step.tool_name != "diagnose_node":
            continue
        try:
            obs = json.loads(step.content)
        except (json.JSONDecodeError, TypeError):
            continue

        status = str(obs.get("status", "")).strip().lower()
        obs_node = str(obs.get("node_id", ""))
        obs_fault = obs.get("fault_type", "")

        if status != "fault":
            continue

        saw_fault_observation = True

        if is_no_fault:
            return False

        if obs_node == diag_root:
            if not fault_exact_match(obs_fault, diag_fault):
                return False
            saw_matching_root_fault = True

    if saw_matching_root_fault:
        return True
    if saw_fault_observation:
        # A definitive non-root Fault without a later same-node root Fault is
        # unsupported. In cross-system traces, however, downstream Faults can be
        # valid symptom evidence once a same-node root Fault closes the chain.
        return False

    # Warning/uncertain trajectories can be faithful without a definitive Fault
    # observation when explicit same-node sensor verification is present; those
    # checks are enforced by generation and validation gates.
    return True


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
            When provided, runtime model predictions are used.
        seed: Random seed for reproducibility.

    Returns:
        List of generated trajectories.
    """
    rng = random.Random(seed)
    trajectories = []
    failed = 0
    skipped_inconsistent = 0
    inconsistent_samples = []

    for i, scenario in enumerate(scenarios):
        try:
            state = None
            if scenario_states:
                state = scenario_states.get(scenario.scenario_id)
            traj = generate_trajectory(
                scenario, tool_executor, rng, scenario_state=state,
            )
            if not (
                traj.metadata.get("label_consistent", False)
                and traj.metadata.get("tool_faithful", False)
            ):
                skipped_inconsistent += 1
                if len(inconsistent_samples) < 20:
                    inconsistent_samples.append({
                        "scenario_id": scenario.scenario_id,
                        "ground_truth": traj.ground_truth,
                        "final_diagnosis": traj.metadata.get("final_diagnosis", {}),
                        "label_consistent": traj.metadata.get("label_consistent", False),
                        "tool_faithful": traj.metadata.get("tool_faithful", False),
                    })
                continue
            trajectories.append(traj)
        except Exception as e:
            failed += 1
            logger.warning(f"[{scenario.scenario_id}] Generation failed: {e}")
            if failed > max(50, len(scenarios) * 0.35):
                logger.error("Too many failures, stopping batch generation")
                break

        if (i + 1) % 100 == 0 or (i + 1) == len(scenarios):
            logger.info(f"Generated {i + 1}/{len(scenarios)} trajectories")

    logger.info(
        "Generated %s trajectories total (failed: %s, skipped_inconsistent: %s)",
        len(trajectories), failed, skipped_inconsistent,
    )
    if inconsistent_samples:
        logger.info(
            "Label-inconsistent trajectory samples (first %s): %s",
            len(inconsistent_samples),
            inconsistent_samples,
        )
    return trajectories
