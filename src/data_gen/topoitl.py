"""TopoITL structured state recovery for tool-interactive trajectories.

The formatter keeps the normal ShareGPT/ReAct transcript, but prepends a
compact supervised state/action block to assistant decisions.  The labels are
recovered by replaying visible tool observations only; the full topology or
ground-truth label is not used as policy input.
"""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple


TOPOITL_TAG_STATE = "topoitl_state"
TOPOITL_TAG_ACTION = "topoitl_action"
TOPOITL_VERSION = "topoitl_structured_v2026_05_30"

ACTION_PRIMITIVES = {"ground", "expand", "trace", "verify", "conclude"}


def _norm(value: Any) -> str:
    return str(value or "").strip().lower()


def _node_system(node_id: str) -> str:
    text = str(node_id or "")
    if "::" in text and not text.startswith("system::"):
        return text.split("::", 1)[0]
    if text.startswith("system::"):
        return text.split("::", 1)[1]
    return ""


def _clip_list(items: Iterable[Any], limit: int) -> List[Any]:
    raw = list(items)
    if limit > 0 and len(raw) > limit:
        raw = raw[-limit:]
    result = []
    seen = set()
    for item in raw:
        key = json.dumps(item, sort_keys=True, ensure_ascii=False) if isinstance(item, dict) else str(item)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
        if limit > 0 and len(result) >= limit:
            break
    return result


def _parse_json(text: Any) -> Optional[Dict[str, Any]]:
    if isinstance(text, dict):
        return text
    if not isinstance(text, str):
        return None
    try:
        parsed = json.loads(text)
    except (TypeError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    """Extract the first XML-wrapped tool call from an assistant turn."""
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text or "", re.DOTALL)
    if not match:
        return None
    return _parse_json(match.group(1))


def extract_final_diagnosis(text: str) -> Optional[Dict[str, Any]]:
    """Extract final diagnosis JSON from an assistant turn."""
    match = re.search(r"<diagnosis>\s*(\{.*?\})\s*</diagnosis>", text or "", re.DOTALL)
    if not match:
        return None
    return _parse_json(match.group(1))


def classify_action(tool_call: Optional[Dict[str, Any]], diagnosis: Optional[Dict[str, Any]] = None) -> str:
    """Map a concrete tool call or terminal output to a diagnostic primitive."""
    if diagnosis is not None:
        return "conclude"
    name = _norm((tool_call or {}).get("name"))
    if name == "get_system_overview":
        return "ground"
    if name == "get_node_children":
        return "expand"
    if name in {"get_upstream_nodes", "get_downstream_nodes", "get_related_systems"}:
        return "trace"
    if name in {"diagnose_node", "get_node_sensors"}:
        return "verify"
    return "verify"


@dataclass
class TopoITLState:
    """Compact visible diagnosis state used as TopoITL supervision."""

    visible_nodes: List[str] = field(default_factory=list)
    visible_edges: List[Dict[str, str]] = field(default_factory=list)
    symptom_systems: List[str] = field(default_factory=list)
    symptom_nodes: List[str] = field(default_factory=list)
    candidate_roots: List[Dict[str, Any]] = field(default_factory=list)
    propagation_frontier: List[Dict[str, Any]] = field(default_factory=list)
    checked_nodes: List[Dict[str, Any]] = field(default_factory=list)

    def copy(self) -> "TopoITLState":
        return copy.deepcopy(self)

    def add_node(self, node_id: str) -> None:
        node_id = str(node_id or "").strip()
        if node_id and node_id not in self.visible_nodes:
            self.visible_nodes.append(node_id)

    def add_edge(self, source: str, target: str, relation: str = "", medium: str = "") -> None:
        source = str(source or "").strip()
        target = str(target or "").strip()
        if not source or not target:
            return
        self.add_node(source)
        self.add_node(target)
        edge = {
            "source": source,
            "target": target,
            "relation": str(relation or "").strip(),
            "medium": str(medium or "").strip(),
        }
        if edge not in self.visible_edges:
            self.visible_edges.append(edge)

    def add_symptom_system(self, system_id: str) -> None:
        system_id = str(system_id or "").strip()
        if system_id and system_id not in self.symptom_systems:
            self.symptom_systems.append(system_id)

    def add_symptom_node(self, node_id: str) -> None:
        node_id = str(node_id or "").strip()
        if node_id and node_id not in self.symptom_nodes:
            self.symptom_nodes.append(node_id)
        sys_id = _node_system(node_id)
        if sys_id:
            self.add_symptom_system(sys_id)

    def add_candidate(self, node_id: str, status: str, fault_type: str = "", confidence: Any = None) -> None:
        node_id = str(node_id or "").strip()
        if not node_id:
            return
        self.add_node(node_id)
        item = {
            "node_id": node_id,
            "status": str(status or "").strip(),
            "fault_type": str(fault_type or "None").strip(),
        }
        try:
            item["confidence"] = round(float(confidence), 4)
        except (TypeError, ValueError):
            pass
        existing = [c for c in self.candidate_roots if c.get("node_id") != node_id]
        existing.append(item)
        self.candidate_roots = existing

    def remove_candidate(self, node_id: str) -> None:
        node_id = str(node_id or "").strip()
        if node_id:
            self.candidate_roots = [
                c for c in self.candidate_roots
                if c.get("node_id") != node_id
            ]

    def add_frontier(self, node_id: str, direction: str = "", reason: str = "") -> None:
        node_id = str(node_id or "").strip()
        if not node_id:
            return
        self.add_node(node_id)
        item = {
            "node_id": node_id,
            "direction": str(direction or "").strip(),
            "reason": str(reason or "").strip(),
        }
        if item not in self.propagation_frontier:
            self.propagation_frontier.append(item)

    def add_checked(self, node_id: str, status: str, confidence: Any = None) -> None:
        node_id = str(node_id or "").strip()
        if not node_id:
            return
        self.add_node(node_id)
        item = {"node_id": node_id, "status": str(status or "").strip()}
        try:
            item["confidence"] = round(float(confidence), 4)
        except (TypeError, ValueError):
            pass
        self.checked_nodes = [c for c in self.checked_nodes if c.get("node_id") != node_id]
        self.checked_nodes.append(item)

    def to_label(self) -> Dict[str, Any]:
        """Return a deterministic compact JSON-serializable state label."""
        return {
            "G_obs": {
                "nodes": _clip_list(self.visible_nodes, 8),
                "edges": _clip_list(self.visible_edges, 8),
            },
            "symptom_grounding": {
                "systems": _clip_list(self.symptom_systems, 4),
                "nodes": _clip_list(self.symptom_nodes, 4),
            },
            "candidate_roots": _clip_list(self.candidate_roots, 4),
            "propagation_frontier": _clip_list(self.propagation_frontier, 4),
            "checked_nodes": _clip_list(self.checked_nodes, 4),
        }


def initialize_state(user_text: str = "") -> TopoITLState:
    """Initialize visible state from the user prompt without using GT labels."""
    state = TopoITLState()
    text = str(user_text or "")

    bracket = re.match(r"\[([^\]]+)\]", text.strip())
    if bracket:
        display = bracket.group(1).strip().lower()
        aliases = {
            "chiller plant": "chiller_plant",
            "boiler plant": "boiler_plant",
            "single-duct ahu": "sdahu",
            "dual-duct ahu": "ddahu",
            "rooftop unit": "rtu",
            "fan coil unit": "fcu",
            "parallel fan powered unit": "pfpu",
            "series fan powered unit": "sfpu",
        }
        if display in aliases:
            state.add_symptom_system(aliases[display])

    lowered = text.lower()
    keyword_aliases = {
        "chilled water": "chiller_plant",
        "cooling": "chiller_plant",
        "hot water": "boiler_plant",
        "heating": "boiler_plant",
        "ahu": "sdahu",
        "fan coil": "fcu",
        "rooftop": "rtu",
    }
    for keyword, system_id in keyword_aliases.items():
        if keyword in lowered:
            state.add_symptom_system(system_id)

    return state


def update_state_from_observation(
    state: TopoITLState,
    tool_name: str,
    arguments: Optional[Dict[str, Any]],
    observation: Any,
) -> TopoITLState:
    """Apply a visible tool observation to the TopoITL state."""
    new_state = state.copy()
    args = arguments or {}
    result = _parse_json(observation) or {}
    tool = _norm(tool_name)

    if tool == "get_system_overview":
        for sys_info in result.get("systems", []) or []:
            node_id = sys_info.get("node_id") or f"system::{sys_info.get('system_id', '')}"
            new_state.add_node(node_id)
        return new_state

    if tool == "get_node_children":
        parent = result.get("parent_node") or args.get("node_id")
        new_state.add_node(parent)
        for child in result.get("children", []) or []:
            child_id = child.get("node_id")
            new_state.add_edge(parent, child_id, relation="has_part")
            if new_state.symptom_systems and _node_system(child_id) in new_state.symptom_systems:
                new_state.add_frontier(child_id, "check_component", "child_of_symptom_system")
        return new_state

    if tool in {"get_upstream_nodes", "get_downstream_nodes"}:
        anchor = (
            result.get("target_node")
            or result.get("source_node")
            or args.get("node_id")
        )
        key = "upstream" if tool == "get_upstream_nodes" else "downstream"
        direction = "upstream" if key == "upstream" else "downstream"
        new_state.add_node(anchor)
        for item in result.get(key, []) or []:
            node_id = item.get("node_id")
            if key == "upstream":
                new_state.add_edge(node_id, anchor, item.get("relation", ""), item.get("medium", ""))
            else:
                new_state.add_edge(anchor, node_id, item.get("relation", ""), item.get("medium", ""))
            new_state.add_frontier(node_id, direction, "topology_trace")
        return new_state

    if tool == "get_related_systems":
        system_id = result.get("system_id") or args.get("system_id")
        if system_id:
            new_state.add_node(f"system::{system_id}")
        for key, direction in (
            ("upstream_connections", "upstream"),
            ("downstream_connections", "downstream"),
        ):
            for item in result.get(key, []) or []:
                src = item.get("connected_via")
                dst = item.get("target_component")
                if key == "upstream_connections":
                    new_state.add_edge(src, dst, "cross_system", item.get("medium", ""))
                    new_state.add_frontier(src, direction, "related_system")
                else:
                    new_state.add_edge(src, dst, "cross_system", item.get("medium", ""))
                    new_state.add_frontier(dst, direction, "related_system")
        return new_state

    if tool == "diagnose_node":
        node_id = result.get("node_id") or args.get("node_id")
        status = str(result.get("status") or "").strip()
        confidence = result.get("confidence")
        new_state.add_checked(node_id, status, confidence)
        status_l = status.lower()
        if status_l in {"fault", "warning", "abnormal"}:
            new_state.add_candidate(
                node_id,
                status,
                fault_type=result.get("fault_type", "None"),
                confidence=confidence,
            )
            direction = str(result.get("suggested_direction") or "").strip()
            if direction:
                new_state.add_frontier(node_id, direction, "diagnostic_hint")
        elif status_l == "normal":
            new_state.remove_candidate(node_id)
        return new_state

    if tool == "get_node_sensors":
        node_id = result.get("component_node") or args.get("node_id")
        new_state.add_node(node_id)
        if result.get("readings_available") is True:
            new_state.add_checked(node_id, "sensor_verified", 1.0)
        return new_state

    return new_state


def action_label(tool_call: Optional[Dict[str, Any]], diagnosis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return the structured action label used by TopoITL."""
    primitive = classify_action(tool_call, diagnosis)
    if diagnosis is not None:
        return {
            "primitive": primitive,
            "type": "final_diagnosis",
            "diagnosis": diagnosis,
        }
    return {
        "primitive": primitive,
        "type": "tool_call",
        "tool": (tool_call or {}).get("name", ""),
        "arguments": (tool_call or {}).get("arguments", {}),
    }


def render_supervision_suffix(
    state: TopoITLState,
    action: Dict[str, Any],
) -> str:
    """Render compact trainable TopoITL labels AFTER the executable action.

    Action-first layout (deployment-viable):

        <think> ... </think>
        <tool_call>{...}</tool_call>            # or <diagnosis>{...}</diagnosis>
        <topoitl_state>{...}</topoitl_state>     # trailing training-only labels
        <topoitl_action>{...}</topoitl_action>

    The TopoITL state ``z_t`` and structured action ``ã_t`` remain *target tokens*
    supervised by the three-term loss (eq:topoitl_loss). They are placed AFTER
    the executable block so that:

    * Multi-turn RL rollouts / evaluation stop generation at ``</tool_call>`` /
      ``</diagnosis>`` and never have to serialize the (growing) state JSON at
      inference — this keeps per-turn generation short and the context bounded,
      which the state-first layout could not (it caused 8k+ token prompts,
      ~200s/turn rollouts, and truncated conclusion turns).
    * The conditioning ``p(ã_t | z_t)`` is still learned: the model sees the
      state/action tokens as supervised targets within the same turn, and the
      next turn's leading context contains the prior observation ``o_t`` so the
      next state block serves as the ℓ_tr (transition) target.

    Inference must NOT emit these labels; the deployed policy stops at the
    executable block. They exist only as SFT supervision and offline audit.
    """
    state_json = json.dumps(state.to_label(), ensure_ascii=False, sort_keys=True)
    action_json = json.dumps(action, ensure_ascii=False, sort_keys=True)
    return (
        f"<{TOPOITL_TAG_STATE}>{state_json}</{TOPOITL_TAG_STATE}>\n"
        f"<{TOPOITL_TAG_ACTION}>{action_json}</{TOPOITL_TAG_ACTION}>"
    )


# Backwards-compatible alias; some callers import ``render_supervision_prefix``.
render_supervision_prefix = render_supervision_suffix


def annotate_sharegpt_conversations(
    conversations: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Annotate assistant decisions with TopoITL state/action labels.

    Returns the annotated conversations and a transition trace stored in sample
    metadata for validation and analysis.
    """
    annotated: List[Dict[str, Any]] = []
    transitions: List[Dict[str, Any]] = []
    state = TopoITLState()
    last_tool_call: Optional[Dict[str, Any]] = None
    pending_before: Optional[Dict[str, Any]] = None
    pending_action: Optional[Dict[str, Any]] = None
    first_user_seen = False

    for conv in conversations:
        role = conv.get("from")
        value = str(conv.get("value", ""))

        if role == "human" and not first_user_seen:
            state = initialize_state(value)
            first_user_seen = True
            annotated.append(conv)
            continue

        if role == "observation":
            before_obs = state.to_label()
            tool_name = str((last_tool_call or {}).get("name") or "")
            arguments = (last_tool_call or {}).get("arguments", {})
            state = update_state_from_observation(state, tool_name, arguments, value)
            if pending_before is not None and pending_action is not None:
                transitions.append({
                    "state": pending_before,
                    "action": pending_action,
                    "observation_tool": tool_name,
                    "state_observed_before_update": before_obs,
                    "state_next": state.to_label(),
                })
            pending_before = None
            pending_action = None
            annotated.append(conv)
            continue

        if role == "gpt":
            tool_call = extract_tool_call(value)
            diagnosis = extract_final_diagnosis(value)
            if tool_call is not None or diagnosis is not None:
                before = state.to_label()
                action = action_label(tool_call, diagnosis)
                suffix = render_supervision_suffix(state, action)
                conv = dict(conv)
                # Action-first: executable block stays FIRST/terminal-visible;
                # state/action labels are appended as trailing training-only
                # supervision. Inference stops at the executable block and never
                # emits these labels, keeping multi-turn rollouts short.
                conv["value"] = f"{value}\n{suffix}"
                if diagnosis is not None:
                    transitions.append({
                        "state": before,
                        "action": action,
                        "state_next": before,
                    })
                    pending_before = None
                    pending_action = None
                    last_tool_call = None
                else:
                    pending_before = before
                    pending_action = action
                    last_tool_call = tool_call
            annotated.append(conv)
            continue

        annotated.append(conv)

    return annotated, transitions
