"""
SFT Formatter - Convert diagnostic trajectories to Qwen2.5 SFT format.

Produces training data in the ShareGPT/Qwen conversational format with:
  - System prompt with tool definitions
  - Multi-turn conversations with <|im_start|>/<|im_end|> tokens
  - Loss masking metadata (only compute loss on assistant turns)
  - Tool call formatting with <tool_call> tags
"""

import json
import os
import hashlib
import re
from typing import Any, Dict, List, Optional

from src.data_gen.topoitl import (
    TOPOITL_VERSION,
    annotate_sharegpt_conversations,
)
from src.data_gen.trajectory_generator import DiagnosticTrajectory, TrajectoryStep
from src.environment.tool_executor import build_tool_schemas
from src.utils.io_utils import save_json, setup_logger

logger = setup_logger(__name__)

SFT_DESIGN_VERSION = "context_preserved_terminal_closure_v2026_06_03"
MAX_PRE_ACTION_THINK_CHARS = 360
REQUIRED_SYSTEM_PROMPT_MARKERS = (
    "verify that same node with get_node_sensors",
    "Do not sweep every node blindly",
    "provide a final diagnosis immediately",
    "no-fault diagnosis requires",
    "Never guess or fabricate information",
    "Use node_id values exactly as returned by tools",
    "connected_via and target_component are authoritative",
    "close the diagnosis immediately",
    "same-node Fault evidence",
    "Do not output TopoITL labels",
    "Never use a normal component as the root cause of no-fault",
    "RTU is self-contained",
    "When evidence is closed, stop calling tools",
    "Do not call diagnose_node on the first turn",
)

# System prompt template - base protocol
_SYSTEM_PROMPT_BASE = """You are an expert HVAC fault diagnosis agent for building automation systems. Your role is to systematically diagnose faults in HVAC systems by investigating the building topology, checking component statuses, and tracing fault propagation paths to identify root causes.

## Available Tools
You have access to the following diagnostic tools:

{tool_schemas}

## Diagnostic Protocol
1. Start by getting an overview of the building's HVAC systems and topology.
2. Do not call diagnose_node on the first turn of a new user request. First obtain get_system_overview, then use exact system/node identifiers returned by topology tools.
3. Based on reported symptoms, topology, and any available tool evidence, identify the most likely affected system(s).
4. Drill into the system hierarchy with topology tools before diagnosing specific components.
5. Use the diagnose_node tool to check one component at a time.
6. For cross-system symptoms, use get_related_systems or upstream/downstream tools before changing systems.
7. Do not change to an upstream plant merely because a local candidate is Normal; cross-system tracing requires either a cross-system user cue, an Abnormal/suggested_direction result, or topology evidence that the checked component is supplied by another system.
8. For get_related_systems results, connected_via and target_component are authoritative visible anchors. If the target_component is the local symptom-side component, inspect or trace that exact node; if connected_via is the upstream plant component, diagnose that exact connected_via node. Do not invent a plant node by copying a downstream component name into another system.
9. In cross-system cases, once topology has linked a downstream symptom to an upstream plant candidate and diagnose_node returns same-node Fault on that candidate/root node, close the diagnosis immediately. Do not run extra downstream impact queries or continue sweeping siblings after same-node Fault evidence is sufficient.
10. If get_upstream_nodes directly returns a central plant root-cause candidate, diagnose that returned candidate immediately instead of expanding another hierarchy first.
11. If get_upstream_nodes is empty but get_related_systems exposed an upstream connected_via candidate for the same medium/system, diagnose that connected_via node rather than repeating the empty upstream query.
12. If diagnose_node returns same-node Fault on the current best root candidate, the next assistant response must be <diagnosis>...</diagnosis>. Do not call another tool after root Fault evidence is closed.
13. If a component is Abnormal, use its suggested_direction plus upstream/downstream topology to continue tracing.
14. If a fault is found in a local single-system case, trace upstream only when the visible topology or suggested_direction requires it; in cross-system cases, do not add impact-scope calls after the upstream root candidate has same-node Fault evidence.
15. If diagnose_node returns Warning on a candidate node, verify that same node with get_node_sensors before concluding. After same-node sensor evidence supports the Warning, close with a calibrated diagnosis instead of sweeping unrelated components.
16. Treat low-confidence Normal, unknown, and data_unavailable as unresolved evidence, not proof of normal operation; pivot using the current visible topology. Do not use an unavailable node as a reason to jump to unrelated systems.
17. Fuse evidence by consistency: definitive diagnose_node results, topology direction, and explicit sensor verification should agree before you conclude.
18. Use node_id values exactly as returned by tools. Do not append parent names, duplicate segments, or invent hierarchy paths. If a tool returns no children, do not create deeper child node_ids.
19. If a tool returns status=error, unknown, or data_unavailable for a node, do not keep drilling into that same invented or unavailable node. Pivot to a valid sibling, upstream/downstream node, related system, or provide an uncertainty-aware diagnosis if evidence is sufficient.
20. Do not sweep every node blindly. Avoid repeating the same tool on the same node. If the best-supported candidate has been checked and the evidence is sufficient, stop exploring and answer.
21. When the current evidence identifies a root-cause candidate, provide a final diagnosis immediately instead of continuing to call tools until the step limit.
22. A no-fault diagnosis requires explicit closure: either at least two relevant diagnose_node observations return high-confidence Normal, or every visible candidate in a single-candidate system has been checked. Low-confidence Normal, unknown, data_unavailable, or unverified Warning evidence cannot close no-fault.
23. Never use a normal component as the root cause of no-fault. For no-fault, output root_cause_node="none", fault_type="no_fault", status="Normal". Once no-fault evidence is closed, do not call get_related_systems, get_upstream_nodes, or get_downstream_nodes.
24. RTU is self-contained. In RTU no-fault review, stay inside RTU components and close no-fault after enough high-confidence Normal checks; do not jump to boiler_plant or chiller_plant.
25. When evidence is closed, stop calling tools. If the latest diagnose_node result is same-node Fault for the best candidate, or no-fault has enough high-confidence Normal checks, the next response must be the final <diagnosis> block.
26. Provide a final diagnosis with root cause, fault type, and confidence level. The diagnosis fields must be non-empty; for no-fault use root_cause_node="none", fault_type="no_fault", status="Normal", and a calibrated confidence.

## Response Format
- Use <think>...</think> tags for compact reasoning immediately before each action.
- Use <tool_call>{{"name": "tool_name", "arguments": {{...}}}}</tool_call> to invoke tools. The complete <tool_call> block must appear before any auxiliary labels.
- When you've identified the root cause, provide your final diagnosis using:
  <diagnosis>{{"root_cause_node": "...", "fault_type": "...", "confidence": 0.XX, "affected_systems": [...]}}</diagnosis>
- Do not output TopoITL labels, hidden state JSON, or any extra text after the executable block. TopoITL is an internal training/audit signal, not an agent response format.

Always reason step-by-step. Never guess or fabricate information; only use data from tool results."""

# HVAC system knowledge for 7B model (compressed: ~400 tokens)
_SYSTEM_KNOWLEDGE_7B = """
## HVAC System Architecture

| System | ID | Upstream | Key Faults |
|--------|----|----------|------------|
| Chiller Plant | chiller_plant | primary | Condenser/cooling tower faults |
| Boiler Plant | boiler_plant | primary | Sensor bias, burner issues |
| Single-Duct AHU | sdahu | Chiller, Boiler | Coil/damper stuck, OA bias |
| Dual-Duct AHU | ddahu | Chiller, Boiler | Damper stuck, coil fouling |
| Rooftop Unit | rtu | self-contained | Evap fouling, refrigerant charge |
| Fan Coil Unit | fcu | Chiller, Boiler | Valve leak/stuck, coil fouling |
| Parallel FPU | pfpu | Boiler, SDAHU | Damper/reheat valve stuck |
| Series FPU | sfpu | Boiler, SDAHU | Damper/reheat valve stuck |

## Fault Propagation
- Chiller Plant -> chilled water -> SDAHU/DDAHU cooling coil, FCU cooling valve
- Boiler Plant -> hot water -> SDAHU/DDAHU heating coil, FCU heating valve, PFPU/SFPU reheat
- SDAHU -> conditioned air -> PFPU/SFPU zones
- Multiple downstream cooling issues -> suspect Chiller Plant
- Multiple downstream heating issues -> suspect Boiler Plant
- Single terminal affected -> fault is local, not upstream
- RTU is self-contained; faults do NOT propagate"""


def format_system_prompt(
    model_size: str = "7b",
    expose_status_summary: bool = False,
) -> str:
    """Generate the system prompt with tool schemas and system knowledge.

    Args:
        model_size: "3b" or "7b". 7B includes detailed HVAC knowledge.
        expose_status_summary: Internal-only switch for audit prompts. Keep
            False for all main SFT/RL/evaluation data.
    """
    tool_str = json.dumps(
        build_tool_schemas(expose_status_summary=expose_status_summary),
        indent=2,
    )
    base = _SYSTEM_PROMPT_BASE.format(tool_schemas=tool_str)
    if model_size == "7b":
        return base + "\n" + _SYSTEM_KNOWLEDGE_7B
    return base


def system_prompt_sha256(model_size: str = "7b") -> str:
    """Stable fingerprint for generated SFT/RL prompts."""
    prompt = format_system_prompt(model_size=model_size, expose_status_summary=False)
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


TOPOITL_LABEL_RE = re.compile(
    r"<topoitl_(?:state|action)>\s*\{.*?\}\s*</topoitl_(?:state|action)>\s*",
    re.DOTALL,
)


def strip_topoitl_labels(text: str) -> str:
    """Remove rendered TopoITL labels from deployable assistant text.

    The labels are retained in sample metadata and optional ablation assets, but
    the main executable agent should not learn to emit them in the interactive
    response stream.
    """
    return TOPOITL_LABEL_RE.sub("", str(text or "")).strip()


def trajectory_to_sft_messages(
    trajectory: DiagnosticTrajectory,
) -> List[Dict[str, str]]:
    """
    Convert a DiagnosticTrajectory into a list of chat messages.

    Returns:
        List of dicts with 'role' and 'content' keys, compatible with
        Qwen2.5 chat template.
    """
    messages = [
        {"role": "system", "content": format_system_prompt()},
    ]

    for step in trajectory.steps:
        if step.role == "user":
            messages.append({"role": "user", "content": step.content})
        elif step.role == "assistant":
            messages.append({"role": "assistant", "content": step.content})
        elif step.role == "tool":
            # Tool results are formatted as a special user message or tool role
            # Qwen2.5 uses the "tool" role for tool responses
            messages.append({
                "role": "tool",
                "name": step.tool_name or "unknown",
                "content": step.content,
            })

    return messages


def _trajectory_to_raw_conversations(
    trajectory: DiagnosticTrajectory,
) -> List[Dict[str, str]]:
    """Convert trajectory steps to raw ShareGPT turns before TopoITL labels."""
    raw_conversations = []

    for step in trajectory.steps:
        if step.role == "user":
            raw_conversations.append({
                "from": "human",
                "value": step.content,
            })
        elif step.role == "assistant":
            raw_conversations.append({
                "from": "gpt",
                "value": _compress_pre_action_reasoning(step.content),
            })
        elif step.role == "tool":
            raw_conversations.append({
                "from": "observation",
                "value": step.content,
            })

    return raw_conversations


def _compress_pre_action_reasoning(text: str) -> str:
    """Keep executable actions early without changing action JSON.

    Some final-diagnosis turns contain a natural-language summary between the
    reasoning block and the machine-readable <diagnosis> tag. For interactive
    evaluation the executable block must arrive first, so move such prose after
    <tool_call>/<diagnosis> while preserving it as auxiliary supervision.
    """
    text = strip_topoitl_labels(text)
    action_match = re.search(
        r"(<tool_call>\s*\{.*?\}\s*</tool_call>|"
        r"<diagnosis>\s*\{.*?\}\s*</diagnosis>)",
        text,
        re.DOTALL,
    )
    if not action_match:
        return text

    prefix = text[:action_match.start()]
    action_block = action_match.group(1)
    suffix = text[action_match.end():]

    match = re.search(r"<think>(.*?)</think>", prefix, re.DOTALL)
    if not match:
        return text

    thought = re.sub(r"\s+", " ", match.group(1)).strip()
    if len(thought) > MAX_PRE_ACTION_THINK_CHARS:
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", thought) if s.strip()]
        compact_parts = []
        total = 0
        for sent in sentences:
            add_len = len(sent) + (1 if compact_parts else 0)
            if compact_parts and total + add_len > MAX_PRE_ACTION_THINK_CHARS:
                break
            compact_parts.append(sent)
            total += add_len

        thought = " ".join(compact_parts).strip()
        if not thought:
            thought = match.group(1).strip()[:MAX_PRE_ACTION_THINK_CHARS].rsplit(" ", 1)[0].strip()
        if not thought:
            thought = match.group(1).strip()[:MAX_PRE_ACTION_THINK_CHARS].strip()

    pre_action_aux = (
        prefix[:match.start()] + prefix[match.end():]
    ).strip()
    parts = [f"<think>{thought}</think>", action_block]
    if pre_action_aux:
        parts.append(pre_action_aux)
    if suffix.strip():
        parts.append(suffix.strip())
    return "\n".join(parts)


def _extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text or "", re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _extract_diagnosis(text: str) -> Optional[Dict[str, Any]]:
    match = re.search(r"<diagnosis>\s*(\{.*?\})\s*</diagnosis>", text or "", re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _parse_observation(value: str) -> Dict[str, Any]:
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _terminal_tool_events(
    conversations: List[Dict[str, str]],
    final_index: int,
) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    pending: Optional[Dict[str, Any]] = None
    for idx, conv in enumerate(conversations[:final_index]):
        role = conv.get("from")
        value = str(conv.get("value", ""))
        if role == "gpt":
            call = _extract_tool_call(value)
            if call is None:
                continue
            pending = {
                "assistant": conv,
                "tool_call": call,
                "assistant_index": idx,
            }
            continue
        if role == "observation" and pending is not None:
            event = dict(pending)
            event["observation"] = conv
            event["observation_index"] = idx
            event["result"] = _parse_observation(value)
            events.append(event)
            pending = None
    return events


def _event_node(event: Dict[str, Any]) -> str:
    call = event.get("tool_call") or {}
    args = call.get("arguments") or {}
    result = event.get("result") or {}
    return str(
        result.get("node_id")
        or result.get("component_node")
        or args.get("node_id")
        or ""
    )


def _event_status(event: Dict[str, Any]) -> str:
    return str((event.get("result") or {}).get("status") or "").strip().lower()


def _event_confidence(event: Dict[str, Any]) -> float:
    try:
        return float((event.get("result") or {}).get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _event_has_readings(event: Dict[str, Any]) -> bool:
    result = event.get("result") or {}
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


def _select_terminal_events(
    events: List[Dict[str, Any]],
    diagnosis: Dict[str, Any],
    scenario_type: str,
) -> List[Dict[str, Any]]:
    """Select the minimal visible evidence needed to teach closure."""
    root = str(diagnosis.get("root_cause_node") or "")
    stype = str(scenario_type or "").lower()
    no_fault = (
        "no_fault" in stype
        or root.lower() in {"", "none", "normal", "no_fault"}
        or str(diagnosis.get("fault_type") or "").lower() in {"no_fault", "normal", "none"}
    )

    if no_fault:
        normal_events = [
            event for event in events
            if (event.get("tool_call") or {}).get("name") == "diagnose_node"
            and _event_status(event) == "normal"
            and _event_confidence(event) >= 0.85
        ]
        if len(normal_events) < 2:
            return []
        selected = normal_events[-3:] if len(normal_events) >= 3 else normal_events[-2:]
        if selected:
            return selected

    if "low_confidence" in stype and root:
        root_diag_events = [
            event for event in events
            if (event.get("tool_call") or {}).get("name") == "diagnose_node"
            and _event_node(event) == root
        ]
        sensor_events = [
            event for event in events
            if (event.get("tool_call") or {}).get("name") == "get_node_sensors"
            and _event_node(event) == root
            and _event_has_readings(event)
        ]
        selected = (root_diag_events[-1:] + sensor_events[-1:])
        if selected:
            return sorted(selected, key=lambda event: event.get("assistant_index", 0))

    if root:
        root_events = [
            event for event in events
            if (event.get("tool_call") or {}).get("name") == "diagnose_node"
            and _event_node(event) == root
        ]
        if root_events:
            return root_events[-1:]

    diagnose_events = [
        event for event in events
        if (event.get("tool_call") or {}).get("name") == "diagnose_node"
    ]
    return diagnose_events[-1:] if diagnose_events else events[-1:]


def _event_identity(event: Dict[str, Any]) -> tuple:
    return (
        event.get("assistant_index", -1),
        event.get("observation_index", -1),
    )


def _event_tool_name(event: Dict[str, Any]) -> str:
    return str((event.get("tool_call") or {}).get("name") or "")


def _is_topology_context_event(event: Dict[str, Any]) -> bool:
    return _event_tool_name(event) in {
        "get_system_overview",
        "get_node_children",
        "get_related_systems",
        "get_upstream_nodes",
        "get_downstream_nodes",
    }


def _is_fault_context_event(event: Dict[str, Any]) -> bool:
    return (
        _event_tool_name(event) == "diagnose_node"
        and _event_status(event) in {"abnormal", "warning", "fault"}
    )


def _terminal_context_events(
    events: List[Dict[str, Any]],
    selected_events: List[Dict[str, Any]],
    scenario_type: str,
) -> List[Dict[str, Any]]:
    """Return visible prefix evidence for terminal-closure augmentation.

    A terminal-closure row should teach "stop after closed evidence", not a
    hidden shortcut from the user prompt directly to the oracle root.  Keep a
    compact prefix of topology and abnormal-evidence turns that makes the final
    diagnostic action reachable from visible context.
    """
    if not selected_events:
        return []

    first_selected_index = min(
        int(event.get("assistant_index", 0) or 0)
        for event in selected_events
    )
    prefix = [
        event for event in events
        if int(event.get("assistant_index", 0) or 0) < first_selected_index
    ]
    if not prefix:
        return []

    stype = str(scenario_type or "").lower()
    max_context = 8 if "cross_system" in stype else 5
    if "no_fault" in stype:
        max_context = 6

    selected: List[Dict[str, Any]] = []
    seen = set()

    def add(event: Dict[str, Any]) -> None:
        ident = _event_identity(event)
        if ident in seen:
            return
        selected.append(event)
        seen.add(ident)

    overview = next(
        (event for event in prefix if _event_tool_name(event) == "get_system_overview"),
        None,
    )
    if overview is not None:
        add(overview)

    first_system_children = next(
        (
            event for event in prefix
            if _event_tool_name(event) == "get_node_children"
            and str(
                ((event.get("tool_call") or {}).get("arguments") or {}).get("node_id")
                or ""
            ).startswith("system::")
        ),
        None,
    )
    if first_system_children is not None:
        add(first_system_children)

    candidates = [
        event for event in prefix
        if _is_topology_context_event(event) or _is_fault_context_event(event)
    ]
    for event in candidates[-max_context:]:
        add(event)

    return sorted(
        selected,
        key=lambda event: int(event.get("assistant_index", 0) or 0),
    )


def _terminal_closure_entry(
    trajectory: DiagnosticTrajectory,
    base_entry: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build a short SFT row that only supervises the final closure action.

    Full expert trajectories teach search and tool use.  These augmentation
    rows teach the policy that, after sufficient visible evidence, the next
    action is a terminal <diagnosis> rather than another exploratory tool call.
    """
    raw = _trajectory_to_raw_conversations(trajectory)
    first_human = next((conv for conv in raw if conv.get("from") == "human"), None)
    final_index = None
    final_conv = None
    final_diag = None
    for idx in range(len(raw) - 1, -1, -1):
        conv = raw[idx]
        if conv.get("from") != "gpt":
            continue
        diagnosis = _extract_diagnosis(str(conv.get("value", "")))
        if diagnosis is not None:
            final_index = idx
            final_conv = conv
            final_diag = diagnosis
            break
    if first_human is None or final_index is None or final_conv is None or not final_diag:
        return None

    events = _terminal_tool_events(raw, final_index)
    selected_events = _select_terminal_events(
        events,
        final_diag,
        trajectory.scenario_type,
    )
    if not selected_events:
        return None

    context_events = _terminal_context_events(
        events,
        selected_events,
        trajectory.scenario_type,
    )

    first_selected_tool = _event_tool_name(selected_events[0])
    if first_selected_tool == "diagnose_node" and not context_events:
        # Without visible prefix evidence this would teach the model to jump
        # from the user prompt straight to the hidden root-cause node.
        return None

    conversations: List[Dict[str, Any]] = [dict(first_human)]
    emitted = set()
    for event in context_events + selected_events:
        ident = _event_identity(event)
        if ident in emitted:
            continue
        emitted.add(ident)
        conversations.append(dict(event["assistant"]))
        conversations.append(dict(event["observation"]))
    conversations.append(dict(final_conv))

    annotated_conversations, transitions = annotate_sharegpt_conversations(conversations)
    if not transitions:
        return None

    metadata = dict(base_entry.get("metadata", {}))
    metadata.update({
        "augmentation_type": "terminal_closure",
        "source_scenario_id": trajectory.scenario_id,
        "n_tool_calls": len(selected_events),
        "n_topoitl_steps": len(transitions),
        "topoitl_transitions": transitions,
        "terminal_supervision": True,
    })
    return {
        "id": f"{base_entry.get('id') or trajectory.scenario_id}__terminal_closure",
        "system": format_system_prompt(),
        "conversations": annotated_conversations,
        "metadata": metadata,
    }


def trajectory_to_sharegpt(
    trajectory: DiagnosticTrajectory,
) -> Dict[str, Any]:
    """
    Convert a trajectory to ShareGPT format for LLaMA-Factory compatibility.

    Merges consecutive gpt turns into a single turn to ensure clean
    turn-taking boundaries (every assistant turn followed by observation
    or end-of-conversation).

    Returns:
        Dict with 'conversations' key containing the formatted dialogue.
    """
    raw_conversations = _trajectory_to_raw_conversations(trajectory)

    # Merge consecutive gpt turns into a single turn
    # This prevents the model from learning to generate multi-turn content
    # in a single output (which causes hallucinated tool responses).
    import re

    def _merge_think_content(existing: str, new_content: str) -> str:
        """Merge two assistant messages, combining their <think> blocks."""
        # Extract think content from both
        existing_thinks = re.findall(r'<think>(.*?)</think>', existing, re.DOTALL)
        new_thinks = re.findall(r'<think>(.*?)</think>', new_content, re.DOTALL)

        # Extract non-think content
        existing_rest = re.sub(r'<think>.*?</think>\s*', '', existing, flags=re.DOTALL).strip()
        new_rest = re.sub(r'<think>.*?</think>\s*', '', new_content, flags=re.DOTALL).strip()

        # Combine think contents with sentence-level deduplication
        all_thinks = existing_thinks + new_thinks
        if len(all_thinks) > 1:
            # Deduplicate at sentence level to prevent "复读机" content
            seen_sents = set()
            deduped_parts = []
            for think_text in all_thinks:
                sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', think_text) if s.strip()]
                unique_sents = []
                for s in sents:
                    if s not in seen_sents:
                        seen_sents.add(s)
                        unique_sents.append(s)
                if unique_sents:
                    deduped_parts.append(" ".join(unique_sents))
            combined_think = "\n".join(deduped_parts) if deduped_parts else ""
        else:
            combined_think = "\n".join(all_thinks) if all_thinks else ""

        # Build merged message
        parts = []
        if combined_think:
            parts.append(f"<think>{combined_think}</think>")
        if existing_rest:
            parts.append(existing_rest)
        if new_rest:
            parts.append(new_rest)
        return _compress_pre_action_reasoning("\n".join(parts))

    conversations = []
    for conv in raw_conversations:
        if (conversations
                and conversations[-1]["from"] == "gpt"
                and conv["from"] == "gpt"):
            # Merge: combine think blocks
            conversations[-1]["value"] = _merge_think_content(
                conversations[-1]["value"], conv["value"]
            )
        else:
            conversations.append(conv)

    annotated_conversations, topoitl_transitions = annotate_sharegpt_conversations(
        conversations
    )

    return {
        "id": trajectory.scenario_id,
        "system": format_system_prompt(),
        "conversations": annotated_conversations,
        "metadata": {
            "design_version": SFT_DESIGN_VERSION,
            "topoitl_version": TOPOITL_VERSION,
            "system_prompt_sha256": system_prompt_sha256(),
            "scenario_id": trajectory.scenario_id,
            "scenario_type": trajectory.scenario_type,
            "ground_truth": trajectory.ground_truth,
            "n_tool_calls": trajectory.metadata.get("n_tool_calls", 0),
            "n_topoitl_steps": len(topoitl_transitions),
            "path_length": trajectory.metadata.get("path_length", 0),
            "diagnostic_path_nodes": trajectory.metadata.get("diagnostic_path_nodes", []),
            "reference_propagation_edges": trajectory.metadata.get("reference_propagation_edges", []),
            "symptom_node": trajectory.metadata.get("symptom_node", ""),
            "affected_systems": trajectory.metadata.get("affected_systems", []),
            "source_file": trajectory.metadata.get("source_file", ""),
            "time_window_start": trajectory.metadata.get("time_window_start", 0),
            "time_window_end": trajectory.metadata.get("time_window_end", 0),
            "difficulty": trajectory.metadata.get("difficulty", "medium"),
            "trajectory_quality": trajectory.metadata.get("trajectory_quality", "unknown"),
            "tool_faithful": trajectory.metadata.get("tool_faithful", True),
            "label_consistent": trajectory.metadata.get("label_consistent", True),
            "final_diagnosis": trajectory.metadata.get("final_diagnosis", {}),
            "topoitl_transitions": topoitl_transitions,
        },
    }


def format_sft_dataset(
    trajectories: List[DiagnosticTrajectory],
    output_path: str,
    format_type: str = "sharegpt",
    add_terminal_closure_examples: bool = False,
) -> str:
    """
    Format all trajectories into a SFT training dataset file.

    Args:
        trajectories: List of generated trajectories.
        output_path: Path to save the formatted dataset.
        format_type: "sharegpt" for LLaMA-Factory, "messages" for raw messages.
        add_terminal_closure_examples: When True, append short closure-only
            augmentation rows. Disabled by default for paper alignment: the
            full expert trajectories already teach closure, and EPO's terminal
            reward handles premature-stop behavior at the RL stage. Augmenting
            here skews the trajectory-length statistics (avg turns / tool calls)
            away from the reported dataset and biases the policy toward early
            termination.

    Returns:
        Path to the saved dataset file.
    """
    dataset = []
    n_base_examples = 0
    n_terminal_closure_examples = 0
    seen_ids = {}

    for traj in trajectories:
        if format_type == "sharegpt":
            entry = trajectory_to_sharegpt(traj)
        else:
            entry = {
                "id": traj.scenario_id,
                "messages": trajectory_to_sft_messages(traj),
                "metadata": {
                    "design_version": SFT_DESIGN_VERSION,
                    "topoitl_version": TOPOITL_VERSION,
                    "system_prompt_sha256": system_prompt_sha256(),
                    "scenario_type": traj.scenario_type,
                    "ground_truth": traj.ground_truth,
                    "diagnostic_path_nodes": traj.metadata.get("diagnostic_path_nodes", []),
                    "symptom_node": traj.metadata.get("symptom_node", ""),
                    "affected_systems": traj.metadata.get("affected_systems", []),
                    "source_file": traj.metadata.get("source_file", ""),
                    "time_window_start": traj.metadata.get("time_window_start", 0),
                    "time_window_end": traj.metadata.get("time_window_end", 0),
                    "label_consistent": traj.metadata.get("label_consistent", True),
                    "final_diagnosis": traj.metadata.get("final_diagnosis", {}),
                },
            }
        n_base_examples += 1

        base_id = str(entry.get("id") or traj.scenario_id)
        count = seen_ids.get(base_id, 0)
        seen_ids[base_id] = count + 1
        if count:
            entry["id"] = f"{base_id}__dup{count}"
        else:
            entry["id"] = base_id
        dataset.append(entry)

        if format_type == "sharegpt" and add_terminal_closure_examples:
            terminal_entry = _terminal_closure_entry(traj, entry)
            if terminal_entry is not None:
                terminal_id = str(terminal_entry.get("id") or f"{base_id}__terminal_closure")
                terminal_count = seen_ids.get(terminal_id, 0)
                seen_ids[terminal_id] = terminal_count + 1
                if terminal_count:
                    terminal_entry["id"] = f"{terminal_id}__dup{terminal_count}"
                else:
                    terminal_entry["id"] = terminal_id
                dataset.append(terminal_entry)
                n_terminal_closure_examples += 1

    # Save as JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(dataset)} SFT examples to {output_path}")

    # Also save a summary
    summary = {
        "design_version": SFT_DESIGN_VERSION,
        "system_prompt_sha256": system_prompt_sha256(),
        "total_examples": len(dataset),
        "base_examples": n_base_examples,
        "terminal_closure_examples": n_terminal_closure_examples,
        "format": format_type,
        "scenario_type_counts": {},
        "augmentation_type_counts": {},
        "avg_turns": 0,
        "avg_tool_calls": 0,
    }

    from collections import Counter
    type_counts = Counter()
    augmentation_counts = Counter()
    total_turns = 0
    total_tools = 0

    for entry in dataset:
        if format_type == "sharegpt":
            meta = entry.get("metadata", {})
            type_counts[meta.get("scenario_type", "unknown")] += 1
            augmentation_counts[meta.get("augmentation_type", "full_trajectory")] += 1
            total_turns += len(entry.get("conversations", []))
            total_tools += meta.get("n_tool_calls", 0)
        else:
            meta = entry.get("metadata", {})
            type_counts[meta.get("scenario_type", "unknown")] += 1
            augmentation_counts[meta.get("augmentation_type", "full_trajectory")] += 1
            total_turns += len(entry.get("messages", []))

    summary["scenario_type_counts"] = dict(type_counts)
    summary["augmentation_type_counts"] = dict(augmentation_counts)
    summary["avg_turns"] = total_turns / max(len(dataset), 1)
    summary["avg_tool_calls"] = total_tools / max(len(dataset), 1)

    summary_path = output_path.replace(".jsonl", "_summary.json")
    save_json(summary, summary_path)

    return output_path
