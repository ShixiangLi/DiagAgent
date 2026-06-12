"""
Reasoning Composer - High-diversity compositional reasoning generation.

Replaces fixed templates with a compositional approach:
  [Observation Summary] + [Reasoning Logic] + [Next Step Plan]

Each component is independently sampled and combined to produce
diverse, evidence-grounded diagnostic reasoning text.
"""

import json
import random
from typing import Any, Dict, List, Optional, Tuple

from src.utils.io_utils import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Building blocks - each list should have 8-15+ variants
# ============================================================================

# --- Initial reasoning (first turn) ---

_INITIAL_OPENERS = [
    "I've received a fault report.",
    "A diagnostic request has been submitted.",
    "There's been an alert from the building management system.",
    "A maintenance ticket has come in regarding HVAC performance.",
    "The monitoring system has flagged an issue.",
    "Unusual HVAC behavior has been reported.",
    "Building operations flagged a potential equipment issue.",
    "A performance degradation alert was triggered.",
    "I need to investigate a reported HVAC malfunction.",
    "The BMS has raised an alarm that requires investigation.",
]

_INITIAL_PLANS = [
    "Let me start by understanding the building's HVAC topology to identify which systems are involved.",
    "I'll first get an overview of all HVAC systems to determine where to start investigating.",
    "My first step is to map out the building systems and identify potential fault sources.",
    "I need to examine the full system topology before narrowing down the investigation.",
    "Let me pull up the system architecture to plan an efficient diagnostic path.",
    "I should begin with a high-level view of all systems to prioritize my investigation.",
    "To efficiently diagnose this, I'll start with a top-down system overview.",
    "Let me survey the building's HVAC layout to identify which subsystems could be involved.",
]

# --- System selection ---

_SYS_SELECT_STARTERS = [
    "The symptom domain makes {system_name} ({system_id}) a plausible candidate, so I will inspect its components.",
    "I'll start by drilling into {system_name} ({system_id}) as a candidate system and check the evidence.",
    "The {system_name} system is relevant to this type of alert. Let me get its component list.",
    "Given the nature of the alert, {system_name} ({system_id}) is a reasonable place to start. Let me inspect it.",
    "I'll focus on {system_name} first as a candidate, then verify with component-level tools.",
    "The symptom pattern is compatible with {system_name}, but I need tool evidence. Retrieving its component tree.",
    "Starting with {system_name} ({system_id}) as a topology-relevant candidate before drawing conclusions.",
]

# --- Diagnose request ---

_DIAGNOSE_PRE = [
    "Let me run diagnostics on {node_name} to check its operational status.",
    "I'll diagnose {node_name} to see if this component is the source of the issue.",
    "Checking {node_name} for any abnormalities that could explain the reported symptoms.",
    "Time to check {node_name}'s operating status.",
    "I need to evaluate {node_name}'s current operating condition.",
    "Let me query the diagnostic tool for {node_name} to get its fault prediction.",
    "Running a component diagnostic check on {node_name}.",
    "I'll check whether {node_name} is operating within expected parameters.",
    "Performing diagnostic analysis on {node_name} to assess its operating state.",
    "Next, I should examine {node_name} - it's a potential contributor to the observed symptoms.",
    "Let me evaluate {node_name}'s current operating evidence.",
    "I'll investigate {node_name} to determine if it's functioning correctly.",
]

# --- Normal response (after observation shows Normal) ---

_NORMAL_OBSERVATIONS = [
    "The diagnostic result reports {node_name} is operating normally with {confidence:.1%} confidence.",
    "Results show {node_name} is operating normally - {confidence:.1%} confidence.",
    "The diagnostic result shows {node_name} has no anomalies (confidence: {confidence:.1%}).",
    "{node_name} passed the diagnostic check with {confidence:.1%} normal-status confidence.",
    "The diagnostic result indicates {node_name} is within normal operating ranges ({confidence:.1%}).",
    "The tool result indicates {node_name} is functioning correctly at {confidence:.1%} confidence.",
    "No deviation detected at {node_name}; confidence is {confidence:.1%}.",
    "{node_name}'s observed behavior falls within the expected baseline ({confidence:.1%} confidence).",
]

_NORMAL_CONCLUSIONS = [
    "This component is not supported as the root cause, so I should continue with the next visible candidate in the current diagnostic plan.",
    "This clears {node_name} for the current branch; I should use the visible topology to choose the next planned check.",
    "I can rule out {node_name} and continue with another already visible candidate, not jump to an unrelated system.",
    "This node is clean. I should stay with the current evidence path unless topology explicitly points elsewhere.",
    "Ruling this one out narrows the search; the next action should follow the current component list or traced anchor.",
    "{node_name} can be eliminated from the suspect list for this branch.",
    "Since {node_name} appears normal, I should continue only along the planned topology path.",
    "Not the culprit. I will use the current visible candidates to choose the next check.",
]

# --- Uncertain / unavailable response (after observation is not Normal) ---

_UNCERTAIN_OBSERVATIONS = [
    "The diagnostic result for {node_name} is inconclusive ({status}; confidence: {confidence:.1%}).",
    "The diagnostic result cannot reliably clear {node_name}: {status} at {confidence:.1%} confidence.",
    "{node_name} did not return a usable Normal confirmation ({status}, {confidence:.1%}).",
    "The available evidence for {node_name} is unavailable or too weak to interpret as normal ({status}).",
]

_UNCERTAIN_CONCLUSIONS = [
    "I should not treat this as normal evidence; I will continue with topology and other observations.",
    "This does not rule the node out, so I need additional evidence before eliminating it.",
    "Because this is inconclusive, I will pivot based on the reported symptoms and topology rather than call it normal.",
    "This is a data-availability limitation, not proof of normal operation.",
]

# --- Abnormal response (after observation shows Abnormal) ---

_ABNORMAL_OBSERVATIONS = [
    "{node_name} shows abnormal behavior: {hint}.",
    "The diagnostics indicate {node_name} has anomalous readings: {hint}.",
    "{node_name} is exhibiting abnormality - {hint}.",
    "Alert: {node_name} returned an Abnormal status. Indicators show {hint}.",
    "The diagnostic result detected abnormal conditions at {node_name}: {hint}.",
    "Something's off with {node_name}. The analysis shows: {hint}.",
    "{node_name} deviates from normal: {hint}.",
    "Anomaly detected at {node_name}: {hint}.",
]

_ABNORMAL_REASONING = [
    "However, the abnormality pattern suggests this node is showing symptoms rather than being the root cause.",
    "The suggested direction is {direction}, indicating the root cause is {direction} of this component.",
    "This is likely a downstream effect. The fault probably originates {direction}.",
    "This component is affected but not the source - the issue propagates from {direction}.",
    "Based on the abnormality pattern, I should trace {direction} to find the primary fault.",
    "The abnormal readings here are consistent with a propagation effect from {direction}.",
    "This node is experiencing collateral impact. The actual fault is likely {direction}.",
    "The indicator pattern matches a secondary effect. I need to look {direction} for the root cause.",
]

_ABNORMAL_NEXT = [
    "I should trace {direction} to find the root cause.",
    "Let me continue investigating {direction}.",
    "I need to trace further {direction} to pinpoint the source.",
    "Continuing the diagnostic path {direction}.",
    "Moving {direction} along the causal chain to find the source.",
    "Let me follow the {direction} propagation path.",
]

# --- Upstream trace ---

_UPSTREAM_OBSERVATIONS = [
    "The abnormality at {node_name} suggests an upstream issue. Let me check what feeds into this component.",
    "Since {node_name} is showing symptoms but isn't the root cause, I should trace upstream to find the source.",
    "The direction indicates upstream. Let me find what systems or components feed into {node_name}.",
    "{node_name}'s abnormality is consistent with an upstream fault propagation. Tracing the supply chain.",
    "The symptom pattern at {node_name} points to an upstream source. Let me identify the feeding systems.",
    "This downstream symptom at {node_name} necessitates an upstream investigation.",
    "The causal direction points upstream from {node_name}. Let me trace the connections.",
    "I suspect the root cause feeds into {node_name} from upstream. Checking cross-system links.",
]

# --- Fault found (with evidence citation) ---

def compose_fault_found(
    node_name: str,
    fault_type: str,
    confidence: float,
    sensor_readings: Dict[str, float],
    rng: random.Random,
) -> str:
    """Compose a fault-found reasoning block WITH sensor evidence.

    Uses confidence-appropriate language:
    - High (>=0.7): definitive/confirmed
    - Medium (0.5-0.7): likely/probable
    - Low (<0.5): possible/candidate
    """

    if confidence >= 0.7:
        _FAULT_OPENERS = [
            f"Root cause identified! {node_name} has a definitive fault: {fault_type} (confidence: {confidence:.1%}).",
            f"The diagnostics confirm {node_name} is the source of the problem: {fault_type} detected with {confidence:.1%} confidence.",
            f"Found it - {node_name} has {fault_type}. This is definitive same-node fault evidence.",
            f"Confirmed: {node_name} is experiencing {fault_type} with {confidence:.1%} certainty.",
            f"The diagnostic result has positively identified {fault_type} at {node_name} ({confidence:.1%} confidence).",
        ]
    elif confidence >= 0.5:
        _FAULT_OPENERS = [
            f"Likely root cause: {node_name} shows {fault_type} (confidence: {confidence:.1%}). This is the strongest candidate.",
            f"The diagnostic result flags {node_name} as the probable source: {fault_type} at {confidence:.1%} confidence.",
            f"{node_name} is the most likely fault source - {fault_type} detected with moderate confidence ({confidence:.1%}).",
            f"Diagnosis points to {node_name}: {fault_type} ({confidence:.1%}). While not conclusive, this is the top prediction.",
        ]
    else:
        _FAULT_OPENERS = [
            f"Possible root cause at {node_name}: {fault_type} (confidence: {confidence:.1%}). This is the top candidate despite low certainty.",
            f"The strongest diagnostic prediction is {fault_type} at {node_name} ({confidence:.1%} confidence). Further evidence is needed to confirm.",
            f"{node_name} shows signs of {fault_type} ({confidence:.1%}). This is a low-confidence detection but the strongest signal available.",
            f"Primary candidate: {fault_type} at {node_name} with {confidence:.1%} confidence. The evidence is uncertain but this is the most plausible fault.",
        ]

    opener = rng.choice(_FAULT_OPENERS)

    # Build evidence citation from sensor readings
    evidence = _format_sensor_evidence(sensor_readings, rng, fault_type=fault_type)
    if evidence:
        _EVIDENCE_INTROS = [
            f"The sensor readings confirm this: {evidence}.",
            f"Relevant measurements: {evidence}.",
            f"Supporting data: {evidence}.",
        ]
        evidence_text = rng.choice(_EVIDENCE_INTROS)
    else:
        evidence_text = ""

    return f"{opener} {evidence_text}".strip()


def compose_final_diagnosis(
    root_node: str,
    fault_type: str,
    affected: str,
    sensor_evidence: str,
    rng: random.Random,
) -> str:
    """Compose the final diagnosis reasoning block."""

    _FINALS = [
        f"Based on my investigation, I've traced the issue from the observed symptoms to the root cause. "
        f"The supported diagnosis is {fault_type} at {root_node}; affected scope from the visible evidence: {affected}.",

        f"Diagnosis complete. The root cause is {fault_type} at {root_node}. "
        f"The reported affected systems are limited to the topology and tool evidence observed: {affected}.",

        f"Investigation concluded. {root_node} is the primary fault source ({fault_type}), "
        f"with affected scope traced from visible topology evidence: {affected}.",

        f"After systematic analysis, the root cause has been isolated to {root_node} "
        f"({fault_type}). Evidence-supported affected scope: {affected}.",

        f"The diagnostic investigation is complete. Root cause: {fault_type} at {root_node}. "
        f"Visible tool evidence supports the following affected systems: {affected}.",

        f"Summary: {root_node} exhibits {fault_type}. "
        f"The affected-system scope is traced through visible topology evidence: {affected}.",

        f"Concluding diagnosis: {root_node} has {fault_type}, supported by systematic "
        f"topology exploration and component diagnostics. Affected systems: {affected}.",
    ]

    result = rng.choice(_FINALS)
    if sensor_evidence:
        _EVIDENCE_SUFFIXES = [
            f" Critical evidence: {sensor_evidence}.",
            f" Key diagnostic indicators: {sensor_evidence}.",
            f" The diagnosis is grounded in: {sensor_evidence}.",
        ]
        result += rng.choice(_EVIDENCE_SUFFIXES)

    return result


# --- No-fault conclusion (with evidence) ---

def compose_no_fault_conclusion(
    checked_nodes: List[str],
    rng: random.Random,
) -> str:
    """Compose a no-fault conclusion citing what was checked."""

    nodes_str = ", ".join(checked_nodes) if checked_nodes else "multiple components"

    _NO_FAULT = [
        f"After checking {nodes_str}, all diagnostics return normal. "
        f"The system appears to be operating within expected parameters.",

        f"No faults were detected during this investigation. "
        f"I examined {nodes_str} - all returned normal diagnostics with high confidence.",

        f"The diagnostic sweep across {nodes_str} found no anomalies. "
        f"All sensor readings fall within normal operating ranges.",

        f"Investigation complete. All tested components ({nodes_str}) appear normal. "
        f"No fault condition detected in the examined subsystems.",

        f"After systematically checking {nodes_str}, I found no evidence of faults. "
        f"Each component's observed data aligned with expected baseline values.",

        f"All diagnostic checks came back clean for {nodes_str}. "
        f"The reported symptoms may have been transient or caused by external factors.",

        f"Comprehensive diagnosis of {nodes_str} reveals normal operation across the board. "
        f"No root cause identified - the system is functioning as expected.",
    ]
    return rng.choice(_NO_FAULT)


# --- Wrong system check ---

_WRONG_SYSTEM_REASONING = [
    "Let me also check the {system_name} system to rule it out as a potential source.",
    "Before concluding, I should verify that {system_name} is not involved.",
    "I'll briefly investigate {system_name} to ensure it's operating normally.",
    "To be thorough, let me check whether {system_name} could be a contributing factor.",
    "I should rule out {system_name} before narrowing my conclusion.",
    "It's worth checking {system_name} to ensure the fault isn't originating there.",
    "Let me quickly scan {system_name} to eliminate it from the suspect list.",
    "For completeness, I'll verify {system_name} isn't the upstream cause.",
]

# --- Wrong system result ---

def compose_wrong_system_ack(
    node_name: str,
    system_name: str,
    actual_status: str,
    confidence: float,
    rng: random.Random,
) -> str:
    """Compose acknowledgment after checking a wrong-system node."""

    if actual_status in ("Normal", "success"):
        _ACKS = [
            f"{node_name} in the {system_name} system is operating normally ({confidence:.1%}). "
            f"This system is not the source of the reported issue. Let me investigate the system indicated by the symptoms.",

            f"The {system_name} system checks out - {node_name} is normal ({confidence:.1%}). "
            f"I can eliminate this system and focus elsewhere.",

            f"No issues found in {system_name}: {node_name} returned normal status ({confidence:.1%}). "
            f"Moving on to the next candidate system.",

            f"{node_name} in {system_name} passes the diagnostic check at {confidence:.1%} confidence. "
            f"The fault source is in a different system.",

            f"Cleared: {system_name}'s {node_name} shows normal operation. "
            f"The root cause must be in another system.",
        ]
    else:
        _ACKS = [
            f"Interesting - {node_name} in {system_name} shows {actual_status} status. "
            f"However, this may be a secondary effect. Let me continue investigating the primary suspect system.",

            f"{node_name} in {system_name} returned {actual_status}. "
            f"This could be a propagation effect rather than the root cause. Continuing investigation.",

            f"The {system_name} system's {node_name} is showing {actual_status}. "
            f"This warrants attention but may not be the primary fault source. Continuing.",
        ]
    return rng.choice(_ACKS)


# --- Cross-system transition ---

def compose_cross_system_transition(
    from_system: str,
    to_system: str,
    rng: random.Random,
) -> str:
    """Compose reasoning for transitioning between systems."""

    _TRANSITIONS = [
        f"The abnormality suggests the root cause is in an upstream system. "
        f"Let me trace upstream to the {to_system} system.",

        f"Evidence points to a cross-system fault propagation. "
        f"The {from_system} symptoms may originate from {to_system}. I will verify that candidate.",

        f"Cross-system analysis needed: the {from_system} anomalies appear to be caused by "
        f"an upstream issue in {to_system}. Switching investigation target for verification.",

        f"The fault propagation pattern makes {to_system} the upstream candidate. "
        f"I will inspect that system before deciding.",

        f"Based on the topology links, {to_system} feeds into {from_system}. "
        f"{to_system} is the next component-source candidate to test.",

        f"Following the causal chain upstream from {from_system} to {to_system}.",

        f"The abnormal readings in {from_system} are consistent with an upstream fault. "
        f"Expanding investigation to {to_system}.",
    ]
    return rng.choice(_TRANSITIONS)


# --- Upstream trace acknowledgment ---

def compose_upstream_ack(
    from_name: str,
    to_name: str,
    rng: random.Random,
    connected_via: Optional[str] = None,
    target_component: Optional[str] = None,
) -> str:
    """Compose acknowledgment after upstream trace reveals connection."""
    if connected_via or target_component:
        anchor = []
        if target_component:
            anchor.append(f"target_component={target_component}")
        if connected_via:
            anchor.append(f"connected_via={connected_via}")
        anchor_text = ", ".join(anchor)
        options = [
            f"The upstream trace and related-system map point from {from_name} to {to_name}; "
            f"the exact visible anchor is {anchor_text}. I should use those node_ids directly.",
            f"The cross-system dependency is grounded by {anchor_text}. "
            f"I will keep the diagnosis on those returned anchors rather than inventing a component name.",
            f"Topology now gives a concrete {to_name} candidate for {from_name}: {anchor_text}. "
            f"The next diagnostic action should use the exact returned node_id.",
        ]
        return rng.choice(options)

    _ACKS = [
        f"The upstream trace reveals a connection to the {to_name} system. "
        f"The abnormality in {from_name} may be caused by an upstream fault. "
        f"Let me investigate the {to_name} system.",

        f"Found a cross-system link: {to_name} feeds into {from_name}. "
        f"If {to_name} has a fault, it could explain the downstream symptoms. Investigating.",

        f"Upstream analysis shows {to_name} as a potential source for {from_name}'s issues. "
        f"Switching focus to {to_name}.",

        f"The topology confirms {to_name} -> {from_name} dependency. "
        f"{to_name} is now the upstream candidate to verify.",

        f"Cross-system dependency identified: {to_name} supplies {from_name}. "
        f"An upstream fault in {to_name} would explain the observed anomalies.",

        f"The upstream path reaches {to_name}, so I will test that system before "
        f"calling it the root cause.",

        f"This trace exposes {to_name} as the feeding system for {from_name}; "
        f"the next step is same-system component evidence.",
    ]
    return rng.choice(_ACKS)


# ============================================================================
# Sensor evidence formatting
# ============================================================================

# Fault-type -> causally relevant sensor prefixes (priority order)
_FAULT_SENSOR_PRIORITY: Dict[str, List[str]] = {
    # RTU / DX faults - compressor, capacity, supply air are relevant
    "condfouling":   ["COND", "COMP", "RTU_SEN_CAPA", "RTU_SA_TEMP", "RTU_TOT", "RTU_SA_FLOW"],
    "evapfouling":   ["EVAP", "COMP", "RTU_SEN_CAPA", "RTU_SA_TEMP", "RTU_TOT", "RTU_SA_FLOW"],
    "overcharge":    ["COMP", "RTU_SEN_CAPA", "RTU_SA_TEMP", "RTU_TOT", "RTU_SA_FLOW"],
    "undercharge":   ["COMP", "RTU_SEN_CAPA", "RTU_SA_TEMP", "RTU_TOT", "RTU_SA_FLOW"],
    "liquidpipe":    ["COMP", "RTU_SEN_CAPA", "RTU_SA_TEMP", "RTU_TOT"],
    "suctionpipe":   ["COMP", "RTU_SEN_CAPA", "RTU_SA_TEMP", "RTU_TOT"],
    # Boiler faults - hot water loop sensors
    "boiler":        ["HWL", "SEC_POW", "PM_", "HWP"],
    "hot_water":     ["HWL", "SEC_POW", "PM_", "HWP"],
    # Chiller faults - chilled/condenser water sensors (NOT OA_TEMP)
    "chiller":       ["CWL", "CT_", "CHWST", "CHWRT", "CHILLER", "GPM"],
    "bypass":        ["CWL", "CHWST", "CHWRT", "BYPASS", "CHILLER", "GPM"],
    "coolingtower":  ["CT_", "CWL", "COND", "CHWST", "CHWRT"],
    "secondary_chilled": ["CWL", "CHWST", "CHWRT", "GPM", "CT_"],
    # AHU faults - coil, damper, airflow sensors
    "DMPRStuck":     ["OA_CFM", "MA_TEMP", "DMPR", "SA_TEMP", "SA_", "RA_TEMP"],
    "VLVStuck":      ["CHWC", "HWC", "VLV", "SA_TEMP", "MA_TEMP", "GPM"],
    "Fouling":       ["CHWC", "HWC", "VLV", "EWT", "LWT", "SA_TEMP", "MA_TEMP", "DAT"],
    "SensorBias":    ["TEMP", "CFM", "HUMD", "SPT", "SA_", "RA_", "MA_", "DAT"],
    "Reheat":        ["RH_", "VAV", "DAT", "HWP"],
    "VAVDMPRStuck":  ["VAV", "CFM", "DAT", "DMPR"],
}


def _format_sensor_evidence(
    sensor_readings: Dict[str, float],
    rng: random.Random,
    max_sensors: int = 3,
    fault_type: Optional[str] = None,
) -> str:
    """Format sensor readings into a human-readable evidence string.

    When *fault_type* is provided, sensors causally related to the fault
    are prioritised so that cited evidence is domain-appropriate.

    Post-processing:
    - Clamps WB ≤ DB (wet-bulb cannot exceed dry-bulb)
    - Rejects environment-only sensors (OA_TEMP*) as sole evidence for
      process faults (bypass, chiller, coolingtower)
    """
    if not sensor_readings:
        return ""

    # Physical constraint: clamp WB ≤ DB
    readings = dict(sensor_readings)
    if "OA_TEMP" in readings and "OA_TEMP_WB" in readings:
        db = readings["OA_TEMP"]
        wb = readings["OA_TEMP_WB"]
        if isinstance(db, (int, float)) and isinstance(wb, (int, float)):
            if wb > db:
                readings["OA_TEMP_WB"] = round(db - rng.uniform(2.0, 5.0), 4)

    # Physical constraint: airflow and water-flow sensors cannot be negative.
    for key, value in list(readings.items()):
        key_upper = key.upper()
        is_flow = any(token in key_upper for token in ("CFM", "FLOW", "GPM"))
        if is_flow and isinstance(value, (int, float)) and value < 0:
            readings[key] = 0.0

    all_sensors = list(readings.items())

    # --- prioritise causally relevant sensors ---
    priority_prefixes: List[str] = []
    if fault_type:
        ft_lower = fault_type.lower()
        for key, prefixes in _FAULT_SENSOR_PRIORITY.items():
            if key.lower() in ft_lower or ft_lower in key.lower():
                priority_prefixes = prefixes
                break

    if priority_prefixes:
        # Split into high / low priority
        high = [(n, v) for n, v in all_sensors
                if any(n.upper().startswith(p.upper()) or p.upper() in n.upper()
                       for p in priority_prefixes)]
        low  = [(n, v) for n, v in all_sensors if (n, v) not in high]

        # Pick from high-priority first, fill with low-priority if needed
        if len(high) >= max_sensors:
            selected = rng.sample(high, max_sensors)
        elif high:
            needed = max_sensors - len(high)
            extra = rng.sample(low, min(needed, len(low))) if low else []
            selected = high + extra
        else:
            # No causally relevant sensors available at all
            # Return empty -> caller will use generic evidence string
            return ""
    else:
        # No fault type info - fall back to random sample
        if len(all_sensors) > max_sensors:
            selected = rng.sample(all_sensors, max_sensors)
        else:
            selected = all_sensors

    # Final guard: reject if ALL selected sensors are environment-only (OA_*)
    env_prefixes = ("OA_TEMP", "OA_HUMD")
    non_env = [n for n, _ in selected if not any(n.startswith(p) for p in env_prefixes)]
    if not non_env and fault_type:
        # All sensors are environment - not valid for process faults
        return ""

    parts = []
    for name, value in selected:
        if isinstance(value, float):
            parts.append(f"{name}={value:.2f}")
        else:
            parts.append(f"{name}={value}")

    return ", ".join(parts)


# ============================================================================
# Compositional assembly
# ============================================================================

def compose_initial_reasoning(rng: random.Random) -> str:
    """Compose an initial reasoning block (high diversity)."""
    opener = rng.choice(_INITIAL_OPENERS)
    plan = rng.choice(_INITIAL_PLANS)
    return f"{opener} {plan}"


def compose_system_selection(system_name: str, system_id: str, rng: random.Random) -> str:
    """Compose system selection reasoning."""
    return rng.choice(_SYS_SELECT_STARTERS).format(
        system_name=system_name, system_id=system_id,
    )


def compose_diagnose_request(node_name: str, rng: random.Random) -> str:
    """Compose a diagnose request reasoning."""
    return rng.choice(_DIAGNOSE_PRE).format(node_name=node_name)


def compose_normal_response(
    node_name: str,
    confidence: float,
    sensor_readings: Dict[str, float],
    rng: random.Random,
) -> str:
    """Compose a normal-status response WITH sensor evidence."""
    obs = rng.choice(_NORMAL_OBSERVATIONS).format(
        node_name=node_name, confidence=confidence,
    )
    conclusion = rng.choice(_NORMAL_CONCLUSIONS).format(node_name=node_name)

    # Optionally cite a sensor reading as evidence
    if sensor_readings and rng.random() < 0.6:
        evidence = _format_sensor_evidence(sensor_readings, rng, max_sensors=2)
        if evidence:
            _EVIDENCE_PHRASES = [
                f"Key readings: {evidence} - all within acceptable ranges.",
                f"Sensor check: {evidence} (nominal).",
                f"Verified by sensor data: {evidence}.",
            ]
            evidence_text = rng.choice(_EVIDENCE_PHRASES)
            return f"{obs} {evidence_text} {conclusion}"

    return f"{obs} {conclusion}"


def compose_uncertain_response(
    node_name: str,
    status: str,
    confidence: float,
    message: str,
    sensor_readings: Dict[str, float],
    rng: random.Random,
) -> str:
    """Compose an uncertainty response without implying normal operation."""
    clean_status = status or "unknown"
    obs = rng.choice(_UNCERTAIN_OBSERVATIONS).format(
        node_name=node_name,
        status=clean_status,
        confidence=confidence,
    )
    conclusion = rng.choice(_UNCERTAIN_CONCLUSIONS)

    details = ""
    if message:
        compact = " ".join(str(message).split())
        if len(compact) > 140:
            compact = compact[:137] + "..."
        compact = compact.rstrip(". ")
        details = f" Tool note: {compact}."

    if sensor_readings and rng.random() < 0.35:
        evidence = _format_sensor_evidence(sensor_readings, rng, max_sensors=2)
        if evidence:
            return f"{obs}{details} Available readings: {evidence}. {conclusion}"

    return f"{obs}{details} {conclusion}"


def compose_abnormal_response(
    node_name: str,
    hint: str,
    direction: str,
    sensor_readings: Dict[str, float],
    rng: random.Random,
) -> str:
    """Compose an abnormal-status response WITH sensor evidence."""
    obs = rng.choice(_ABNORMAL_OBSERVATIONS).format(
        node_name=node_name, hint=hint,
    )
    reasoning = rng.choice(_ABNORMAL_REASONING).format(direction=direction)
    next_step = rng.choice(_ABNORMAL_NEXT).format(direction=direction)

    # Include sensor evidence
    if sensor_readings and rng.random() < 0.7:
        evidence = _format_sensor_evidence(sensor_readings, rng, max_sensors=2)
        if evidence:
            _EVIDENCE_PHRASES = [
                f"Relevant readings: {evidence}.",
                f"Sensor data: {evidence}.",
                f"The measurements show: {evidence}.",
            ]
            evidence_text = rng.choice(_EVIDENCE_PHRASES)
            return f"{obs} {evidence_text} {reasoning} {next_step}"

    return f"{obs} {reasoning} {next_step}"


def compose_upstream_trace(node_name: str, rng: random.Random) -> str:
    """Compose an upstream trace reasoning."""
    return rng.choice(_UPSTREAM_OBSERVATIONS).format(node_name=node_name)


def compose_wrong_system(system_name: str, rng: random.Random) -> str:
    """Compose a wrong-system check reasoning."""
    return rng.choice(_WRONG_SYSTEM_REASONING).format(system_name=system_name)


# ============================================================================
# New reasoning blocks - topology-guided exploration
# ============================================================================

def compose_anomaly_score_selection(
    system_name: str,
    system_id: str,
    anomaly_score: float,
    rng: random.Random,
) -> str:
    """Compose system selection reasoning without hidden health-score claims.

    The function name is kept for compatibility with older generator code, but
    main-experiment trajectories must not mention anomaly_score or health_status
    because those fields are not exposed to the agent.
    """
    if anomaly_score <= 0.05:
        _NORMAL_SELECTIONS = [
            f"{system_name} ({system_id}) is a plausible candidate from the "
            f"symptom context. I will verify its component structure directly "
            f"before drawing any conclusion.",

            f"I will inspect {system_name} ({system_id}) as a candidate system, "
            f"then test nodes only after exposing its topology.",

            f"The overview gives me the building layout but not enough evidence "
            f"to clear {system_name}. I will perform a targeted component check.",

            f"I need tool evidence before deciding whether {system_name} is "
            f"normal. I will start by listing its components.",
        ]
        return rng.choice(_NORMAL_SELECTIONS)

    _SELECTIONS = [
        f"The symptom pattern and system role make {system_name} ({system_id}) "
        f"a reasonable first investigation target.",

        f"{system_name} is a plausible source for the reported behavior, so I "
        f"should start by exposing its component hierarchy.",

        f"The reported symptoms are consistent with issues in {system_name} "
        f"({system_id}). Let me examine its components.",

        f"{system_name} is topologically relevant to this request. I will drill "
        f"into its components and test evidence one node at a time.",

        f"The overview identifies {system_name} as one of the available HVAC "
        f"systems. Based on the symptom context, I will focus there first.",

        f"I will begin with {system_name} because its function matches the "
        f"reported symptom domain.",

        f"Before diagnosing individual nodes, I need the component list for "
        f"{system_name}.",
    ]
    return rng.choice(_SELECTIONS)


def compose_multi_system_ranking(
    systems_ranked: list,  # [(name, id, score), ...]
    rng: random.Random,
) -> str:
    """Compose reasoning about multiple topology-relevant systems."""
    if not systems_ranked:
        return "Let me examine the building systems."

    top = systems_ranked[0]
    descriptions = [n for n, _, _ in systems_ranked[:3]]
    ranked_str = ", ".join(descriptions)

    _RANKINGS = [
        f"Candidate systems from the topology and symptom context: {ranked_str}. "
        f"I'll start with {top[0]}.",

        f"Several systems could plausibly explain the symptom: {ranked_str}. "
        f"Prioritizing {top[0]} for investigation.",

        f"The building topology leaves several candidates: {ranked_str}. "
        f"Beginning with {top[0]} because it is the best initial match.",

        f"Topology-based triage suggests checking {top[0]} first, then "
        f"expanding to {ranked_str} if the evidence remains inconclusive.",
    ]
    return rng.choice(_RANKINGS)


def compose_system_elimination(
    eliminated_system: str,
    n_nodes_checked: int,
    rng: random.Random,
) -> str:
    """Compose reasoning after eliminating a system (all nodes Normal)."""
    _ELIMINATIONS = [
        f"After checking {n_nodes_checked} components in {eliminated_system}, "
        f"all returned Normal status. This system can be ruled out as the fault source. "
        f"I need to investigate the next candidate system.",

        f"{eliminated_system} is clear - {n_nodes_checked} components checked, "
        f"all normal. The fault must be elsewhere. Let me move to the next system.",

        f"No faults found in {eliminated_system} after checking {n_nodes_checked} nodes. "
        f"Eliminating this system and redirecting investigation.",

        f"Investigation of {eliminated_system} complete: {n_nodes_checked} components, "
        f"zero faults. Proceeding to the next topology-relevant system.",

        f"{eliminated_system} shows normal operation across all {n_nodes_checked} "
        f"checked components. Pivoting to the next system in my priority list.",
    ]
    return rng.choice(_ELIMINATIONS)


def compose_system_inconclusive(
    system_name: str,
    n_nodes_checked: int,
    rng: random.Random,
) -> str:
    """Compose reasoning after a system check produced inconclusive evidence."""
    _INCONCLUSIVE = [
        f"I checked {n_nodes_checked} component(s) in {system_name}, but the evidence was not sufficient to clear the system. "
        f"I will avoid treating unavailable or unknown tool results as Normal and continue with the topology-guided search.",

        f"The {system_name} probe is inconclusive rather than clean. "
        f"I need to keep moving through the diagnostic path instead of declaring this system normal.",

        f"{system_name} did not provide enough usable evidence for elimination. "
        f"I will pivot using the symptom location, topology, and subsequent targeted diagnostic calls.",
    ]
    return rng.choice(_INCONCLUSIVE)


def compose_system_pivot(
    from_system: str,
    to_system: str,
    to_score: float,
    rng: random.Random,
) -> str:
    """Compose reasoning for pivoting from an eliminated system to the next."""
    _PIVOTS = [
        f"Since {from_system} is clear, I'll now investigate {to_system} "
        f"as the next topology-relevant candidate.",

        f"Moving investigation from {from_system} to {to_system}. Let me get "
        f"its component structure.",

        f"With {from_system} eliminated, {to_system} is the next plausible "
        f"system to verify based on the remaining symptom context.",

        f"No issues in {from_system}. Next candidate: {to_system}. Let me "
        f"examine its components.",

        f"Redirecting from {from_system} to {to_system}; the topology and "
        f"reported symptoms make it the next system to verify.",
    ]
    return rng.choice(_PIVOTS)


def compose_status_summary_request(system_name: str, rng: random.Random) -> str:
    """Compose reasoning for using the system-wide status summary tool."""
    options = [
        f"Before probing components one by one, I should get a system-wide status summary for {system_name} to prioritize the most suspicious nodes.",
        f"A quick component status summary for {system_name} will help avoid an inefficient blind sweep.",
        f"I'll request a status summary across {system_name} so I can focus on candidate components first.",
        f"To narrow the search efficiently, I need a node status summary for {system_name}.",
    ]
    return rng.choice(options)


def compose_status_summary_ack(
    system_name: str,
    candidate_count: int,
    rng: random.Random,
) -> str:
    """Compose reasoning after receiving a system status summary."""
    if candidate_count > 0:
        options = [
            f"The {system_name} summary highlights {candidate_count} candidate node(s). I should verify the strongest candidate directly.",
            f"The status summary narrows {system_name} to {candidate_count} non-normal node(s), so I can avoid scanning unrelated components.",
            f"{system_name} has {candidate_count} candidate node(s) in the summary. I'll drill into those rather than sweeping the whole system.",
        ]
    else:
        options = [
            f"The {system_name} summary shows no non-normal candidates. I should either verify representative nodes or pivot to a related system.",
            f"No candidate nodes appear in the {system_name} status summary, so a limited verification is enough before moving on.",
            f"The summary for {system_name} is clean. I can avoid a full component sweep unless other evidence points back here.",
        ]
    return rng.choice(options)


def compose_related_systems_request(system_name: str, rng: random.Random) -> str:
    """Compose reasoning for querying cross-system relationships."""
    options = [
        f"The symptoms may propagate across systems, so I should check which systems are connected to {system_name}.",
        f"Before jumping systems, I need the cross-system connections for {system_name}.",
        f"A related-systems query will show whether {system_name} is being fed by an upstream plant or feeding downstream equipment.",
        f"To trace propagation cleanly, I'll inspect the systems related to {system_name}.",
    ]
    return rng.choice(options)


def compose_related_systems_ack(
    from_system: str,
    to_system: str,
    rng: random.Random,
    connected_via: Optional[str] = None,
    target_component: Optional[str] = None,
    medium: Optional[str] = None,
) -> str:
    """Compose reasoning after related-system topology is returned."""
    if connected_via or target_component:
        anchor = []
        if target_component:
            anchor.append(f"target_component={target_component}")
        if connected_via:
            anchor.append(f"connected_via={connected_via}")
        if medium:
            anchor.append(f"medium={medium}")
        anchor_text = ", ".join(anchor)
        options = [
            f"The related-system topology links {from_system} with {to_system} through exact anchors: {anchor_text}. "
            f"I must use these returned node_ids directly for the cross-system trace.",
            f"The connection map does not just name a system; it exposes node-level anchors ({anchor_text}). "
            f"I should trace the target_component and diagnose the connected_via candidate if it is the plant fault source.",
            f"Cross-system propagation is now grounded by {anchor_text}. "
            f"I will not copy a downstream component name into {to_system}; the next actions must use the returned anchors.",
        ]
        return rng.choice(options)

    options = [
        f"The related-system topology links {from_system} with {to_system}. I should inspect {to_system} next as the likely propagation source.",
        f"The connection map supports a transition from {from_system} to {to_system}; this is the correct direction for root-cause tracing.",
        f"With the cross-system relationship confirmed, {to_system} becomes the next diagnostic target.",
    ]
    return rng.choice(options)


def compose_related_systems_impact_request(
    system_name: str,
    rng: random.Random,
) -> str:
    """Compose reasoning for checking propagation scope after a root fault."""
    options = [
        f"Since the fault can propagate beyond {system_name}, I should check its related systems before finalizing impact.",
        f"Before closing the diagnosis, I need the cross-system connection map for {system_name}.",
        f"The root fault is localized, but the affected-system scope depends on cross-system links from {system_name}.",
        f"I will inspect related systems for {system_name} to verify how this fault can affect downstream equipment.",
    ]
    return rng.choice(options)


def compose_related_systems_impact_ack(
    system_name: str,
    upstream_count: int,
    downstream_count: int,
    rng: random.Random,
) -> str:
    """Compose reasoning after related-system impact topology is returned."""
    options = [
        f"The related-system map for {system_name} shows {downstream_count} downstream and {upstream_count} upstream connection(s), which constrains the affected-system list.",
        f"Cross-system topology is now checked for {system_name}: {downstream_count} downstream connection(s), {upstream_count} upstream connection(s). I can use this to report propagation scope.",
        f"The connection map confirms the impact boundary around {system_name}; downstream links are the relevant propagation path for this root fault.",
    ]
    return rng.choice(options)


def compose_children_query(
    parent_name: str,
    rng: random.Random,
    purpose: str = "diagnosis",
) -> str:
    """Compose topology-expansion reasoning for get_node_children.

    High-frequency topology expansion used to rely on a single sentence, which
    caused SFT data to overfit one phrase instead of the underlying action
    policy.  This function keeps the action semantics stable while varying the
    observable reason for expanding a parent node.
    """
    name = str(parent_name or "this node").strip() or "this node"

    if purpose == "no_fault":
        openers = [
            f"Before I can close a no-fault review, I need finer topology under {name}.",
            f"A no-fault decision for this system needs visible leaf checks, so I will expand {name}.",
            f"{name} is still an aggregate node; I should expose its components before clearing the case.",
            f"I need more than a broad aggregate view of {name} before declaring normal operation.",
            f"To avoid a shallow no-fault closure, I will list the components inside {name}.",
            f"The no-fault review is not closed until representative visible components under {name} are checked.",
        ]
        reasons = [
            "The next diagnostic calls should target concrete components rather than a parent grouping.",
            "This keeps the normal-evidence chain tied to actual visible nodes.",
            "Low-level candidates are needed for a defensible clean-system conclusion.",
            "A parent-level listing is the safest way to choose valid node_id values.",
        ]
    elif purpose == "root_reveal":
        openers = [
            f"The upstream trace points into {name}, and I need its component hierarchy before diagnosing the root candidate.",
            f"{name} is the relevant upstream system, but the candidate node must be exposed by topology first.",
            f"Before testing the suspected root in {name}, I will reveal the returned hierarchy.",
            f"The root candidate is inside {name}; I should list its components and use the exact returned node_id.",
            f"I need the visible component path in {name} before making a same-node diagnostic call.",
        ]
        reasons = [
            "That prevents inventing a plant component path.",
            "The diagnosis should use node ids returned by tools.",
            "Topology evidence has to precede the root-node check.",
            "This keeps the cross-system trace grounded in visible structure.",
        ]
    else:
        openers = [
            f"I will expand {name} so the next diagnostic step uses a visible component.",
            f"Before choosing a node to diagnose, I need the child structure under {name}.",
            f"{name} is still a parent in the topology; listing its children will expose valid candidates.",
            f"To continue the topology-guided search, I should reveal the components below {name}.",
            f"I need to inspect the immediate children of {name} before selecting the next node.",
            f"The current path reaches {name}; I will query its children to keep the investigation grounded.",
            f"Expanding {name} will show which exact node_id values can be checked next.",
            f"I should not guess a descendant of {name}; the child list needs to come from the tool.",
        ]
        reasons = [
            "This keeps the search tied to the visible topology.",
            "The next action should be based on returned nodes, not inferred names.",
            "It also avoids skipping over intermediate equipment.",
            "That gives a concrete frontier for the next diagnostic call.",
            "The topology result will determine whether to diagnose or expand further.",
        ]

    connectors = [
        " ",
        " ",
        " ",
        " Next, ",
    ]
    second = rng.choice(reasons)
    if second.startswith("This") or second.startswith("That") or second.startswith("It"):
        return f"{rng.choice(openers)}{rng.choice(connectors)}{second}"
    return f"{rng.choice(openers)} {second}"


def compose_observable_system_start(
    system_name: str,
    rng: random.Random,
    context: str = "general",
) -> str:
    """Compose reasoning for starting from a system named by visible symptoms."""
    name = str(system_name or "the visible system").strip() or "the visible system"
    if context == "cross":
        options = [
            f"The visible equipment cue points to {name}. I will inspect that downstream side first, then trace upstream only if tool evidence supports it.",
            f"The report gives an observable starting point in {name}, so I will begin there and use topology before assigning a plant-level cause.",
            f"{name} is the system exposed by the symptom wording. I will verify local evidence there before moving across systems.",
            f"I will start with {name} because it is visible in the request, then follow returned topology if the evidence propagates upstream.",
        ]
    elif context == "low_confidence":
        options = [
            f"The report gives a visible cue for {name}. I will inspect that system first and require same-node evidence because the signal may be weak.",
            f"{name} is the observable starting point. I will check it carefully and avoid closing on a weak result without sensor support.",
            f"The symptom wording names equipment in {name}; I will start there and verify the candidate with same-node evidence.",
            f"I will enter {name} from the visible cue, then treat any low-confidence result as unresolved until sensors support it.",
        ]
    else:
        options = [
            f"The user report gives a visible equipment cue for {name}. I will start there and verify whether it is the root cause or an affected node.",
            f"{name} is the observable system in the request, so I will inspect it first and let tool evidence decide the next step.",
            f"The symptom wording grounds the initial search in {name}. I will use topology and diagnostics to confirm or redirect.",
            f"I should begin with {name} because it is visible from the report, not because of hidden root-cause knowledge.",
        ]
    return rng.choice(options)


def compose_explicit_system_start(system_name: str, rng: random.Random) -> str:
    """Compose reasoning when the prompt explicitly brackets a system name."""
    name = str(system_name or "the named system").strip() or "the named system"
    options = [
        f"The request explicitly names {name}, so I will start inside that system and verify evidence directly.",
        f"Because {name} is named in the prompt, the first topology expansion should be within that system.",
        f"The user supplied {name} as the investigation context. I will inspect its components before drawing conclusions.",
        f"I will honor the explicit {name} scope first instead of inferring an unseen downstream starting point.",
    ]
    return rng.choice(options)


def compose_cross_downstream_start(
    downstream_name: str,
    rng: random.Random,
    visible_cue: bool = False,
) -> str:
    """Compose the first downstream-side expansion in cross-system cases."""
    name = str(downstream_name or "the downstream system").strip() or "the downstream system"
    if visible_cue:
        options = [
            f"The visible symptom cue points to {name}. I will inspect that local system first, then trace upstream if the evidence is not closed there.",
            f"{name} is the observable starting side of the cross-system case. I will check it before assigning responsibility upstream.",
            f"The report grounds the investigation in {name}; topology will decide whether the source is local or upstream.",
            f"I will begin at {name} because that is where the symptom is visible, then follow returned connections if needed.",
        ]
    else:
        options = [
            f"The symptom pattern suggests an affected downstream or terminal system. I will inspect {name} first and trace upstream only through tools.",
            f"Before assigning a plant-level cause, I should verify the downstream side at {name}.",
            f"{name} is a plausible affected system, so I will start there and use topology to test propagation.",
            f"I will use {name} as the observable symptom-side entry point, then follow explicit cross-system links if local evidence remains secondary.",
        ]
    return rng.choice(options)


def compose_upstream_anchor_trace(
    local_name: str,
    target_component: str,
    rng: random.Random,
) -> str:
    """Compose reasoning for tracing upstream from a related-systems anchor."""
    local = str(local_name or "the local symptom node").strip() or "the local symptom node"
    target = str(target_component or "the returned target component").strip() or "the returned target component"
    options = [
        f"The local symptom node is {local}. The related-system result exposes target_component={target}, so I will trace upstream from that exact node.",
        f"Cross-system topology gives a concrete anchor, target_component={target}. I will use it to trace upstream from {local}.",
        f"Rather than inventing a plant-side node, I will follow the returned target_component={target} upstream from {local}.",
        f"The connection map grounds the trace at target_component={target}; I will query upstream from that visible component.",
        f"{local} is not enough to close the case. The returned anchor target_component={target} is the correct node for the upstream trace.",
    ]
    return rng.choice(options)


def compose_warning_response(
    node_name: str,
    confidence: float,
    sensor_readings: Dict[str, float],
    rng: random.Random,
) -> str:
    """Compose reasoning when diagnose_node returns Warning (low confidence)."""
    pct = f"{confidence:.1%}"
    _WARNINGS = [
        f"The diagnostic result for {node_name} is borderline - "
        f"confidence is only {pct}, below the definitive threshold. "
        f"I should verify this with raw sensor data before concluding.",

        f"{node_name} returned a Warning status at {pct} confidence. "
        f"This is inconclusive. I need to cross-check with the actual "
        f"sensor readings to assess whether the evidence supports a fault.",

        f"Interesting - {node_name} shows a Warning but confidence is only "
        f"{pct} confident. The result is ambiguous. Let me examine the "
        f"sensor data directly to get more evidence.",

        f"The diagnostic result is uncertain for {node_name} ({pct}). "
        f"A low-confidence Warning doesn't confirm a fault. "
        f"Sensor-level verification is needed.",

        f"{node_name}: Warning status detected, but confidence ({pct}) is "
        f"below the reliable threshold. I should check the actual sensor "
        f"readings to make an evidence-based determination.",
    ]

    result = rng.choice(_WARNINGS)

    # Add sensor evidence if available
    if sensor_readings and rng.random() < 0.5:
        evidence = _format_sensor_evidence(sensor_readings, rng, max_sensors=2)
        if evidence:
            result += f" Current readings: {evidence}."

    return result


def compose_sensor_verification(
    node_name: str,
    sensor_findings: str,
    rng: random.Random,
) -> str:
    """Compose reasoning after examining sensors to verify a Warning."""
    _VERIFICATIONS = [
        f"Sensor analysis for {node_name}: {sensor_findings}. "
        f"Combined with the earlier Warning, this supports the candidate "
        f"fault without treating the low-confidence result as definitive by itself.",

        f"After examining the sensor data, {sensor_findings}. "
        f"Despite the low-confidence diagnostic result, the sensor evidence "
        f"supports keeping {node_name} as the fault candidate.",

        f"Cross-referencing sensor data with the Warning: {sensor_findings}. "
        f"The same-node readings are consistent with the borderline warning. "
        f"I can make a supported diagnosis without overstating the confidence.",

        f"Sensor verification complete: {sensor_findings}. "
        f"The data supports the candidate abnormal operation at {node_name}, "
        f"strengthening but not replacing the borderline detection.",

        f"The raw sensor readings ({sensor_findings}) support the anomaly "
        f"flagged at {node_name}. "
        f"The fault is supported despite the low diagnostic confidence.",
    ]
    return rng.choice(_VERIFICATIONS)
