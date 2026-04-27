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
    "An anomaly has been reported in the HVAC infrastructure.",
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
    "Based on the reported symptoms in {system_name}, I should investigate the {system_name} ({system_id}) system first.",
    "The symptoms point to the {system_name} system. Let me examine its components.",
    "I'll start by drilling into the {system_name} ({system_id}) system to check its components.",
    "The {system_name} system is the most likely source given the symptom profile. Let me get its component list.",
    "Given the nature of the alert, the {system_name} ({system_id}) is a strong candidate. Let me inspect it.",
    "I'll focus on the {system_name} system first - it matches the symptom signature.",
    "The symptom pattern is consistent with {system_name} issues. Retrieving its component tree.",
    "Starting my investigation with {system_name} ({system_id}) as it's the primary suspect based on the symptom location.",
]

# --- Diagnose request ---

_DIAGNOSE_PRE = [
    "Let me run diagnostics on {node_name} to check its operational status.",
    "I'll diagnose {node_name} to see if this component is the source of the issue.",
    "Checking {node_name} for any abnormalities that could explain the reported symptoms.",
    "Time to run a health check on {node_name}.",
    "I need to evaluate {node_name}'s current operating condition.",
    "Let me query the Oracle model for {node_name} to get its fault prediction.",
    "Running fault detection on {node_name} using the sensor data.",
    "I'll check whether {node_name} is operating within expected parameters.",
    "Performing diagnostic analysis on {node_name} to assess its health.",
    "Next, I should examine {node_name} - it's a potential contributor to the observed symptoms.",
    "Let me evaluate {node_name}'s sensor readings against the trained model baseline.",
    "I'll investigate {node_name} to determine if it's functioning correctly.",
]

# --- Normal response (after observation shows Normal) ---

_NORMAL_OBSERVATIONS = [
    "The diagnostic model reports {node_name} is operating normally with {confidence:.1%} confidence.",
    "Results show {node_name} is healthy - {confidence:.1%} confidence in normal operation.",
    "The Oracle confirms {node_name} has no anomalies (confidence: {confidence:.1%}).",
    "{node_name} passed the diagnostic check with {confidence:.1%} normal-status confidence.",
    "Sensor analysis indicates {node_name} is within normal operating ranges ({confidence:.1%}).",
    "The fault model predicts {node_name} is functioning correctly at {confidence:.1%} confidence.",
    "No deviation detected at {node_name}. Model confidence: {confidence:.1%}.",
    "{node_name}'s sensor readings all fall within the expected baseline ({confidence:.1%} confidence).",
]

_NORMAL_CONCLUSIONS = [
    "This component is not the source of the issue. I should investigate other components.",
    "The fault must be elsewhere. Let me continue checking other candidates.",
    "I can rule out {node_name} and move on to the next suspect.",
    "This node is clean. I need to check other components or trace to a different system.",
    "Ruling this one out. The root cause is somewhere else in the system.",
    "{node_name} can be eliminated from the suspect list.",
    "Since {node_name} is healthy, the problem lies elsewhere. Continuing investigation.",
    "Not the culprit. I'll shift my attention to other components.",
]

# --- Abnormal response (after observation shows Abnormal) ---

_ABNORMAL_OBSERVATIONS = [
    "{node_name} shows abnormal behavior: {hint}.",
    "The diagnostics indicate {node_name} has anomalous readings: {hint}.",
    "{node_name} is exhibiting abnormality - {hint}.",
    "Alert: {node_name} returned an Abnormal status. Indicators show {hint}.",
    "The model detected anomalous conditions at {node_name}: {hint}.",
    "Something's off with {node_name}. The analysis shows: {hint}.",
    "{node_name} deviates from normal: {hint}.",
    "Anomaly detected at {node_name} - the Oracle flags: {hint}.",
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
            f"Found it - {node_name} has {fault_type}. This explains the downstream symptoms observed earlier.",
            f"Confirmed: {node_name} is experiencing {fault_type} with {confidence:.1%} certainty.",
            f"The fault model has positively identified {fault_type} at {node_name} ({confidence:.1%} confidence).",
        ]
    elif confidence >= 0.5:
        _FAULT_OPENERS = [
            f"Likely root cause: {node_name} shows {fault_type} (confidence: {confidence:.1%}). This is the strongest candidate.",
            f"The Oracle flags {node_name} as the probable source: {fault_type} at {confidence:.1%} confidence.",
            f"{node_name} is the most likely fault source - {fault_type} detected with moderate confidence ({confidence:.1%}).",
            f"Diagnosis points to {node_name}: {fault_type} ({confidence:.1%}). While not conclusive, this is the top prediction.",
        ]
    else:
        _FAULT_OPENERS = [
            f"Possible root cause at {node_name}: {fault_type} (confidence: {confidence:.1%}). This is the top candidate despite low certainty.",
            f"The Oracle's best prediction is {fault_type} at {node_name} ({confidence:.1%} confidence). Further evidence needed to confirm.",
            f"{node_name} shows signs of {fault_type} ({confidence:.1%}). This is a low-confidence detection but the strongest signal available.",
            f"Primary candidate: {fault_type} at {node_name} with {confidence:.1%} confidence. The model is uncertain but this is the most plausible fault.",
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
        f"The fault at {root_node} ({fault_type}) is causing propagation effects on downstream systems: {affected}.",

        f"Diagnosis complete. The root cause is {fault_type} at {root_node}, affecting {affected}. "
        f"The diagnostic path confirms the causal chain from upstream fault to downstream symptoms.",

        f"Investigation concluded. {root_node} is the primary fault source ({fault_type}), "
        f"with cascading effects observed in: {affected}.",

        f"After systematic analysis, the root cause has been isolated to {root_node} "
        f"({fault_type}). Downstream impact detected in: {affected}.",

        f"The diagnostic investigation is complete. Root cause: {fault_type} at {root_node}. "
        f"The fault has propagated to affect: {affected}.",

        f"Summary: {root_node} exhibits {fault_type}, which is the origin of the symptoms "
        f"observed across {affected}. The causal chain has been verified.",

        f"Concluding diagnosis: {root_node} has {fault_type}, confirmed through systematic "
        f"elimination and direct sensor analysis. Affected systems: {affected}.",
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
        f"I examined {nodes_str} - all passed health checks with high confidence.",

        f"The diagnostic sweep across {nodes_str} found no anomalies. "
        f"All sensor readings fall within normal operating ranges.",

        f"Investigation complete. All tested components ({nodes_str}) are healthy. "
        f"No fault condition detected in the examined subsystems.",

        f"After systematically checking {nodes_str}, I found no evidence of faults. "
        f"Each component's sensor data aligned with expected baseline values.",

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

            f"The {system_name} system checks out - {node_name} is healthy ({confidence:.1%}). "
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
        f"The {from_system} symptoms may originate from {to_system}. Investigating.",

        f"Cross-system analysis needed: the {from_system} anomalies appear to be caused by "
        f"an upstream issue in {to_system}. Switching investigation target.",

        f"The fault propagation pattern suggests {to_system} as the upstream source. "
        f"Transitioning my investigation to that system.",

        f"Based on the topology links, {to_system} feeds into {from_system}. "
        f"The root cause is likely in {to_system}. Let me check its components.",

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
) -> str:
    """Compose acknowledgment after upstream trace reveals connection."""

    _ACKS = [
        f"The upstream trace reveals a connection to the {to_name} system. "
        f"The abnormality in {from_name} may be caused by an upstream fault. "
        f"Let me investigate the {to_name} system.",

        f"Found a cross-system link: {to_name} feeds into {from_name}. "
        f"If {to_name} has a fault, it could explain the downstream symptoms. Investigating.",

        f"Upstream analysis shows {to_name} as a potential source for {from_name}'s issues. "
        f"Switching focus to {to_name}.",

        f"The topology confirms {to_name} -> {from_name} dependency. "
        f"The root cause likely originates in {to_name}. Continuing there.",

        f"Cross-system dependency identified: {to_name} supplies {from_name}. "
        f"An upstream fault in {to_name} would explain the observed anomalies.",
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
# New reasoning blocks - anomaly-score & exploration
# ============================================================================

def compose_anomaly_score_selection(
    system_name: str,
    system_id: str,
    anomaly_score: float,
    rng: random.Random,
) -> str:
    """Compose system selection reasoning based on anomaly score."""
    pct = f"{anomaly_score:.0%}"
    if anomaly_score <= 0.05:
        _NORMAL_SELECTIONS = [
            f"The system overview shows {system_name} ({system_id}) is not elevated "
            f"({pct} anomaly score). Since the user asked about this system, I will "
            f"verify its components directly.",

            f"{system_name} ({system_id}) has a normal overview score ({pct}). "
            f"I should still inspect it because the request names this system.",

            f"The overview does not flag {system_name}; its anomaly score is {pct}. "
            f"I will perform a targeted health check rather than treating it as a "
            f"primary fault suspect.",

            f"All evidence so far keeps {system_name} at a normal anomaly level "
            f"({pct}). I will continue with a limited verification of its components.",
        ]
        return rng.choice(_NORMAL_SELECTIONS)

    _SELECTIONS = [
        f"The system overview shows {system_name} ({system_id}) has the highest "
        f"anomaly score ({pct}). This makes it the primary investigation target.",

        f"Based on the anomaly scores, {system_name} stands out with {pct} - "
        f"significantly higher than other systems. I should start investigating here.",

        f"{system_name} ({system_id}) has an anomaly score of {pct}, "
        f"indicating potential issues. Let me examine its components.",

        f"The health indicators show {system_name} at {pct} anomaly level. "
        f"This is the most likely fault source. Let me drill into its components.",

        f"Analyzing the system overview: {system_name} ({pct} anomaly) is the "
        f"top priority. Other systems show lower scores. Focusing investigation here.",

        f"The Oracle's system-level analysis flags {system_name} with a {pct} "
        f"anomaly score. I'll begin my investigation with this system.",

        f"System health check reveals {system_name} as the primary suspect "
        f"(anomaly: {pct}). Let me get its component list.",
    ]
    return rng.choice(_SELECTIONS)


def compose_multi_system_ranking(
    systems_ranked: list,  # [(name, id, score), ...]
    rng: random.Random,
) -> str:
    """Compose reasoning about multiple systems with anomaly scores."""
    if not systems_ranked:
        return "Let me examine the building systems."

    top = systems_ranked[0]
    descriptions = [f"{n} ({s:.0%})" for n, _, s in systems_ranked[:3]]
    ranked_str = ", ".join(descriptions)

    _RANKINGS = [
        f"System overview anomaly rankings: {ranked_str}. "
        f"I'll start with {top[0]} as it has the highest anomaly score.",

        f"Multiple systems show elevated anomaly scores: {ranked_str}. "
        f"Prioritizing {top[0]} for investigation.",

        f"The health scan reveals several systems of interest: {ranked_str}. "
        f"Beginning with {top[0]} - the most anomalous.",

        f"Anomaly analysis shows: {ranked_str}. "
        f"I'll investigate {top[0]} first, then check others if needed.",
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
        f"all healthy. The fault must be elsewhere. Let me move to the next system.",

        f"No faults found in {eliminated_system} after checking {n_nodes_checked} nodes. "
        f"Eliminating this system and redirecting investigation.",

        f"Investigation of {eliminated_system} complete: {n_nodes_checked} components, "
        f"zero faults. The anomaly score may have been a false positive. "
        f"Proceeding to the next highest-scoring system.",

        f"{eliminated_system} shows normal operation across all {n_nodes_checked} "
        f"checked components. Pivoting to the next system in my priority list.",
    ]
    return rng.choice(_ELIMINATIONS)


def compose_system_pivot(
    from_system: str,
    to_system: str,
    to_score: float,
    rng: random.Random,
) -> str:
    """Compose reasoning for pivoting from an eliminated system to the next."""
    pct = f"{to_score:.0%}"
    _PIVOTS = [
        f"Since {from_system} is clear, I'll now investigate {to_system} "
        f"(anomaly score: {pct}). This is the next highest priority.",

        f"Moving investigation from {from_system} to {to_system} ({pct} anomaly). "
        f"Let me get its component structure.",

        f"With {from_system} eliminated, {to_system} becomes the primary suspect "
        f"(anomaly: {pct}). Switching focus.",

        f"No issues in {from_system}. Next candidate: {to_system} at {pct} "
        f"anomaly score. Let me examine its components.",

        f"Redirecting from {from_system} (clear) to {to_system} (anomaly: {pct}). "
        f"The anomaly score suggests potential issues in this system.",
    ]
    return rng.choice(_PIVOTS)


def compose_status_summary_request(system_name: str, rng: random.Random) -> str:
    """Compose reasoning for using the system-wide status summary tool."""
    options = [
        f"Before probing components one by one, I should get a system-wide status summary for {system_name} to prioritize the most suspicious nodes.",
        f"A quick component status summary for {system_name} will help avoid an inefficient blind sweep.",
        f"I'll ask the Oracle for a status summary across {system_name} so I can focus on candidate components first.",
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
) -> str:
    """Compose reasoning after related-system topology is returned."""
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


def compose_warning_response(
    node_name: str,
    confidence: float,
    sensor_readings: Dict[str, float],
    rng: random.Random,
) -> str:
    """Compose reasoning when Oracle returns Warning (low confidence)."""
    pct = f"{confidence:.1%}"
    _WARNINGS = [
        f"The Oracle reports a borderline result for {node_name} - "
        f"confidence is only {pct}, below the definitive threshold. "
        f"I should verify this with raw sensor data before concluding.",

        f"{node_name} returned a Warning status at {pct} confidence. "
        f"This is inconclusive. I need to cross-check with the actual "
        f"sensor readings to confirm or rule out a fault.",

        f"Interesting - {node_name} shows a Warning but the model is only "
        f"{pct} confident. The result is ambiguous. Let me examine the "
        f"sensor data directly to get more evidence.",

        f"The diagnostic model is uncertain about {node_name} ({pct}). "
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
        f"Combined with the Oracle's Warning, this provides sufficient "
        f"evidence to confirm the diagnosis.",

        f"After examining the sensor data, {sensor_findings}. "
        f"Despite the Oracle's low confidence, the sensor evidence "
        f"supports a fault determination at {node_name}.",

        f"Cross-referencing sensor data with the Warning: {sensor_findings}. "
        f"The sensor deviations corroborate the Oracle's suspicion. "
        f"I can now make a more confident diagnosis.",

        f"Sensor verification complete: {sensor_findings}. "
        f"The data confirms abnormal operation at {node_name}, "
        f"validating the Oracle's borderline detection.",

        f"The raw sensor readings ({sensor_findings}) confirm the anomaly "
        f"flagged by the Oracle at {node_name}. "
        f"The fault is real despite the low model confidence.",
    ]
    return rng.choice(_VERIFICATIONS)
