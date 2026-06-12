"""
Scenario Sampler v2 - 2x4 matrix distribution with compositional prompts.

Two major categories:
  Non-Ambiguous: User specifies the target system
  Ambiguous:     User only describes symptoms, no system specified

Four sub-types within each (no_fault only in Non-Ambiguous):
  single_system, cross_system, no_fault, low_confidence

Prompt generation uses compositional templates for high diversity.
"""

import logging
import random
from collections import Counter
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from src.environment.fault_scenario import FaultScenario

logger = logging.getLogger(__name__)

# ============================================================================
# Distribution config
# ============================================================================

DEFAULT_DISTRIBUTION: Dict[str, float] = {
    # Non-Ambiguous (specified system)
    "na_single_system":    0.16,  # 480 / 3000
    "na_cross_system":     0.18,  # 540 / 3000
    "na_no_fault":         0.14,  # 420 / 3000
    "na_low_confidence":   0.09,  # 270 / 3000
    # Ambiguous (no system specified)
    "a_single_system":     0.14,  # 420 / 3000
    "a_cross_system":      0.20,  # 600 / 3000
    "a_low_confidence":    0.09,  # 270 / 3000
}

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

SYSTEM_OBSERVATION_AREAS = {
    "chiller_plant": [
        "central chilled-water plant",
        "chilled-water loop",
        "condenser-water loop",
        "cooling tower area",
    ],
    "boiler_plant": [
        "central hot-water plant",
        "hot-water loop",
        "boiler room",
        "heating-water distribution loop",
    ],
    "sdahu": [
        "single-duct air-handling unit",
        "supply-air duct serving the single-duct AHU zones",
        "outdoor-air section of an air handler",
    ],
    "ddahu": [
        "dual-duct air-handling unit",
        "hot-deck and cold-deck air handler",
        "dual-duct mixing box area",
    ],
    "rtu": [
        "rooftop packaged unit",
        "DX rooftop unit",
        "roof-mounted cooling unit",
    ],
    "fcu": [
        "fan-coil zone unit",
        "terminal fan-coil unit",
        "local zone coil unit",
    ],
    "pfpu": [
        "parallel fan-powered terminal unit",
        "terminal reheat box with parallel fan",
        "zone terminal unit with local fan",
    ],
    "sfpu": [
        "series fan-powered terminal unit",
        "series-flow terminal reheat box",
        "zone terminal unit in series fan mode",
    ],
}

SYSTEM_OBSERVATION_MARKER = "Observable location:"

# Multi-system pairs for no_fault multi-system checks
MULTI_SYSTEM_PAIRS = [
    ("chiller_plant", "fcu"),
    ("boiler_plant", "fcu"),
    ("chiller_plant", "sdahu"),
    ("boiler_plant", "ddahu"),
    ("sdahu", "pfpu"),
    ("sdahu", "sfpu"),
    ("chiller_plant", "ddahu"),
]


# ============================================================================
# Symptom templates - the compositional building blocks
# ============================================================================

_SYMPTOMS_TEMPERATURE = [
    "Supply air temperature deviates from setpoint",
    "Temperature control loop is showing sustained offset",
    "Return air temperature is higher than expected",
    "Zone temperature cannot reach the cooling setpoint",
    "Zone temperature cannot reach the heating setpoint",
    "Temperature differential across the coil is outside normal range",
    "Discharge air temperature is trending upward",
    "Supply water temperature is higher than setpoint",
    "Hot water temperature is lower than expected",
]

_SYMPTOMS_AIRFLOW = [
    "Airflow volume is lower than the minimum setpoint",
    "Fan speed is oscillating unexpectedly",
    "Static pressure cannot be maintained at setpoint",
    "Damper position is not responding to control signals",
    "Return air damper appears to be stuck",
    "Outdoor air fraction is higher than expected",
]

_SYMPTOMS_ENERGY = [
    "Energy consumption has increased unexpectedly",
    "Power draw is higher than baseline for the season",
    "COP has dropped below the efficiency threshold",
    "Equipment runtime has increased significantly",
]

_SYMPTOMS_COMFORT = [
    "Occupant complaints about thermal comfort",
    "Occupant reports room is too hot during afternoon hours",
    "Occupant reports room is too cold in the morning",
    "Users report inconsistent temperature across the zone",
    "Intermittent comfort complaints from multiple occupants",
]

_SYMPTOMS_EQUIPMENT = [
    "Unusual vibration detected in the equipment",
    "Abnormal noise reported from the mechanical room",
    "Condenser water flow is lower than expected",
    "Compressor discharge pressure is abnormally high",
    "Pump differential pressure is outside expected range",
    "Valve actuator feedback signal is inconsistent",
]

_SYMPTOMS_AMBIGUOUS_MULTI = [
    "Multiple zones report insufficient cooling simultaneously",
    "Several air handlers show elevated supply air temperatures",
    "Cooling performance has degraded across multiple areas",
    "Building-wide heating complaints during morning startup",
    "Chilled water delta-T across several coils is lower than expected",
    "Hot water supply temperature has dropped across the building",
]

# --- System-specific symptom pools (domain-aware) ---

_SYMPTOMS_RTU = [
    "Compressor discharge pressure is abnormally high",
    "Supply air temperature deviates from setpoint",
    "Cooling capacity has decreased noticeably",
    "Refrigerant suction pressure is outside normal range",
    "Condenser fan is not maintaining adequate heat rejection",
    "Discharge air temperature is trending upward",
    "Compressor cycling frequency has increased",
    "Return air temperature is higher than expected",
    "Airflow volume is lower than the minimum setpoint",
    "COP has dropped below the efficiency threshold",
    "Equipment runtime has increased significantly",
]

_SYMPTOMS_RTU_REFRIGERANT = [
    "Refrigerant suction pressure is outside normal range",
    "Cooling capacity has decreased noticeably",
    "Supply air temperature is warmer than expected during cooling demand",
    "Compressor cycling frequency has increased",
    "Compressor discharge pressure is outside normal range",
    "DX circuit performance indicates a possible refrigerant-side restriction",
]

_SYMPTOMS_CHILLER = [
    "Chilled water supply temperature is above setpoint",
    "Condenser water flow is lower than expected",
    "Chiller COP has dropped below the efficiency threshold",
    "Cooling tower approach temperature is too high",
    "Chilled water differential pressure is outside normal range",
    "Secondary chilled water loop pressure is fluctuating",
    "Condenser water temperature is significantly above setpoint",
    "Temperature differential across the coil is outside normal range",
    "Energy consumption has increased unexpectedly",
    "Power draw is higher than baseline for the season",
]

_SYMPTOMS_CHILLER_BYPASS = [
    "Bypass valve position feedback is inconsistent with the command",
    "Chilled water supply temperature is above setpoint while the bypass valve is active",
    "Chilled water delta-T is lower than expected, suggesting bypass flow",
    "The chilled water bypass valve is not responding to control signals",
    "Chilled water loop performance indicates unintended bypass flow",
    "Central chilled-water cooling output is degraded with signs of bypass valve malfunction",
]

_SYMPTOMS_CHILLER_TOWER = [
    "Cooling tower approach temperature is too high",
    "Condenser water temperature is significantly above setpoint",
    "Condenser water return temperatures are higher than expected",
    "Cooling tower heat rejection performance appears degraded",
    "Condenser water flow is lower than expected",
]

_SYMPTOMS_CHILLER_SECONDARY = [
    "Chilled water differential pressure is outside normal range",
    "Secondary chilled water loop pressure is fluctuating",
    "Chilled water flow through downstream coils is unstable",
]

_SYMPTOMS_BOILER = [
    "Hot water supply temperature is below setpoint",
    "Hot water temperature is lower than expected",
    "Hot water return temperature is higher than expected",
    "Boiler differential pressure is outside normal range",
    "Heating coil discharge temperature is too low",
    "Zone temperature cannot reach the heating setpoint",
    "Temperature control loop is showing sustained offset",
    "Energy consumption has increased unexpectedly",
    "Power draw is higher than baseline for the season",
    "Unusual vibration detected in the equipment",
]

_SYMPTOMS_AHU = [
    "Supply air temperature deviates from setpoint",
    "Return air temperature is higher than expected",
    "Damper position is not responding to control signals",
    "Airflow volume is lower than the minimum setpoint",
    "Static pressure cannot be maintained at setpoint",
    "Outdoor air fraction is higher than expected",
    "Discharge air temperature is trending upward",
    "Temperature differential across the coil is outside normal range",
    "Fan speed is oscillating unexpectedly",
    "Zone temperature cannot reach the cooling setpoint",
    "Zone temperature cannot reach the heating setpoint",
]

_SYMPTOMS_TERMINAL = [
    "Zone temperature cannot reach the cooling setpoint",
    "Zone temperature cannot reach the heating setpoint",
    "Discharge air temperature is trending upward",
    "Airflow volume is lower than the minimum setpoint",
    "Valve actuator feedback signal is inconsistent",
    "Temperature differential across the coil is outside normal range",
    "Damper position is not responding to control signals",
    "Occupant complaints about thermal comfort",
    "Users report inconsistent temperature across the zone",
]

_SYMPTOMS_FCU = [
    "Zone temperature cannot reach the cooling setpoint",
    "Zone temperature cannot reach the heating setpoint",
    "Valve actuator feedback signal is inconsistent",
    "Return air damper appears to be stuck",
    "Temperature differential across the coil is outside normal range",
    "Discharge air temperature is trending upward",
    "Fan speed is oscillating unexpectedly",
    "Occupant reports room is too hot during afternoon hours",
    "Occupant reports room is too cold in the morning",
]

_SYSTEM_SYMPTOM_MAP = {
    "rtu": _SYMPTOMS_RTU,
    "chiller_plant": _SYMPTOMS_CHILLER,
    "boiler_plant": _SYMPTOMS_BOILER,
    "sdahu": _SYMPTOMS_AHU,
    "ddahu": _SYMPTOMS_AHU,
    "fcu": _SYMPTOMS_FCU,
    "pfpu": _SYMPTOMS_TERMINAL,
    "sfpu": _SYMPTOMS_TERMINAL,
}


_HEATING_HINTS = (
    "heating setpoint",
    "heating target",
    "heating demand",
    "heating output",
    "hot water",
    "reheat",
    "warm-up",
    "too cold",
    "below the heating",
    "heating mode",
)

_COOLING_HINTS = (
    "cooling setpoint",
    "cooling target",
    "cooling demand",
    "cooling output",
    "cooling capacity",
    "chilled water",
    "condenser",
    "compressor",
    "refrigerant",
    "too hot",
    "warmer than expected",
    "peak cooling",
)

_TERMINAL_HEATING_SYMPTOMS = [
    "Zone temperature cannot reach the heating setpoint",
    "Terminal reheat coil discharge temperature is too low",
    "Reheat output is lower than expected",
    "Discharge air temperature is below the heating target",
    "Airflow volume is lower than the minimum setpoint",
    "Damper position is not responding to control signals",
    "Users report inconsistent temperature across the zone",
]

_TERMINAL_COOLING_SYMPTOMS = [
    "Zone temperature cannot reach the cooling setpoint",
    "Cooling output to the zone is lower than expected",
    "Discharge air temperature is trending upward",
    "Supply air temperature is above the cooling target",
    "Airflow volume is lower than the minimum setpoint",
    "Damper position is not responding to control signals",
    "Users report inconsistent temperature across the zone",
]

_TERMINAL_NEUTRAL_SYMPTOMS = [
    "Zone temperature is drifting from its target",
    "Reheat output is unstable",
    "Airflow volume is lower than the minimum setpoint",
    "Damper position is not responding to control signals",
    "Discharge air temperature is trending away from setpoint",
    "Users report inconsistent temperature across the zone",
]

_TERMINAL_DAMPER_SYMPTOMS = [
    "Damper position is not responding to control signals",
    "Airflow volume is lower than the minimum setpoint",
    "Terminal airflow volume cannot be maintained at setpoint",
    "Terminal static pressure cannot be maintained at setpoint",
    "Return air damper appears to be stuck",
    "Mixed air temperature is abnormal, suggesting the damper may be stuck",
    "Terminal airflow is lower than expected",
]


def _filter_symptoms_by_mode(symptoms: list, mode: str) -> list:
    """Remove opposite-thermal phrases while keeping other relevant symptoms."""
    if mode not in ("heating", "cooling"):
        return list(symptoms)

    blocked = _COOLING_HINTS if mode == "heating" else _HEATING_HINTS
    filtered = [
        symptom for symptom in symptoms
        if not any(hint in symptom.lower() for hint in blocked)
    ]
    return filtered or list(symptoms)


def _terminal_symptoms_for_mode(mode: str) -> list:
    if mode == "heating":
        return list(_TERMINAL_HEATING_SYMPTOMS)
    if mode == "cooling":
        return list(_TERMINAL_COOLING_SYMPTOMS)
    return list(_TERMINAL_NEUTRAL_SYMPTOMS)


def _terminal_symptoms_for_fault(scenario: FaultScenario, fault: str, mode: str) -> list:
    """Return a terminal-specific symptom pool for PFPU/SFPU faults."""
    if any(k in fault for k in ("dmpr", "damper", "vavdmpr", "oadmpr", "oablockage", "airflow", "vav")):
        return list(_TERMINAL_DAMPER_SYMPTOMS)
    if any(k in fault for k in ("reheatcoil", "reheatvlv", "reheat", "heating", "hot_water", "boiler", "hwc", "hwl")):
        return _filter_symptoms_by_mode(list(_TERMINAL_HEATING_SYMPTOMS), "heating")
    if any(k in fault for k in ("cooling", "chilled", "evap", "cond", "coolingtower", "chwc")):
        return _filter_symptoms_by_mode(list(_TERMINAL_COOLING_SYMPTOMS), "cooling")
    return _terminal_symptoms_for_mode(mode)


def _get_symptoms_for_system(system_id: str) -> list:
    """Return domain-appropriate symptoms for a given system."""
    return _SYSTEM_SYMPTOM_MAP.get(
        system_id,
        _SYMPTOMS_TEMPERATURE + _SYMPTOMS_AIRFLOW + _SYMPTOMS_ENERGY,
    )


def _get_symptoms_for_scenario(scenario: FaultScenario) -> list:
    """Return symptoms that fit both the system and the fault physics."""
    pool = list(_get_symptoms_for_system(scenario.root_cause_system))
    fault = str(scenario.fault_type).lower()
    mode = _fault_thermal_mode(scenario)

    if scenario.root_cause_system in ("pfpu", "sfpu"):
        selected = _terminal_symptoms_for_fault(scenario, fault, mode)
        return selected or pool

    def pick(keywords: Tuple[str, ...]) -> list:
        return [s for s in pool if any(k in s.lower() for k in keywords)]

    if scenario.root_cause_system == "chiller_plant" and "bypass" in fault:
        selected = list(_SYMPTOMS_CHILLER_BYPASS)
    elif scenario.root_cause_system == "chiller_plant" and any(
        k in fault for k in ("coolingtower", "tower", "cond")
    ):
        selected = list(_SYMPTOMS_CHILLER_TOWER)
    elif scenario.root_cause_system == "chiller_plant" and "secondary_chilled" in fault:
        selected = list(_SYMPTOMS_CHILLER_SECONDARY)
    elif scenario.root_cause_system == "rtu" and any(
        k in fault
        for k in (
            "suctionpipe",
            "liquidpipe",
            "undercharge",
            "overcharge",
            "refrigerant",
            "filterrestriction",
        )
    ):
        selected = list(_SYMPTOMS_RTU_REFRIGERANT)
    elif any(k in fault for k in ("coi_", "coil", "vlvstuck", "vlvleak", "reheatcoil")):
        selected = pick(("coil", "temperature", "setpoint", "valve", "discharge air"))
    elif any(k in fault for k in ("heating", "reheat", "hot_water", "boiler", "hwc")):
        selected = pick(("heating", "hot water", "too cold", "warm"))
    elif any(k in fault for k in (
        "cooling", "chiller", "chilled", "evap", "cond", "coolingtower", "chwc",
    )):
        selected = pick(("cooling", "chilled", "condenser", "compressor", "too hot", "cop"))
    elif any(k in fault for k in ("damper", "dmpr", "oa", "outside")):
        selected = pick(("damper", "outdoor air", "airflow", "static pressure"))
    elif any(k in fault for k in ("fan", "airflow", "vav")):
        selected = pick(("airflow", "fan", "static pressure", "damper"))
    elif "pressure" in fault:
        selected = pick(("pressure", "pump", "flow"))
    elif "temp" in fault or "bias" in fault:
        selected = pick(("temperature", "setpoint", "supply air", "return air"))
    else:
        selected = []

    if selected:
        return _filter_symptoms_by_mode(selected, mode)
    return _filter_symptoms_by_mode(pool, mode)

_CONTEXTS = [
    "during peak cooling season",
    "during heating season transition",
    "during morning startup",
    "during overnight unoccupied operation",
    "under partial load conditions",
    "after recent maintenance activities",
    "during high ambient temperature conditions",
    "after a recent control setpoint change",
    "during afternoon peak load",
    "during building warm-up period",
]

_HEATING_CONTEXTS = [
    "during heating season transition",
    "during morning startup",
    "during overnight unoccupied operation",
    "under partial load conditions",
    "after recent maintenance activities",
    "after a recent control setpoint change",
    "during building warm-up period",
]

_COOLING_CONTEXTS = [
    "during peak cooling season",
    "during morning startup",
    "under partial load conditions",
    "after recent maintenance activities",
    "during high ambient temperature conditions",
    "after a recent control setpoint change",
    "during afternoon peak load",
]


def _fault_thermal_mode(scenario: FaultScenario) -> str:
    """Return heating/cooling/neutral from the fault label."""
    fault = str(scenario.fault_type).lower()
    if any(k in fault for k in ("heating", "reheat", "hot_water", "boiler", "hwc", "hwl")):
        return "heating"
    if any(k in fault for k in (
        "cooling", "chiller", "chilled", "evap", "cond", "coolingtower",
        "chwc", "overcharge", "undercharge", "filterrestriction",
        "bypass",
    )):
        return "cooling"
    return "neutral"


def _get_context_for_scenario(scenario: FaultScenario, rng: random.Random) -> str:
    """Choose an operating context that does not contradict fault physics."""
    mode = _fault_thermal_mode(scenario)
    if mode == "heating":
        return rng.choice(_HEATING_CONTEXTS)
    if mode == "cooling":
        return rng.choice(_COOLING_CONTEXTS)
    return rng.choice(_CONTEXTS)


def _observable_area(system_id: str, rng: random.Random) -> str:
    """Return a learnable but non-ID observation cue for ambiguous prompts."""
    options = SYSTEM_OBSERVATION_AREAS.get(system_id)
    if not options:
        return "the affected HVAC equipment area"
    return rng.choice(options)


def _with_observable_area(prompt: str, system_id: str, rng: random.Random) -> str:
    """Append an observable equipment/location cue without exposing system IDs."""
    area = _observable_area(system_id, rng)
    return f"{prompt} {SYSTEM_OBSERVATION_MARKER} {area}."


def _diagnostic_path_downstream_system(scenario: FaultScenario) -> str:
    """Return the visible downstream system from the diagnostic path."""
    path = getattr(scenario, "diagnostic_path", None)
    root = scenario.root_cause_system
    for pnode in getattr(path, "nodes", []) or []:
        system_id = getattr(pnode, "system_id", "")
        if system_id and system_id != root:
            return system_id

    for system_id in scenario.affected_systems or []:
        if system_id and system_id != root:
            return system_id
    return root

_NO_FAULT_REQUESTS_SINGLE = [
    "Please check if {system_name} is operating normally. Recent energy data looks unusual.",
    "Run a diagnostic check on {system_name}. Users reported intermittent issues.",
    "Verify the operational status of {system_name}. Maintenance team wants an operating check.",
    "Please inspect {system_name} for any potential issues. Seasonal commissioning check.",
    "Check {system_name} performance. BMS triggered a transient alarm earlier.",
    "Diagnose {system_name} operating status. Routine preventive maintenance request.",
    "Investigate {system_name} operation. Energy consumption seems slightly elevated.",
    "Perform a comprehensive check on {system_name}. Preparing for peak season.",
]

_NO_FAULT_REQUESTS_MULTI = [
    "Please check the cooling supply chain: {sys_a} and {sys_b}. Verify both are operating normally.",
    "Run an operating check on {sys_a} and {sys_b}. Users reported intermittent comfort issues.",
    "Inspect {sys_a} and its downstream {sys_b} for any potential problems. Seasonal checkup.",
    "Verify the operational status of both {sys_a} and {sys_b}. Maintenance team requests.",
    "Check the {sys_a}-to-{sys_b} supply chain. Performance metrics look slightly off.",
]


# ============================================================================
# Prompt composers - one per scenario slot
# ============================================================================

def _compose_na_single_prompt(scenario: FaultScenario, rng: random.Random) -> str:
    """Non-Ambiguous x single_system: [System] + domain-appropriate symptom."""
    sys_name = SYSTEM_DISPLAY_NAMES.get(scenario.root_cause_system, scenario.root_cause_system)
    symptom_pool = _get_symptoms_for_scenario(scenario)
    symptom = rng.choice(symptom_pool)
    ctx = _get_context_for_scenario(scenario, rng)

    templates = [
        f"[{sys_name}] {symptom} {ctx}. Please diagnose the issue.",
        f"[{sys_name}] System alert - {symptom}. Investigate and identify the root cause.",
        f"[{sys_name}] A fault has been detected: {symptom} {ctx}. Diagnose the problem.",
        f"[{sys_name}] {symptom}. This was observed {ctx}. Please investigate.",
        f"[{sys_name}] Maintenance report: {symptom} {ctx}. Perform fault diagnosis.",
    ]
    return rng.choice(templates)


def _compose_na_cross_prompt(scenario: FaultScenario, rng: random.Random) -> str:
    """Non-Ambiguous x cross_system: [Downstream system] + upstream hint."""
    downstream = _diagnostic_path_downstream_system(scenario)
    ds_name = SYSTEM_DISPLAY_NAMES.get(downstream, downstream)
    root_name = SYSTEM_DISPLAY_NAMES.get(scenario.root_cause_system, scenario.root_cause_system)

    mode = _fault_thermal_mode(scenario)
    if mode == "heating":
        symptom_pool = [
            "Zone temperature cannot reach the heating setpoint",
            "Heating coil discharge temperature is too low",
            "Discharge air temperature is below the heating target",
            "Downstream heating output is lower than expected",
            "Temperature control loop is showing sustained heating offset",
        ]
    elif mode == "cooling":
        symptom_pool = [
            "Zone temperature cannot reach the cooling setpoint",
            "Supply air temperature is above cooling setpoint",
            "Cooling coil discharge temperature is higher than expected",
            "Chilled water delta-T across the cooling coil is lower than expected",
            "Downstream cooling output is lower than expected",
        ]
    else:
        symptom_pool = _get_symptoms_for_system(downstream)
    symptom = rng.choice(symptom_pool)
    ctx = _get_context_for_scenario(scenario, rng)

    templates = [
        f"[{ds_name}] {symptom} {ctx}. Initial analysis suggests the root cause may be upstream. Investigate.",
        f"[{ds_name}] {symptom}. Performance degradation could be linked to an upstream supply issue. Diagnose.",
        f"[{ds_name}] {symptom} {ctx}. Check if the issue originates from the {ds_name} itself or upstream systems.",
        f"[{ds_name}] Multiple indicators suggest {symptom}. The {ds_name} receives supply from upstream. Please investigate the full chain.",
        f"[{ds_name}] {symptom} {ctx}. If {ds_name} components appear normal, trace upstream to find root cause.",
    ]
    return rng.choice(templates)


def _compose_na_no_fault_prompt(
    scenario: FaultScenario,
    rng: random.Random,
    multi_system: bool = False,
    second_system: str = "",
) -> str:
    """Non-Ambiguous x no_fault: 'Check if X is OK' request."""
    sys_name = SYSTEM_DISPLAY_NAMES.get(scenario.root_cause_system, scenario.root_cause_system)

    if multi_system and second_system:
        sys_b_name = SYSTEM_DISPLAY_NAMES.get(second_system, second_system)
        template = rng.choice(_NO_FAULT_REQUESTS_MULTI)
        return template.format(sys_a=sys_name, sys_b=sys_b_name)
    else:
        template = rng.choice(_NO_FAULT_REQUESTS_SINGLE)
        return template.format(system_name=sys_name)


def _compose_na_low_conf_prompt(scenario: FaultScenario, rng: random.Random) -> str:
    """Non-Ambiguous x low_confidence: [System] + weak signal."""
    sys_name = SYSTEM_DISPLAY_NAMES.get(scenario.root_cause_system, scenario.root_cause_system)
    symptom = rng.choice(_get_symptoms_for_scenario(scenario))
    ctx = _get_context_for_scenario(scenario, rng)

    templates = [
        f"[{sys_name}] Borderline evidence observed: {symptom} {ctx}. The fault signal is weak. Please investigate and verify.",
        f"[{sys_name}] A potential fault has been flagged with low certainty: {symptom} {ctx}. Perform detailed verification.",
        f"[{sys_name}] Diagnostic confidence is unusually low for this symptom: {symptom} {ctx}. Check sensor data and explain whether the evidence supports a fault.",
        f"[{sys_name}] Weak abnormal signal detected: {symptom} {ctx}. The readings are near the decision boundary. Investigate thoroughly.",
        f"[{sys_name}] Monitoring system flagged a marginal issue: {symptom} {ctx}. Diagnostic confidence is below threshold. Verify with raw data.",
    ]
    return rng.choice(templates)


def _compose_a_single_prompt(scenario: FaultScenario, rng: random.Random) -> str:
    """Ambiguous x single_system: domain-appropriate symptoms, NO system name."""
    symptom = rng.choice(_get_symptoms_for_scenario(scenario))
    ctx = _get_context_for_scenario(scenario, rng)

    templates = [
        f"{symptom} {ctx}. Identify the affected system and diagnose the root cause.",
        f"{symptom}. This has been observed {ctx}. Please investigate which system is responsible.",
        f"Maintenance team reports: {symptom} {ctx}. Determine which HVAC system is at fault.",
        f"BMS alert: {symptom} {ctx}. The source system is unknown. Investigate.",
        f"Performance issue detected: {symptom} {ctx}. No specific system identified. Please diagnose.",
        f"Building operations flagged: {symptom}. This occurred {ctx}. Find the root cause.",
    ]
    return _with_observable_area(
        rng.choice(templates),
        scenario.root_cause_system,
        rng,
    )


def _compose_a_cross_prompt(scenario: FaultScenario, rng: random.Random) -> str:
    """Ambiguous x cross_system: multi-zone symptoms suggesting upstream cause."""
    # Use system-aware multi-zone symptoms
    root = scenario.root_cause_system
    if root == "boiler_plant":
        symptom_pool = [
            "Building-wide heating complaints during morning startup",
            "Hot water supply temperature has dropped across the building",
            "Multiple zones report insufficient heating simultaneously",
            "Heating performance has degraded across multiple areas",
            "Several reheat coils show reduced output",
        ]
    elif root == "chiller_plant":
        symptom_pool = [
            "Multiple zones report insufficient cooling simultaneously",
            "Several air handlers show elevated supply air temperatures",
            "Cooling performance has degraded across multiple areas",
            "Chilled water delta-T across several coils is lower than expected",
            "Multiple AHU cooling coils show degraded performance",
        ]
    else:
        symptom_pool = _SYMPTOMS_AMBIGUOUS_MULTI
    symptom = rng.choice(symptom_pool)
    ctx = _get_context_for_scenario(scenario, rng)
    downstream = _diagnostic_path_downstream_system(scenario)

    templates = [
        f"{symptom} {ctx}. Determine the root cause across the HVAC systems.",
        f"{symptom}. This was observed {ctx}. The issue may originate from a central system. Investigate.",
        f"Building-wide alert: {symptom} {ctx}. Multiple systems may be involved. Identify the source.",
        f"{symptom} {ctx}. Cross-system diagnosis may be required. Find the root cause.",
        f"Operations report: {symptom}. The pattern suggests a shared upstream issue {ctx}. Diagnose.",
    ]
    return _with_observable_area(rng.choice(templates), downstream, rng)


def _compose_a_low_conf_prompt(scenario: FaultScenario, rng: random.Random) -> str:
    """Ambiguous x low_confidence: weak but physically grounded symptom.

    Earlier versions used very generic prompts such as "slow degradation
    trend". Those are too under-specified: the supervised trajectory then
    jumps to the correct system using information the user never provided,
    which teaches an implicit shortcut the model cannot reproduce at eval
    time. Keep the system alias hidden, but expose a domain-specific symptom
    family so topology-guided search is learnable rather than guessed.
    """
    symptom_pool = list(_get_symptoms_for_scenario(scenario))
    if not symptom_pool:
        symptom_pool = _get_symptoms_for_system(scenario.root_cause_system)
    symptom = rng.choice(symptom_pool)
    ctx = _get_context_for_scenario(scenario, rng)

    templates = [
        f"{symptom} {ctx}. The signal is weak and the source system is not identified. Investigate and verify with evidence.",
        f"{symptom}. Observed {ctx}. Standard diagnostics are borderline; identify the likely system and verify the candidate.",
        f"Subtle issue reported: {symptom} {ctx}. No specific system has been named. Trace the topology and support the conclusion with evidence.",
        f"{symptom} {ctx}. The anomaly is near the decision boundary. Find the responsible component without assuming a system.",
        f"Gradual change detected: {symptom} {ctx}. Determine whether this is a real fault and verify any uncertain candidate.",
    ]
    return _with_observable_area(
        rng.choice(templates),
        scenario.root_cause_system,
        rng,
    )


# ============================================================================
# Prompt routing
# ============================================================================

_PROMPT_COMPOSERS = {
    "na_single_system":   _compose_na_single_prompt,
    "na_cross_system":    _compose_na_cross_prompt,
    "na_no_fault":        _compose_na_no_fault_prompt,
    "na_low_confidence":  _compose_na_low_conf_prompt,
    "a_single_system":    _compose_a_single_prompt,
    "a_cross_system":     _compose_a_cross_prompt,
    "a_low_confidence":   _compose_a_low_conf_prompt,
}


def compose_scenario_prompt(scenario: FaultScenario, rng: random.Random) -> str:
    """Generate a compositional prompt for the given scenario."""
    composer = _PROMPT_COMPOSERS.get(scenario.scenario_type, _compose_na_single_prompt)
    return composer(scenario, rng)


# ============================================================================
# Scenario creation helpers
# ============================================================================

def create_ambiguous_scenarios(
    base_scenarios: List[FaultScenario],
) -> List[FaultScenario]:
    """Create ambiguous versions of single_system and cross_system scenarios.

    Clones the scenario with scenario_type prefixed by 'a_'.
    """
    ambiguous = []
    for s in base_scenarios:
        if s.scenario_type == "single_system":
            a = replace(s,
                scenario_id=f"a_{s.scenario_id}",
                scenario_type="a_single_system",
            )
            ambiguous.append(a)
        elif s.scenario_type == "cross_system":
            a = replace(s,
                scenario_id=f"a_{s.scenario_id}",
                scenario_type="a_cross_system",
            )
            ambiguous.append(a)
    return ambiguous


def create_low_confidence_scenarios(
    base_scenarios: List[FaultScenario],
) -> List[FaultScenario]:
    """Create low-confidence versions (both NA and Ambiguous).

    Clones single_system scenarios with low_confidence flag.
    """
    lc = []
    for s in base_scenarios:
        if s.scenario_type == "single_system":
            # Non-Ambiguous low_confidence
            na_lc = replace(s,
                scenario_id=f"na_lc_{s.scenario_id}",
                scenario_type="na_low_confidence",
            )
            lc.append(na_lc)
            # Ambiguous low_confidence
            a_lc = replace(s,
                scenario_id=f"a_lc_{s.scenario_id}",
                scenario_type="a_low_confidence",
            )
            lc.append(a_lc)
    return lc


def create_no_fault_scenarios(
    base_scenarios: List[FaultScenario],
) -> List[FaultScenario]:
    """Create no_fault scenarios for Non-Ambiguous checks.

    Uses real no-fault scenarios as templates.

    Older versions cloned fault scenarios and only changed ``fault_type`` to
    no_fault. That left the source CSV, root-cause node, and diagnostic path
    pointing at a real fault, which is poisonous for RL because the live Oracle
    still observes faulted data. This function keeps the sample count shape but
    always anchors cloned scenarios to baseline data and no-fault ground truth.
    """
    nf = []
    normal_by_system: Dict[str, List[FaultScenario]] = {}
    for s in base_scenarios:
        is_true_no_fault = (
            s.scenario_type == "no_fault"
            or str(s.root_cause_node).lower() in ("none", "")
            or str(s.fault_type).lower() in ("normal", "no_fault")
        )
        if is_true_no_fault:
            normal_by_system.setdefault(s.root_cause_system, []).append(s)

    template_offsets = Counter()
    for s in base_scenarios:
        if s.scenario_type != "single_system":
            continue
        sys_id = s.root_cause_system
        templates = normal_by_system.get(sys_id, [])
        if not templates:
            logger.warning(
                "Skipping na_no_fault clone for %s: no baseline scenario found",
                sys_id,
            )
            continue

        idx = template_offsets[sys_id] % len(templates)
        template_offsets[sys_id] += 1
        template = templates[idx]

        window_size = max(
            1,
            (template.time_window_end or 0) - (template.time_window_start or 0),
        )
        if window_size <= 1:
            window_size = 15

        # Keep the source data/window/path from the baseline template.  A
        # fault scenario may come from a longer file than the baseline file;
        # inheriting its row window would bind no-fault prompts to an
        # out-of-range or unrelated operating condition.
        start = template.time_window_start
        nf_scenario = replace(
            template,
            scenario_id=f"nf_{s.scenario_id}",
            scenario_type="na_no_fault",
            root_cause_node="none",
            fault_type="Normal",
            fault_intensity="none",
            affected_systems=[sys_id],
            source_file=template.source_file,
            time_window_start=start,
            time_window_end=template.time_window_end or (start + window_size),
            difficulty="easy",
        )
        nf.append(nf_scenario)
    return nf


# ============================================================================
# Stratified sampling
# ============================================================================

def sample_scenarios(
    all_scenarios: List[FaultScenario],
    n_total: int = 3000,
    distribution: Optional[Dict[str, float]] = None,
    seed: int = 42,
) -> List[FaultScenario]:
    """Stratified sampling with compositional prompt generation.

    Groups scenarios by scenario_type, samples according to distribution,
    and generates unique prompts for each.
    """
    if distribution is None:
        distribution = DEFAULT_DISTRIBUTION

    rng = random.Random(seed)

    # Also map legacy types to new types for backward compat
    type_map = {
        "single_system": "na_single_system",
        "cross_system": "na_cross_system",
    }

    # Group by scenario type
    by_type: Dict[str, List[FaultScenario]] = {}
    for s in all_scenarios:
        st = type_map.get(s.scenario_type, s.scenario_type)
        by_type.setdefault(st, []).append(s)

    sampled = []
    for stype, fraction in distribution.items():
        n_target = int(n_total * fraction)
        pool = by_type.get(stype, [])
        if not pool:
            logger.warning(f"No scenarios for type '{stype}', skipping {n_target}")
            continue

        # Over-sample with replacement if pool is too small
        if len(pool) < n_target:
            selected = rng.choices(pool, k=n_target)
        else:
            selected = rng.sample(pool, n_target)

        # Generate unique prompts
        for i, s in enumerate(selected):
            prompt_rng = random.Random(rng.randint(0, 2**31))
            prompt_source = replace(s, scenario_type=stype)
            prompt = compose_scenario_prompt(prompt_source, prompt_rng)
            s_with_prompt = replace(s,
                scenario_id=f"{s.scenario_id}_{stype}_{i}",
                description=prompt,
                scenario_type=stype,
            )
            sampled.append(s_with_prompt)

    rng.shuffle(sampled)
    logger.info(f"Sampled {len(sampled)} scenarios:")
    type_counts = Counter(s.scenario_type for s in sampled)
    for st, count in sorted(type_counts.items()):
        logger.info(f"  {st}: {count}")
    return sampled
