"""
Fault Scenario — Define and manage fault scenarios for agent evaluation.

Each FaultScenario encapsulates:
  - Which system/node is faulted and what the fault type is
  - A guided diagnostic path from symptom to root cause
  - The ground truth diagnostic path and root cause
  - Pre-computed feature vectors for each node (for fast model prediction)
"""

import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.environment.diagnostic_path import (
    DiagnosticPath, DiagnosticPathGenerator, PathNode, generate_no_fault_path,
)
from src.node_models.data_loader import (
    discover_fault_files,
    FaultFileInfo,
    get_fault_file_row_count,
)
from src.node_models.feature_engineer import build_node_features, compute_window_features
from src.utils.io_utils import save_json, load_json, setup_logger

logger = setup_logger(__name__)


@dataclass
class FaultScenario:
    """A single fault diagnosis scenario for the agent."""
    scenario_id: str
    scenario_type: str                  # "single_system", "cross_system", "no_fault", "ambiguous"
    description: str                    # Human-readable fault description (user query)

    # Ground truth
    root_cause_system: str              # System where root cause lies
    root_cause_node: str                # Node ID of root cause
    fault_type: str                     # Fault type label
    fault_intensity: str                # Fault intensity
    affected_systems: List[str]         # All systems affected
    optimal_path: List[str]             # Optimal sequence of tool calls

    # Guided diagnostic path
    diagnostic_path: Optional[DiagnosticPath] = None

    # Data references
    source_file: str = ""               # Source CSV filename
    time_window_start: int = 0          # Row index of scenario start
    time_window_end: int = 0            # Row index of scenario end

    # Difficulty
    difficulty: str = "medium"          # "easy", "medium", "hard"


@dataclass
class FaultScenarioState:
    """
    Runtime state for a fault scenario, providing sensor data to the agent tools.

    Stores pre-computed feature vectors for each node at a specific time window.
    """
    scenario: FaultScenario
    node_features: Dict[str, np.ndarray]     # node_id → feature vector
    sensor_readings: Dict[str, Dict[str, float]]  # node_id → {sensor: value}
    system_data_cache: Dict[str, pd.DataFrame]    # system_id → data slice

    def get_node_features(self, node_id: str) -> Optional[np.ndarray]:
        """Get pre-computed features for a node, or None if unavailable."""
        return self.node_features.get(node_id)

    def get_sensor_readings(self, node_id: str) -> Dict[str, float]:
        """Get raw sensor readings for a node at the scenario's time window."""
        return self.sensor_readings.get(node_id, {})


def required_nrows_for_scenario(
    scenario: FaultScenario,
    minimum: Optional[int] = None,
    fallback_window: int = 15,
) -> Optional[int]:
    """Return the row count needed to cover a scenario's declared window.

    Callers may still provide a minimum for smoke-test throughput control, but
    the returned value must never be smaller than ``time_window_end``.  Loading
    fewer rows would force the Oracle onto an unrelated fallback window.
    """
    try:
        start = int(getattr(scenario, "time_window_start", 0) or 0)
    except (TypeError, ValueError):
        start = 0
    try:
        end = int(getattr(scenario, "time_window_end", 0) or 0)
    except (TypeError, ValueError):
        end = 0
    try:
        min_rows = int(minimum or 0)
    except (TypeError, ValueError):
        min_rows = 0

    needed = end if end > start else start + max(1, int(fallback_window))
    needed = max(needed, min_rows)
    return needed if needed > 0 else None


def _sample_valid_window_start(
    rng: random.Random,
    row_count: int,
    window_size: int,
    preferred_min_start: int = 1000,
) -> Optional[int]:
    """Sample a time-window start that is valid for the source file length."""
    max_start = int(row_count) - int(window_size)
    if max_start < 0:
        return None
    lower = min(max(0, int(preferred_min_start)), max_start)
    return rng.randint(lower, max_start)


def create_scenario_state(
    scenario: FaultScenario,
    system_data: Dict[str, pd.DataFrame],
    topology_builder,
    registry=None,
) -> FaultScenarioState:
    """
    Create a FaultScenarioState from real CSV data.

    Loads real sensor readings and pre-computes Oracle features for each
    component in the affected system(s) at the scenario's time window.

    Args:
        scenario: The fault scenario to create state for.
        system_data: Pre-loaded system DataFrames {system_id → full DataFrame}.
        topology_builder: TopologyBuilder with the built graph.
        registry: Optional ModelRegistry to get feature column names.

    Returns:
        FaultScenarioState with real sensor data and model features.
    """
    node_features = {}
    sensor_readings = {}
    data_cache = {}

    # Determine which systems to load sensor data for
    systems_to_load = set(system_data.keys())

    for sys_id in systems_to_load:
        sys_df = system_data.get(sys_id)
        if sys_df is None or sys_df.empty:
            continue

        # Slice to the scenario's declared time window.  Do not silently wrap
        # around when a caller loaded too few rows: that binds the scenario to
        # an unrelated operating condition and can make the Oracle contradict
        # the ground-truth root cause.
        start = scenario.time_window_start
        end = scenario.time_window_end
        if end > start and end <= len(sys_df):
            window_df = sys_df.iloc[start:end]
        else:
            raise ValueError(
                "Scenario time window is unavailable for "
                f"{scenario.scenario_id}/{sys_id}: start={start}, end={end}, "
                f"loaded_rows={len(sys_df)}"
            )

        if window_df.empty:
            continue

        data_cache[sys_id] = window_df

        # Extract sensor readings and features for each component
        components = topology_builder.get_system_components(sys_id)
        for comp in components:
            nid = comp["node_id"]
            sensor_names = comp.get("sensor_names", [])

            # Real sensor readings: average over the time window
            readings = {}
            for sn in sensor_names:
                if sn in window_df.columns:
                    try:
                        col = pd.to_numeric(window_df[sn], errors='coerce')
                        val = col.mean()
                        if pd.notna(val):
                            readings[sn] = round(float(val), 4)
                    except (TypeError, ValueError):
                        pass
            if readings:
                # Physical constraint: wet-bulb ≤ dry-bulb temperature
                if "OA_TEMP" in readings and "OA_TEMP_WB" in readings:
                    db = readings["OA_TEMP"]
                    wb = readings["OA_TEMP_WB"]
                    if wb > db:
                        readings["OA_TEMP_WB"] = round(db - 3.0, 4)
                sensor_readings[nid] = readings

        # Pre-compute Oracle features for the system
        oracle_id = f"{sys_id}::oracle"
        if registry and registry.has_model(oracle_id):
            meta = registry.get_metadata(oracle_id)
            feature_cols = meta.get("feature_cols", []) if meta else []
            if feature_cols:
                # Extract unique sensor base names by stripping statistical suffixes
                stat_suffixes = ("_mean", "_std", "_min", "_max", "_range",
                                 "_slope", "_delta", "_skew", "_kurtosis")
                sensor_names_from_features = set()
                for fc in feature_cols:
                    # Skip non-sensor features like hour_of_day, day_of_week
                    if fc in ("hour_of_day", "day_of_week", "is_occupied", "timestamp"):
                        continue
                    base = fc
                    for sfx in stat_suffixes:
                        if base.endswith(sfx):
                            base = base[:-len(sfx)]
                            break
                    if base in window_df.columns:
                        sensor_names_from_features.add(base)

                sensor_list = list(sensor_names_from_features)
                if not sensor_list:
                    # Fallback: use all numeric columns from window_df
                    sensor_list = [
                        c for c in window_df.select_dtypes(include=["number"]).columns
                    ]

                # Build features from the window
                try:
                    feat_df = compute_window_features(
                        window_df, sensor_list,
                        window_size=min(15, len(window_df)),
                        stride=min(15, len(window_df)),
                    )
                    if not feat_df.empty:
                        # Align feature columns
                        available = [c for c in feature_cols if c in feat_df.columns]
                        if available:
                            # Pad with zeros for missing features
                            full_feat = np.zeros(len(feature_cols), dtype=np.float32)
                            for j, c in enumerate(feature_cols):
                                if c in feat_df.columns:
                                    full_feat[j] = float(feat_df[c].iloc[0])
                            node_features[oracle_id] = full_feat
                except Exception:
                    pass  # Feature computation failed, Oracle won't be used

    return FaultScenarioState(
        scenario=scenario,
        node_features=node_features,
        sensor_readings=sensor_readings,
        system_data_cache=data_cache,
    )
# ============================================================================

# Maps fault types to human-readable symptom descriptions
SYMPTOM_TEMPLATES = {
    # Chiller Plant
    "coolingtower_bias": [
        "The condenser water temperature readings from Cooling Tower 1 seem inconsistent with expected values.",
        "Building operators report that cooling tower performance appears degraded, with unexpected condenser water temperatures.",
    ],
    "chiller_bias": [
        "The chilled water supply temperature from Chiller 1 is not matching its setpoint.",
        "Zone temperatures are drifting upward despite the chiller plant running, suggesting a chilled water temperature issue.",
    ],
    "secondary_chilled_water_pressure_bias": [
        "The differential pressure in the secondary chilled water loop appears abnormal.",
        "Secondary chilled water pumps are operating at unusual speeds, and pressure readings seem off.",
    ],
    "bypass_leakage": [
        "The condenser water three-way valve appears to be allowing unintended flow.",
        "Condenser water temperatures are higher than expected, possibly due to bypass valve issues.",
    ],
    "bypass_stuck": [
        "The condenser water three-way valve is not responding properly to control signals.",
        "The bypass valve in the condenser loop seems stuck, affecting cooling tower performance.",
    ],
    "coolingtower_fouling": [
        "Cooling tower heat rejection capacity has decreased noticeably.",
        "Condenser water return temperatures are higher than expected, suggesting possible cooling tower fouling.",
    ],
    "coolingtower_PI": [
        "The condenser water supply temperature is oscillating or not settling at its setpoint.",
        "The cooling tower fan speed control appears unstable.",
    ],
    # Boiler Plant
    "boiler_bias": [
        "The hot water supply temperature from Boiler 1 does not match expected values.",
        "Heating performance in downstream zones has degraded; boiler sensor readings may be inaccurate.",
    ],
    "hot_water_temp_bias": [
        "The hot water loop supply temperature sensor shows unexpected readings.",
        "Zone heating is inadequate despite the boiler running at expected capacity.",
    ],
    "hot_water_pressure_bias": [
        "The hot water loop differential pressure readings seem incorrect.",
        "Hot water pump speeds are fluctuating abnormally.",
    ],
    "boiler_foul": [
        "The boiler is consuming more gas than expected for the current heating load.",
        "Hot water supply temperature is lower than the setpoint despite full boiler operation.",
    ],
    "boiler_PI": [
        "The boiler supply temperature is oscillating and not maintaining setpoint.",
        "The boiler control loop appears to be hunting, with unstable temperature output.",
    ],
    # SDAHU
    "sa_bias": [
        "The AHU supply air temperature readings don't match zone feedback.",
        "Zone temperatures are not being maintained properly; supply air temperature may be incorrectly measured.",
    ],
    "oa_bias": [
        "The outdoor air temperature sensor on the AHU appears to be reading incorrectly.",
        "Economizer operation seems abnormal, possibly due to an outdoor air temperature sensor issue.",
    ],
    "coi_leakage": [
        "The cooling coil valve appears to allow flow even when commanded fully closed.",
        "Supply air temperature is warmer than expected during cooling demand.",
    ],
    "coi_stuck": [
        "The cooling coil valve is not modulating properly in response to control signals.",
        "Supply air temperature cannot reach setpoint; the cooling coil valve may be stuck.",
    ],
    "damper_stuck": [
        "The outdoor air damper does not appear to be responding to control commands.",
        "Mixed air temperature is abnormal, suggesting the outdoor air damper may be stuck.",
    ],
    "vav_damper_stuck": [
        "The terminal damper does not appear to be responding to control commands.",
        "Terminal airflow volume is lower than expected, suggesting the damper may be stuck.",
    ],
    "reheat_valve_stuck": [
        "The reheat valve is not modulating properly in response to control signals.",
        "Zone temperature cannot reach the heating setpoint because reheat output is limited.",
    ],
    "reheat_coil_fouling": [
        "The reheat coil heat transfer appears degraded.",
        "Zone temperature cannot reach the heating setpoint because the reheat coil output is too low.",
    ],
    # General / cross-system
    "normal": [
        "The building automation system reports normal operation, but a routine diagnostic check is requested.",
        "Please verify that all HVAC systems are operating normally.",
    ],
}

_HEATING_SYMPTOM_PHRASES = [
    "heating setpoint",
    "hot water",
    "reheat",
    "warm-up",
    "heating demand",
    "too cold",
    "below the setpoint",
]

_COOLING_SYMPTOM_PHRASES = [
    "cooling setpoint",
    "chilled water",
    "cooling demand",
    "peak cooling",
    "too hot",
    "warmer than expected",
    "cooling capacity",
]


def _filter_templates_by_mode(templates: List[str], fault_type: str) -> List[str]:
    """Keep symptom templates aligned with the fault's thermal direction."""
    fault = fault_type.lower()
    if any(k in fault for k in ("dmprstuck", "vavdmpr", "oadmpr", "oablockage", "damper")):
        blocked = _HEATING_SYMPTOM_PHRASES + _COOLING_SYMPTOM_PHRASES
        filtered = [
            tmpl for tmpl in templates
            if any(word in tmpl.lower() for word in ("damper", "airflow", "pressure", "mixed air", "outdoor air", "terminal"))
        ]
        return filtered or list(templates)
    if any(k in fault for k in ("heating", "reheat", "hot_water", "boiler", "hwc", "hwl")):
        blocked = _COOLING_SYMPTOM_PHRASES
    elif any(k in fault for k in ("cooling", "chiller", "chilled", "evap", "cond", "coolingtower", "chwc", "coi")):
        blocked = _HEATING_SYMPTOM_PHRASES
    else:
        return list(templates)
    filtered = [
        tmpl for tmpl in templates
        if not any(phrase in tmpl.lower() for phrase in blocked)
    ]
    return filtered or list(templates)

# Default symptom for unmatched fault types
DEFAULT_SYMPTOMS = [
    "Abnormal operation has been detected in the HVAC system. Please investigate and identify the root cause.",
    "Building operators report comfort complaints and suspect an HVAC equipment issue.",
    "The building management system has flagged a potential fault. Please diagnose the issue.",
]


def _get_symptom_description(fault_type: str, system_name: str) -> str:
    """Generate a human-readable symptom description for a fault type."""
    # Try exact match first, then partial match
    templates = SYMPTOM_TEMPLATES.get(fault_type)
    if templates is None:
        for key, tmpl in SYMPTOM_TEMPLATES.items():
            if key in fault_type:
                templates = tmpl
                break
    if templates is None:
        templates = DEFAULT_SYMPTOMS

    templates = _filter_templates_by_mode(list(templates), fault_type)
    desc = random.choice(templates)
    return f"[{system_name}] {desc}"


def generate_single_system_scenarios(
    data_root: str,
    system_id: str,
    system_name: str,
    topology_builder,
    path_generator: Optional[DiagnosticPathGenerator] = None,
    n_windows_per_fault: int = 5,
    window_size: int = 15,
    random_state: int = 42,
) -> List[FaultScenario]:
    """
    Generate single-system fault scenarios from a system's data files.

    For each fault file, samples multiple time windows to create scenarios.
    Each scenario includes a guided diagnostic path.
    """
    rng = random.Random(random_state)
    files = discover_fault_files(data_root, system_id)
    scenarios = []
    skipped_short_files = 0
    components = topology_builder.get_system_components(system_id)

    for finfo in files:
        try:
            row_count = get_fault_file_row_count(finfo)
        except Exception as exc:
            logger.warning(
                "Skipping %s/%s: failed to read row count: %s",
                system_id,
                finfo.filename,
                exc,
            )
            skipped_short_files += 1
            continue
        if row_count < window_size:
            logger.warning(
                "Skipping %s/%s: row_count=%s < window_size=%s",
                system_id,
                finfo.filename,
                row_count,
                window_size,
            )
            skipped_short_files += 1
            continue

        if finfo.is_fault_free:
            # Generate "no fault" scenarios
            for i in range(min(n_windows_per_fault, 3)):
                start = _sample_valid_window_start(
                    rng,
                    row_count=row_count,
                    window_size=window_size,
                    preferred_min_start=1000,
                )
                if start is None:
                    continue
                no_fault_path = generate_no_fault_path(system_id, topology_builder, rng)
                scenario = FaultScenario(
                    scenario_id=f"{system_id}_nofault_{i}",
                    scenario_type="no_fault",
                    description=_get_symptom_description("normal", system_name),
                    root_cause_system=system_id,
                    root_cause_node="none",
                    fault_type="Normal",
                    fault_intensity="none",
                    affected_systems=[system_id],
                    optimal_path=[
                        "get_system_overview",
                        f"get_node_children:system::{system_id}",
                        f"diagnose_node:{components[0]['node_id']}" if components else "",
                    ],
                    diagnostic_path=no_fault_path,
                    source_file=finfo.filename,
                    time_window_start=start,
                    time_window_end=start + window_size,
                    difficulty="easy",
                )
                scenarios.append(scenario)
        else:
            # Generate fault scenarios with diagnostic paths
            root_node = _infer_root_cause_node(finfo.fault_type, system_id, components)

            for i in range(n_windows_per_fault):
                start = _sample_valid_window_start(
                    rng,
                    row_count=row_count,
                    window_size=window_size,
                    preferred_min_start=1000,
                )
                if start is None:
                    continue

                # Generate guided diagnostic path
                diag_path = None
                if path_generator:
                    diag_path = path_generator.generate_path(
                        fault_type=finfo.label,
                        root_cause_node=root_node,
                        root_cause_system=system_id,
                        topology_builder=topology_builder,
                        downstream_system=None,  # single-system
                        rng=rng,
                    )

                scenario = FaultScenario(
                    scenario_id=f"{system_id}_{finfo.fault_type}_{finfo.fault_intensity}_{i}",
                    scenario_type="single_system",
                    description=_get_symptom_description(finfo.fault_type, system_name),
                    root_cause_system=system_id,
                    root_cause_node=root_node,
                    fault_type=finfo.label,
                    fault_intensity=finfo.fault_intensity,
                    affected_systems=[system_id],
                    optimal_path=[
                        "get_system_overview",
                        f"get_node_children:system::{system_id}",
                        f"diagnose_node:{root_node}",
                    ],
                    diagnostic_path=diag_path,
                    source_file=finfo.filename,
                    time_window_start=start,
                    time_window_end=start + window_size,
                    difficulty=_assess_difficulty(finfo.fault_intensity),
                )
                scenarios.append(scenario)

    logger.info(f"Generated {len(scenarios)} single-system scenarios for '{system_id}'")
    if skipped_short_files:
        logger.warning(
            "Skipped %s files for '%s' because no valid time window was available",
            skipped_short_files,
            system_id,
        )
    return scenarios


def generate_cross_system_scenarios(
    single_scenarios: List[FaultScenario],
    topology_builder,
    path_generator: Optional[DiagnosticPathGenerator] = None,
    n_scenarios: int = 600,
    random_state: int = 42,
) -> List[FaultScenario]:
    """
    Generate cross-system fault propagation scenarios.

    Takes upstream system faults and creates scenarios where the agent must
    trace from downstream symptoms to upstream root cause using the
    guided diagnostic path.
    """
    rng = random.Random(random_state)
    upstream_systems = {"chiller_plant", "boiler_plant"}

    upstream_faults = [
        s for s in single_scenarios
        if s.root_cause_system in upstream_systems
        and s.scenario_type == "single_system"
    ]

    if not upstream_faults:
        logger.warning("No upstream fault scenarios found for cross-system generation")
        return []

    cross_scenarios = []
    attempts = 0
    max_attempts = max(n_scenarios * 5, 100)
    while len(cross_scenarios) < n_scenarios and attempts < max_attempts:
        attempts += 1
        i = len(cross_scenarios)
        base = rng.choice(upstream_faults)

        # Find downstream systems
        base_components = topology_builder.get_system_components(base.root_cause_system)
        downstream_systems = set()
        for comp in base_components:
            ds = topology_builder.get_downstream_nodes(comp["node_id"])
            for d in ds:
                if d.get("system_id") and d["system_id"] != base.root_cause_system:
                    downstream_systems.add(d["system_id"])

        if not downstream_systems:
            continue

        affected = [base.root_cause_system] + sorted(downstream_systems)
        ds_system = rng.choice(list(downstream_systems))

        # Generate cross-system diagnostic path
        diag_path = None
        if path_generator:
            diag_path = path_generator.generate_path(
                fault_type=base.fault_type,
                root_cause_node=base.root_cause_node,
                root_cause_system=base.root_cause_system,
                topology_builder=topology_builder,
                downstream_system=ds_system,
                rng=rng,
            )

        system_config_map = {
            "chiller_plant": "Chiller Plant",
            "boiler_plant": "Boiler Plant",
            "sdahu": "Single-Duct AHU",
            "ddahu": "Dual-Duct AHU",
            "fcu": "Fan Coil Unit",
            "pfpu": "Parallel Fan Powered Unit",
            "sfpu": "Series Fan Powered Unit",
            "rtu": "Rooftop Unit",
        }

        # Use path symptom description if available
        if diag_path and diag_path.symptom_description:
            desc = f"[{system_config_map.get(ds_system, ds_system)}] {diag_path.symptom_description}"
        else:
            desc = (
                f"[{system_config_map.get(ds_system, ds_system)}] "
                f"Comfort complaints have been received from zones served by this system. "
                f"The issue may originate from an upstream system. Please investigate."
            )

        ds_components = topology_builder.get_system_components(ds_system)
        optimal_path = [
            "get_system_overview",
            f"get_node_children:system::{ds_system}",
            f"diagnose_node:{ds_components[0]['node_id']}" if ds_components else "",
            f"get_upstream_nodes:{ds_components[0]['node_id']}" if ds_components else "",
            f"diagnose_node:{base.root_cause_node}",
        ]

        cross_scenarios.append(FaultScenario(
            scenario_id=f"cross_{base.root_cause_system}_{ds_system}_{i}",
            scenario_type="cross_system",
            description=desc,
            root_cause_system=base.root_cause_system,
            root_cause_node=base.root_cause_node,
            fault_type=base.fault_type,
            fault_intensity=base.fault_intensity,
            affected_systems=affected,
            optimal_path=optimal_path,
            diagnostic_path=diag_path,
            source_file=base.source_file,
            time_window_start=base.time_window_start,
            time_window_end=base.time_window_end,
            difficulty="hard",
        ))

    logger.info(f"Generated {len(cross_scenarios)} cross-system scenarios")
    return cross_scenarios


def _infer_root_cause_node(
    fault_type: str,
    system_id: str,
    components: List[Dict],
) -> str:
    """Infer the root cause node based on fault type keywords."""
    fault_lower = fault_type.lower()

    # Keyword-to-component mapping (order matters: more specific first)
    keyword_map = {
        "sensorbias_rmtemp": ["Room", "Test_Room"],
        "sensorbias_vavairflow": [
            "Parallel_FPU", "Series_FPU", "Mixing_Box_VAV", "fcu_zone",
        ],
        "sensorbias_hsa": ["Hot_Deck"],
        "sensorbias_hsp": ["Hot_Deck"],
        "sensorbias_csa": ["Cold_Deck"],
        "sensorbias_csp": ["Cold_Deck"],
        "vlvstuck_cooling": ["Cooling_Coil", "fcu_zone"],
        "vlvleak_cooling": ["Cooling_Coil", "fcu_zone"],
        "vlvstuck_heating": ["Heating_Coil", "Reheating_Coil", "fcu_zone"],
        "vlvleak_heating": ["Heating_Coil", "Reheating_Coil", "fcu_zone"],
        "fouling_cooling": ["Cooling_Coil", "fcu_zone"],
        "fouling_heating": ["Heating_Coil", "Reheating_Coil", "fcu_zone"],
        "filterrestriction": ["fcu_zone", "Filter"],
        "oablockage": ["Outdoor_Air_Damper", "fcu_zone"],
        "oadmpr": ["Outdoor_Air_Damper", "fcu_zone"],
        "fanoutletblockage": ["Supply_Air_Fan", "fcu_zone"],
        "vavdmpr": ["Discharge_Air_Damper", "Mixing_Box_VAV"],
        "dmprstuck_oa": ["Outdoor_Air_Damper"],
        "dmprstuck_cold": ["Cold_Deck"],
        "dmprstuck_hot": ["Hot_Deck"],
        "dmpr": ["Damper", "Outdoor_Air_Damper"],
        "reheatcoil": ["Reheating_Coil", "Heating_Coil"],
        "reheatvlv":  ["Reheating_Coil", "Heating_Coil"],
        "chiller": ["Chiller", "CHL"],
        "coolingtower": ["Cooling_Tower", "CT"],
        "cooling_tower": ["Cooling_Tower", "CT"],
        "boiler": ["Boiler"],
        "bypass": ["Three_Way_Valve", "TWV", "bypass"],
        "pump": ["Pump", "PM"],
        "coil": ["Coil", "Cooling_Coil", "Heating_Coil"],
        "coi": ["Cooling_Coil", "Coil"],
        "damper": ["Damper", "Outdoor_Air_Damper"],
        "fan": ["Fan", "Supply_Fan", "Return_Fan"],
        "valve": ["Valve", "VLV"],
        "sa_bias": ["AHU", "Supply_Air"],
        "oa_bias": ["Outdoor_Air_Damper", "AHU"],
        "secondary": ["Chilled_Water_System", "CWL_SEC"],
        "hot_water_temp": ["Simulated_Boiler_Plant", "HWL"],
        "hot_water_pressure": ["Pump", "HWL"],
        "fouling": ["Cooling_Tower", "Boiler", "Coil"],
    }

    for keyword, comp_names in keyword_map.items():
        if keyword in fault_lower:
            # Priority: exact name match first, then substring match
            for cn in comp_names:
                for comp in components:
                    comp_name = comp.get("name", "")
                    if comp_name == cn:
                        return comp["node_id"]
            for cn in comp_names:
                for comp in components:
                    comp_name = comp.get("name", "")
                    if cn in comp_name:
                        return comp["node_id"]

    # Default: return first component with sensors
    for comp in components:
        if comp.get("sensor_names"):
            return comp["node_id"]

    return components[0]["node_id"] if components else f"{system_id}::unknown"


def _assess_difficulty(intensity: str) -> str:
    """Assess scenario difficulty based on fault intensity."""
    try:
        val = abs(float(intensity.replace("%", "")))
        if val >= 50:
            return "easy"
        elif val >= 20:
            return "medium"
        else:
            return "hard"
    except (ValueError, AttributeError):
        if intensity in ("Severe", "100", "075"):
            return "easy"
        elif intensity in ("Moderate", "050"):
            return "medium"
        return "medium"


def save_scenarios(scenarios: List[FaultScenario], output_path: str) -> None:
    """Serialize scenarios to JSON."""
    data = []
    for s in scenarios:
        entry = {
            "scenario_id": s.scenario_id,
            "scenario_type": s.scenario_type,
            "description": s.description,
            "root_cause_system": s.root_cause_system,
            "root_cause_node": s.root_cause_node,
            "fault_type": s.fault_type,
            "fault_intensity": s.fault_intensity,
            "affected_systems": s.affected_systems,
            "optimal_path": s.optimal_path,
            "source_file": s.source_file,
            "time_window_start": s.time_window_start,
            "time_window_end": s.time_window_end,
            "difficulty": s.difficulty,
        }
        # Serialize diagnostic path if present
        if s.diagnostic_path:
            entry["diagnostic_path"] = {
                "nodes": [
                    {
                        "node_id": n.node_id,
                        "role": n.role,
                        "system_id": n.system_id,
                        "component_name": n.component_name,
                        "abnormal_hint": n.abnormal_hint,
                        "direction": n.direction,
                    }
                    for n in s.diagnostic_path.nodes
                ],
                "root_cause_node": s.diagnostic_path.root_cause_node,
                "fault_type": s.diagnostic_path.fault_type,
                "path_length": s.diagnostic_path.path_length,
            }
        data.append(entry)
    save_json(data, output_path)
    logger.info(f"Saved {len(data)} scenarios to {output_path}")


def load_scenarios(input_path: str) -> List[FaultScenario]:
    """Deserialize scenarios from JSON."""
    data = load_json(input_path)
    scenarios = []
    for d in data:
        scenarios.append(FaultScenario(**d))
    return scenarios
