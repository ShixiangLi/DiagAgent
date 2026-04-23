"""
Scenario Sampler — Stratified sampling of fault scenarios for training data.

Ensures diversity across:
  - Scenario types (single-system, cross-system, no-fault, ambiguous)
  - Systems (all 8 HVAC system types)
  - Fault types and intensities
  - Difficulty levels
"""

import random
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

from src.environment.fault_scenario import FaultScenario
from src.utils.io_utils import setup_logger

logger = setup_logger(__name__)

# Default distribution targets
DEFAULT_DISTRIBUTION = {
    "single_system": 0.40,
    "cross_system": 0.25,
    "no_fault": 0.10,
    "ambiguous": 0.10,
    "low_confidence": 0.15,
}


def sample_scenarios(
    all_scenarios: List[FaultScenario],
    n_total: int = 15000,
    distribution: Optional[Dict[str, float]] = None,
    random_state: int = 42,
) -> List[FaultScenario]:
    """
    Sample a stratified set of scenarios for training data generation.

    Args:
        all_scenarios: Pool of all available scenarios.
        n_total: Total number of scenarios to sample.
        distribution: Dict of scenario_type → target fraction.
        random_state: Random seed for reproducibility.

    Returns:
        Stratified list of sampled scenarios.
    """
    rng = random.Random(random_state)
    dist = distribution or DEFAULT_DISTRIBUTION

    # Group scenarios by type
    by_type: Dict[str, List[FaultScenario]] = defaultdict(list)
    for s in all_scenarios:
        by_type[s.scenario_type].append(s)

    # Calculate target counts per type
    targets = {}
    for stype, frac in dist.items():
        targets[stype] = int(n_total * frac)

    # Adjust for available scenarios
    sampled = []
    for stype, target_n in targets.items():
        pool = by_type.get(stype, [])
        if not pool:
            logger.warning(f"No scenarios of type '{stype}' available")
            continue

        if len(pool) >= target_n:
            selected = rng.sample(pool, target_n)
        else:
            # Oversample with repetition (varying time windows)
            selected = pool.copy()
            while len(selected) < target_n:
                base = rng.choice(pool)
                # Create a variant with a different time window
                variant = FaultScenario(
                    scenario_id=f"{base.scenario_id}_var{len(selected)}",
                    scenario_type=base.scenario_type,
                    description=base.description,
                    root_cause_system=base.root_cause_system,
                    root_cause_node=base.root_cause_node,
                    fault_type=base.fault_type,
                    fault_intensity=base.fault_intensity,
                    affected_systems=base.affected_systems,
                    optimal_path=base.optimal_path,
                    source_file=base.source_file,
                    time_window_start=base.time_window_start + rng.randint(-5000, 5000),
                    time_window_end=base.time_window_end + rng.randint(-5000, 5000),
                    difficulty=base.difficulty,
                    diagnostic_path=base.diagnostic_path,
                )
                selected.append(variant)

        sampled.extend(selected)

    # Shuffle final set
    rng.shuffle(sampled)

    # Log distribution
    type_counts = Counter(s.scenario_type for s in sampled)
    system_counts = Counter(s.root_cause_system for s in sampled)
    diff_counts = Counter(s.difficulty for s in sampled)

    logger.info(f"Sampled {len(sampled)} scenarios:")
    logger.info(f"  By type: {dict(type_counts)}")
    logger.info(f"  By system: {dict(system_counts)}")
    logger.info(f"  By difficulty: {dict(diff_counts)}")

    return sampled


def create_ambiguous_scenarios(
    single_scenarios: List[FaultScenario],
    n_scenarios: int = 1500,
    random_state: int = 42,
) -> List[FaultScenario]:
    """
    Create ambiguous scenarios where the initial investigation leads to the
    wrong node, requiring backtracking. These teach the agent to handle
    uncertainty and explore alternative hypotheses.
    """
    rng = random.Random(random_state)
    fault_scenarios = [s for s in single_scenarios if s.scenario_type == "single_system"]

    ambiguous = []
    for i in range(min(n_scenarios, len(fault_scenarios))):
        base = rng.choice(fault_scenarios)

        # Modify description to be more ambiguous
        ambiguous_desc = (
            f"The building automation system has detected anomalous behavior "
            f"that could involve multiple systems. Sensor readings are showing "
            f"deviations from normal operation. Please conduct a thorough "
            f"investigation to identify the root cause."
        )

        # Extend the optimal path with initial wrong guesses
        wrong_systems = [s for s in ["chiller_plant", "boiler_plant", "sdahu", "rtu"]
                         if s != base.root_cause_system]
        if wrong_systems:
            wrong_sys = rng.choice(wrong_systems)
            extended_path = [
                "get_system_overview",
                f"get_node_children:system::{wrong_sys}",
                f"diagnose_node:{wrong_sys}::first_component",
                # After finding normal, pivot to correct system
                f"get_node_children:system::{base.root_cause_system}",
                f"diagnose_node:{base.root_cause_node}",
            ]
        else:
            extended_path = base.optimal_path

        ambiguous.append(FaultScenario(
            scenario_id=f"ambiguous_{i}",
            scenario_type="ambiguous",
            description=ambiguous_desc,
            root_cause_system=base.root_cause_system,
            root_cause_node=base.root_cause_node,
            fault_type=base.fault_type,
            fault_intensity=base.fault_intensity,
            affected_systems=base.affected_systems,
            optimal_path=extended_path,
            source_file=base.source_file,
            time_window_start=base.time_window_start,
            time_window_end=base.time_window_end,
            difficulty="hard",
            diagnostic_path=base.diagnostic_path,
        ))

    logger.info(f"Created {len(ambiguous)} ambiguous scenarios")
    return ambiguous


def create_low_confidence_scenarios(
    single_scenarios: List[FaultScenario],
    n_scenarios: int = 2000,
    random_state: int = 42,
) -> List[FaultScenario]:
    """
    Create scenarios where node model predictions have low confidence,
    requiring the agent to seek additional evidence or verify predictions.

    These use subtle faults (low intensity) that produce borderline predictions.
    """
    rng = random.Random(random_state)

    # Prefer hard/medium difficulty scenarios (low intensity faults)
    hard_scenarios = [
        s for s in single_scenarios
        if s.difficulty in ("hard", "medium") and s.scenario_type == "single_system"
    ]

    if not hard_scenarios:
        hard_scenarios = [s for s in single_scenarios if s.scenario_type == "single_system"]

    low_conf = []
    for i in range(min(n_scenarios, len(hard_scenarios) * 3)):
        base = rng.choice(hard_scenarios)

        desc = (
            f"A subtle anomaly has been detected in the HVAC system. "
            f"Initial automated checks are inconclusive. The fault signal "
            f"is weak and requires careful investigation. Some diagnostic "
            f"model predictions may have low confidence — please verify "
            f"any uncertain results by checking related components."
        )

        low_conf.append(FaultScenario(
            scenario_id=f"low_conf_{i}",
            scenario_type="low_confidence",
            description=desc,
            root_cause_system=base.root_cause_system,
            root_cause_node=base.root_cause_node,
            fault_type=base.fault_type,
            fault_intensity=base.fault_intensity,
            affected_systems=base.affected_systems,
            optimal_path=base.optimal_path + [
                f"get_node_sensors:{base.root_cause_node}",
                f"diagnose_node:{base.root_cause_node}",
            ],
            source_file=base.source_file,
            time_window_start=base.time_window_start,
            time_window_end=base.time_window_end,
            difficulty="hard",
            diagnostic_path=base.diagnostic_path,
        ))

    logger.info(f"Created {len(low_conf)} low-confidence scenarios")
    return low_conf
