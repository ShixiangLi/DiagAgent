"""
Script 03: Generate SFT training data.

Usage:
    python scripts/03_generate_sft_data.py [--n-total 15000] [--validate]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.topology.topology_builder import TopologyBuilder
from src.topology.topology_tools import TopologyToolExecutor
from src.node_models.model_registry import ModelRegistry
from src.node_models.prediction_tools import PredictionToolExecutor
from src.environment.tool_executor import UnifiedToolExecutor
from src.environment.fault_scenario import (
    generate_single_system_scenarios,
    generate_cross_system_scenarios,
    save_scenarios,
    FaultScenarioState,
)
from src.data_gen.scenario_sampler import (
    sample_scenarios,
    create_ambiguous_scenarios,
    create_low_confidence_scenarios,
    create_no_fault_scenarios,
)
from src.data_gen.trajectory_generator import generate_batch_trajectories
from src.data_gen.sft_formatter import format_sft_dataset
from src.environment.diagnostic_path import DiagnosticPathGenerator
from src.utils.io_utils import setup_logger, ensure_dir, load_yaml

logger = setup_logger("generate_sft_data")

SYSTEM_NAMES = {
    "chiller_plant": "Chiller Plant",
    "boiler_plant": "Boiler Plant",
    "sdahu": "Single-Duct AHU",
    "ddahu": "Dual-Duct AHU",
    "rtu": "Rooftop Unit",
    "fcu": "Fan Coil Unit",
    "pfpu": "Parallel Fan Powered Unit",
    "sfpu": "Series Fan Powered Unit",
}


def main():
    parser = argparse.ArgumentParser(description="Generate SFT training data")
    parser.add_argument("--config", default="configs/topology_config.yaml")
    parser.add_argument("--path-config", default="configs/diagnostic_paths.yaml")
    parser.add_argument("--data-root", default="data/lbnl")
    parser.add_argument("--models-dir", default="outputs/models")
    parser.add_argument("--output-dir", default="outputs/data")
    parser.add_argument("--n-total", type=int, default=3000)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Generate only this many for quick testing")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)

    # Build topology
    logger.info("Building topology...")
    builder = TopologyBuilder(args.config, args.data_root)
    builder.build()

    # Load model registry
    logger.info("Loading model registry...")
    registry = ModelRegistry(args.models_dir)

    # Initialize diagnostic path generator
    logger.info("Loading diagnostic path templates...")
    path_generator = DiagnosticPathGenerator(args.path_config)

    # Create tool executors
    topo_executor = TopologyToolExecutor(builder)
    pred_executor = PredictionToolExecutor(registry, builder)
    tool_executor = UnifiedToolExecutor(topo_executor, pred_executor)

    # Generate scenarios for each system
    logger.info("\n=== Generating Fault Scenarios ===")
    all_scenarios = []

    config = load_yaml(args.config)
    for sys_id in config["systems"]:
        sys_name = SYSTEM_NAMES.get(sys_id, sys_id)
        scenarios = generate_single_system_scenarios(
            args.data_root, sys_id, sys_name, builder,
            path_generator=path_generator,
            n_windows_per_fault=5,
        )
        all_scenarios.extend(scenarios)

    # Generate cross-system scenarios
    cross_scenarios = generate_cross_system_scenarios(
        all_scenarios, builder, path_generator=path_generator,
    )
    all_scenarios.extend(cross_scenarios)

    # Generate ambiguous and low-confidence scenarios
    ambiguous = create_ambiguous_scenarios(all_scenarios)
    all_scenarios.extend(ambiguous)

    low_conf = create_low_confidence_scenarios(all_scenarios)
    all_scenarios.extend(low_conf)

    # Generate no_fault scenarios (Non-Ambiguous only)
    no_fault = create_no_fault_scenarios(all_scenarios)
    all_scenarios.extend(no_fault)

    logger.info(f"Total scenarios generated: {len(all_scenarios)}")

    # Save all scenarios
    save_scenarios(all_scenarios, os.path.join(output_dir, "all_scenarios.json"))

    # Sample for training
    n_target = args.sample_size if args.sample_size else args.n_total
    sampled = sample_scenarios(all_scenarios, n_total=n_target)

    # Load real sensor data per source_file (only load CSVs actually needed)
    logger.info("\n=== Loading Real Sensor Data ===")
    import pandas as pd
    from src.node_models.data_loader import discover_fault_files
    from src.environment.fault_scenario import create_scenario_state

    # For no_fault scenarios, override source_file to use fault-free CSV
    # This ensures Oracle genuinely predicts Normal from baseline data
    normal_file_map = {}  # system_id → fault-free filename
    for sys_id in config["systems"]:
        fault_files = discover_fault_files(args.data_root, sys_id)
        for ff in fault_files:
            if ff.fault_type.lower() == "normal":
                normal_file_map[sys_id] = ff.filename
                break

    for s in sampled:
        if 'no_fault' in s.scenario_type:
            normal_fname = normal_file_map.get(s.root_cause_system, "")
            if normal_fname:
                # Override source_file to use fault-free data
                object.__setattr__(s, 'source_file', normal_fname)
                logger.debug(f"  no_fault {s.scenario_id}: using {normal_fname}")

    # Group sampled scenarios by (system_id, source_file) to avoid duplicate loads
    needed_files = {}  # (sys_id, filename) → list of scenario_ids
    for s in sampled:
        key = (s.root_cause_system, s.source_file)
        needed_files.setdefault(key, []).append(s.scenario_id)

    # Also gather cross-system downstream files
    for s in sampled:
        for sys_id in s.affected_systems:
            if sys_id != s.root_cause_system:
                normal_fname = normal_file_map.get(sys_id, "")
                key = (sys_id, normal_fname)  # downstream systems use clean baseline
                needed_files.setdefault(key, [])

    # Build file path lookup by system
    file_path_map = {}  # (sys_id, filename) → full_path
    for sys_id in config["systems"]:
        fault_files = discover_fault_files(args.data_root, sys_id)
        for ff in fault_files:
            file_path_map[(sys_id, ff.filename)] = ff.filepath
            # Also index as the first file per system (for downstream systems)
            if (sys_id, "") not in file_path_map:
                file_path_map[(sys_id, "")] = ff.filepath

    # Load only the CSVs we need
    loaded_data = {}  # (sys_id, filename) → DataFrame
    for (sys_id, fname), _ in needed_files.items():
        fpath = file_path_map.get((sys_id, fname))
        if fpath and os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath, nrows=50000, low_memory=False)
                loaded_data[(sys_id, fname)] = df
                logger.info(f"  Loaded {fname or 'default'} for {sys_id}: {len(df)} rows")
            except Exception as e:
                logger.warning(f"  Failed to load {fname} for {sys_id}: {e}")

    # Create scenario states from loaded data
    logger.info("\n=== Creating Scenario States ===")
    scenario_states = {}
    for s in sampled:
        try:
            # Build system_data dict for this scenario
            system_data = {}
            # Primary system data from the scenario's source file
            key = (s.root_cause_system, s.source_file)
            if key in loaded_data:
                system_data[s.root_cause_system] = loaded_data[key]
            # Downstream system data (use default/first file)
            for sys_id in s.affected_systems:
                if sys_id != s.root_cause_system and sys_id not in system_data:
                    alt_key = (sys_id, normal_file_map.get(sys_id, ""))
                    if alt_key in loaded_data:
                        system_data[sys_id] = loaded_data[alt_key]

            state = create_scenario_state(s, system_data, builder, registry)
            scenario_states[s.scenario_id] = state
        except Exception as e:
            logger.warning(f"  State creation failed for {s.scenario_id}: {e}")

    logger.info(f"  Created {len(scenario_states)}/{len(sampled)} scenario states")

    # Generate trajectories (using real Oracle predictions)
    logger.info("\n=== Generating Diagnostic Trajectories ===")
    trajectories = generate_batch_trajectories(
        sampled, tool_executor, scenario_states,
    )

    # Format as SFT dataset
    logger.info("\n=== Formatting SFT Dataset ===")
    sft_path = format_sft_dataset(
        trajectories,
        os.path.join(output_dir, "sft_train.jsonl"),
        format_type="sharegpt",
    )

    logger.info(f"\nSFT data generation complete!")
    logger.info(f"  Output: {sft_path}")
    logger.info(f"  Total examples: {len(trajectories)}")

    if args.validate:
        logger.info("\n=== Validation ===")
        import json
        with open(sft_path, "r", encoding="utf-8") as f:
            first_10 = [json.loads(f.readline()) for _ in range(min(10, len(trajectories)))]

        for i, entry in enumerate(first_10):
            convs = entry.get("conversations", [])
            n_turns = len(convs)
            n_tool_calls = sum(1 for c in convs if c.get("from") == "gpt" and "<tool_call>" in c.get("value", ""))
            logger.info(f"  Example {i}: {n_turns} turns, {n_tool_calls} tool calls")

        logger.info("✓ Validation passed")


if __name__ == "__main__":
    main()
