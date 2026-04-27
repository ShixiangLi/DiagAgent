"""
Script 04: Generate RL training data (prompts only, no oracle trajectories).

Usage:
    python scripts/04_generate_rl_data.py [--scenarios-path ...] [--output-dir ...]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.fault_scenario import FaultScenario
from src.environment.diagnostic_path import DiagnosticPath, PathNode
from src.data_gen.rl_formatter import format_rl_dataset, split_rl_data
from src.utils.io_utils import setup_logger, ensure_dir

logger = setup_logger("generate_rl_data")


def _load_scenarios_with_paths(input_path: str):
    """
    Load scenarios from JSON with proper DiagnosticPath deserialization.

    The raw JSON has diagnostic_path as a nested dict that needs to be
    reconstructed into a DiagnosticPath object with PathNode children.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    scenarios = []
    for d in raw_data:
        dp_data = d.pop("diagnostic_path", None)
        fs = FaultScenario(**d)

        # Reconstruct DiagnosticPath from serialized dict
        if dp_data and isinstance(dp_data, dict):
            nodes = [PathNode(**n) for n in dp_data.get("nodes", [])]
            fs.diagnostic_path = DiagnosticPath(
                nodes=nodes,
                root_cause_node=dp_data.get("root_cause_node", ""),
                fault_type=dp_data.get("fault_type", ""),
                symptom_description=dp_data.get("symptom_description", ""),
            )

        scenarios.append(fs)

    return scenarios


def main():
    parser = argparse.ArgumentParser(description="Generate RL training data")
    parser.add_argument("--scenarios-path", default="outputs/data/all_scenarios.json")
    parser.add_argument("--output-dir", default="outputs/data")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)

    # Load scenarios with proper DiagnosticPath deserialization
    scenarios = _load_scenarios_with_paths(args.scenarios_path)
    logger.info(f"Loaded {len(scenarios)} scenarios")

    # Log diagnostic path availability
    n_with_path = sum(1 for s in scenarios if s.diagnostic_path is not None)
    logger.info(f"  {n_with_path}/{len(scenarios)} have DiagnosticPath")

    # Log scenario type distribution
    from collections import Counter
    type_counts = Counter(s.scenario_type for s in scenarios)
    for stype, cnt in sorted(type_counts.items()):
        logger.info(f"  {stype}: {cnt}")

    # Split into train/val/test (stratified by type)
    splits = split_rl_data(scenarios)

    # Format each split
    for split_name, split_scenarios in splits.items():
        output_path = os.path.join(output_dir, f"rl_{split_name}.jsonl")
        format_rl_dataset(split_scenarios, output_path)

    # Verify output format
    logger.info("\n=== Verification ===")
    train_path = os.path.join(output_dir, "rl_train.jsonl")
    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            first = json.loads(f.readline())
        logger.info(f"  Sample keys: {list(first.keys())}")
        gt = first.get("ground_truth", {})
        logger.info(f"  GT keys: {list(gt.keys())}")
        logger.info(f"  optimal_path_length: {gt.get('optimal_path_length')}")
        meta = first.get("metadata", {})
        logger.info(f"  scenario_id in metadata: {'scenario_id' in meta}")

    logger.info("RL data generation complete!")


if __name__ == "__main__":
    main()
