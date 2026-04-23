"""
Script 04: Generate RL training data (prompts only, no oracle trajectories).

Usage:
    python scripts/04_generate_rl_data.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.fault_scenario import load_scenarios
from src.data_gen.rl_formatter import format_rl_dataset, split_rl_data
from src.utils.io_utils import setup_logger, ensure_dir

logger = setup_logger("generate_rl_data")


def main():
    parser = argparse.ArgumentParser(description="Generate RL training data")
    parser.add_argument("--scenarios-path", default="outputs/data/all_scenarios.json")
    parser.add_argument("--output-dir", default="outputs/data")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)

    # Load scenarios
    scenarios = load_scenarios(args.scenarios_path)
    logger.info(f"Loaded {len(scenarios)} scenarios")

    # Split into train/val/test
    splits = split_rl_data(scenarios)

    # Format each split
    for split_name, split_scenarios in splits.items():
        output_path = os.path.join(output_dir, f"rl_{split_name}.jsonl")
        format_rl_dataset(split_scenarios, output_path)

    logger.info("RL data generation complete!")


if __name__ == "__main__":
    main()
