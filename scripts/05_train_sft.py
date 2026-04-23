"""
Script 05: Run SFT training.

Usage:
    python scripts/05_train_sft.py [--config configs/sft_config.yaml] [--max-steps 500]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.sft_trainer import load_sft_config, run_sft_training
from src.utils.io_utils import setup_logger

logger = setup_logger("train_sft")


def main():
    parser = argparse.ArgumentParser(description="Run SFT training")
    parser.add_argument("--config", default="configs/sft_config.yaml")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    args = parser.parse_args()

    config = load_sft_config(args.config)

    # Override with CLI args
    if args.data_path:
        config["data_path"] = args.data_path
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.max_steps:
        config["num_epochs"] = 1  # Will be limited by max_steps
    if args.eval_steps:
        config["eval_steps"] = args.eval_steps

    logger.info("SFT Training Configuration:")
    for k, v in config.items():
        if k != "target_modules":
            logger.info(f"  {k}: {v}")

    best_checkpoint = run_sft_training(config)
    logger.info(f"\nBest checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
