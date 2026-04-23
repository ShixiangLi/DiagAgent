"""
Script 06: Run RL (GRPO) training.

Usage:
    python scripts/06_train_rl.py [--config configs/rl_config.yaml]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.rl_trainer import load_rl_config, run_rl_training
from src.utils.io_utils import setup_logger

logger = setup_logger("train_rl")


def main():
    parser = argparse.ArgumentParser(description="Run GRPO RL training")
    parser.add_argument("--config", default="configs/rl_config.yaml")
    parser.add_argument("--sft-checkpoint", default=None,
                        help="Path to SFT checkpoint (overrides config)")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    args = parser.parse_args()

    config = load_rl_config(args.config)

    if args.sft_checkpoint:
        config["model_name_or_path"] = args.sft_checkpoint
        config["ref_model_name"] = args.sft_checkpoint
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.max_steps:
        config["num_train_steps"] = args.max_steps
    if args.eval_steps:
        config["eval_steps"] = args.eval_steps

    logger.info("RL Training Configuration:")
    for k, v in config.items():
        if not isinstance(v, dict):
            logger.info(f"  {k}: {v}")

    best_checkpoint = run_rl_training(config)
    logger.info(f"\nBest RL checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
