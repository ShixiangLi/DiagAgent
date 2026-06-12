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
    parser.add_argument("--ref-checkpoint", default=None,
                        help="Path to frozen SFT reference checkpoint for KL")
    parser.add_argument("--resume-from-checkpoint", default=None,
                        help="Path to an RL checkpoint directory with training_state.pt")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--save-steps", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--rollout-do-sample", type=int, choices=[0, 1], default=None)
    parser.add_argument("--rollout-top-p", type=float, default=None)
    parser.add_argument("--model-device-map", default=None)
    parser.add_argument("--max-rollout-steps", type=int, default=None)
    parser.add_argument("--phase1-rollout-steps", type=int, default=None)
    parser.add_argument("--phase2-rollout-steps", type=int, default=None)
    parser.add_argument("--phase3-rollout-steps", type=int, default=None)
    parser.add_argument("--rollout-generate-timeout-seconds", type=float, default=None)
    parser.add_argument("--rollout-max-input-tokens", type=int, default=None)
    parser.add_argument("--log-rollout-stages", action="store_true")
    parser.add_argument("--save-train-rollouts", type=int, choices=[0, 1], default=None)
    parser.add_argument("--rollout-warmup-new-tokens", type=int, default=None)
    parser.add_argument("--diag-eval-episodes", type=int, default=None)
    parser.add_argument("--diag-eval-samples-per-type", type=int, default=None)
    parser.add_argument("--diag-eval-max-steps", type=int, default=None)
    parser.add_argument("--diag-eval-max-new-tokens", type=int, default=None)
    parser.add_argument("--diag-eval-csv-nrows", type=int, default=None)
    parser.add_argument("--diag-eval-episode-timeout-seconds", type=float, default=None)
    parser.add_argument("--diag-eval-generate-timeout-seconds", type=float, default=None)
    args = parser.parse_args()

    config = load_rl_config(args.config)

    if args.sft_checkpoint:
        config["model_name_or_path"] = args.sft_checkpoint
    if args.ref_checkpoint:
        config["ref_model_name"] = args.ref_checkpoint
    if args.resume_from_checkpoint:
        config["model_name_or_path"] = args.resume_from_checkpoint
        config["resume_from_checkpoint"] = args.resume_from_checkpoint
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.max_steps is not None:
        config["num_train_steps"] = args.max_steps
    if args.eval_steps is not None:
        config["eval_steps"] = args.eval_steps
    if args.save_steps is not None:
        config["save_steps"] = args.save_steps
    if args.group_size is not None:
        config["group_size"] = args.group_size
    if args.max_new_tokens is not None:
        config["max_new_tokens"] = args.max_new_tokens
    if args.rollout_do_sample is not None:
        config["rollout_do_sample"] = bool(args.rollout_do_sample)
    if args.rollout_top_p is not None:
        config["rollout_top_p"] = args.rollout_top_p
    if args.model_device_map is not None:
        config["model_device_map"] = args.model_device_map
    if args.max_rollout_steps is not None:
        config["max_rollout_steps"] = args.max_rollout_steps
    if any(
        value is not None
        for value in (
            args.phase1_rollout_steps,
            args.phase2_rollout_steps,
            args.phase3_rollout_steps,
        )
    ):
        train_rollout_steps = dict(config.get("train_rollout_steps", {}) or {})
        if args.phase1_rollout_steps is not None:
            train_rollout_steps["phase1"] = args.phase1_rollout_steps
        if args.phase2_rollout_steps is not None:
            train_rollout_steps["phase2"] = args.phase2_rollout_steps
        if args.phase3_rollout_steps is not None:
            train_rollout_steps["phase3"] = args.phase3_rollout_steps
        config["train_rollout_steps"] = train_rollout_steps
    if args.rollout_generate_timeout_seconds is not None:
        config["rollout_generate_timeout_seconds"] = (
            args.rollout_generate_timeout_seconds
        )
    if args.rollout_max_input_tokens is not None:
        config["rollout_max_input_tokens"] = args.rollout_max_input_tokens
    if args.log_rollout_stages:
        config["log_rollout_stages"] = True
    if args.save_train_rollouts is not None:
        config["save_train_rollouts"] = bool(args.save_train_rollouts)
    if args.rollout_warmup_new_tokens is not None:
        config["rollout_warmup_new_tokens"] = args.rollout_warmup_new_tokens
    if args.diag_eval_episodes is not None:
        config["diag_eval_episodes"] = args.diag_eval_episodes
    if args.diag_eval_samples_per_type is not None:
        config["diag_eval_samples_per_type"] = args.diag_eval_samples_per_type
    if args.diag_eval_max_steps is not None:
        config["diag_eval_max_steps"] = args.diag_eval_max_steps
    if args.diag_eval_max_new_tokens is not None:
        config["diag_eval_max_new_tokens"] = args.diag_eval_max_new_tokens
    if args.diag_eval_csv_nrows is not None:
        config["diag_eval_csv_nrows"] = args.diag_eval_csv_nrows
    if args.diag_eval_episode_timeout_seconds is not None:
        config["diag_eval_episode_timeout_seconds"] = (
            args.diag_eval_episode_timeout_seconds
        )
    if args.diag_eval_generate_timeout_seconds is not None:
        config["diag_eval_generate_timeout_seconds"] = (
            args.diag_eval_generate_timeout_seconds
        )

    logger.info("RL Training Configuration:")
    for k, v in config.items():
        if not isinstance(v, dict):
            logger.info(f"  {k}: {v}")

    best_checkpoint = run_rl_training(config)
    logger.info(f"\nBest RL checkpoint: {best_checkpoint}")


if __name__ == "__main__":
    main()
