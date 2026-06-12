"""
Script 07: Run unified evaluation across Base, SFT, and RL models.

Usage:
    python scripts/07_evaluate.py --models base,sft,rl --test-size 200
    python scripts/07_evaluate.py --models sft --prepare-test-data
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.evaluator import (
    run_model_evaluation,
    compare_models,
    prepare_test_scenarios,
)
from src.evaluation.report_generator import generate_markdown_report
from src.utils.io_utils import setup_logger, ensure_dir

logger = setup_logger("evaluate")


def main():
    parser = argparse.ArgumentParser(description="Run unified evaluation")
    parser.add_argument("--models", default="vanilla,sft,rl",
                        help="Comma-separated model names to evaluate (vanilla/base,sft,rl)")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft-model", default="outputs/sft/best")
    parser.add_argument("--rl-model", default="outputs/rl/final")
    parser.add_argument("--sft-data", default="outputs/data/sft_train.jsonl",
                        help="SFT data path for generating test scenarios")
    parser.add_argument("--test-data", default="outputs/data/rl_test.jsonl",
                        help="Evaluation split to use. Defaults to the fixed RL test split.")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Fraction of SFT data to use as test set")
    parser.add_argument("--sampling-strategy", default="stratified",
                        choices=["stratified", "random", "per_type"],
                        help="Sampling strategy when --test-size is smaller than the test set")
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--samples-per-type", type=int, default=None,
                        help="For --sampling-strategy per_type, sample this many cases per scenario type")
    parser.add_argument("--scenario-types", default=None,
                        help="Comma-separated scenario types to include during sampling")
    parser.add_argument("--max-steps", type=int, default=15,
                        help="Maximum tool calls per episode")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Maximum generated tokens per agent turn")
    parser.add_argument("--eval-csv-nrows", type=int, default=50000,
                        help="Maximum rows loaded per source CSV during real-Oracle evaluation")
    parser.add_argument("--episode-timeout-seconds", type=float, default=None,
                        help="Wall-clock timeout per diagnostic episode")
    parser.add_argument("--generate-timeout-seconds", type=float, default=None,
                        help="Wall-clock timeout passed to each model.generate call")
    parser.add_argument("--max-total-outputs", type=int, default=None,
                        help="Maximum assistant generations per episode, including no-action repairs")
    parser.add_argument("--model-device-map", default="single",
                        help="Model placement for evaluation: single, auto, balanced, none, or a device id")
    parser.add_argument("--oracle-mode", default="real",
                        choices=["real", "teacher", "path_aware", "path-aware"],
                        help="Use real Oracle predictions or path-aware teacher routing")
    parser.add_argument("--include-system-health", action="store_true",
                        help="Expose anomaly_score/health_status in get_system_overview")
    parser.add_argument("--prepare-test-data", action="store_true",
                        help="Only prepare test data, don't run evaluation")
    parser.add_argument("--allow-mock", action="store_true",
                        help="Allow mock evaluation if model loading fails")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)

    # Prepare test data only when explicitly requested, or for the legacy
    # generated eval_test path. Paper results should use outputs/data/rl_test.jsonl.
    if args.prepare_test_data or not os.path.exists(args.test_data):
        if (
            not args.prepare_test_data
            and os.path.normpath(args.test_data) != os.path.normpath("outputs/data/eval_test.jsonl")
        ):
            raise FileNotFoundError(
                f"Evaluation data not found: {args.test_data}. "
                "Pass --prepare-test-data for an SFT-derived smoke set, or "
                "provide the fixed test split with --test-data."
            )
        logger.info("Preparing test scenarios from SFT data...")
        prepare_test_scenarios(
            args.sft_data, args.test_data,
            test_ratio=args.test_ratio,
        )
        if args.prepare_test_data:
            logger.info("Test data prepared. Exiting.")
            return

    model_paths = {
        "vanilla": args.base_model,
        "base": args.base_model,
        "sft": args.sft_model,
        "rl": args.rl_model,
    }

    models_to_eval = [m.strip() for m in args.models.split(",") if m.strip()]
    scenario_types = (
        [s.strip() for s in args.scenario_types.split(",") if s.strip()]
        if args.scenario_types else None
    )
    eval_results = []

    for model_name in models_to_eval:
        model_path = model_paths.get(model_name, model_name)
        logger.info(f"\nEvaluating: {model_name} ({model_path})")

        result = run_model_evaluation(
            model_name=model_name,
            model_path=model_path,
            test_scenarios_path=args.test_data,
            output_dir=output_dir,
            max_episodes=args.test_size,
            max_steps=args.max_steps,
            max_new_tokens=args.max_new_tokens,
            eval_csv_nrows=args.eval_csv_nrows,
            episode_timeout_seconds=args.episode_timeout_seconds,
            generate_timeout_seconds=args.generate_timeout_seconds,
            max_total_outputs=args.max_total_outputs,
            allow_mock=args.allow_mock,
            sampling_strategy=args.sampling_strategy,
            sampling_seed=args.sampling_seed,
            samples_per_type=args.samples_per_type,
            scenario_types=scenario_types,
            oracle_mode=args.oracle_mode,
            include_system_health=args.include_system_health,
            model_device_map=args.model_device_map,
        )
        eval_results.append(result)

    # Compare models
    if len(eval_results) > 1:
        comparison = compare_models(
            eval_results,
            os.path.join(output_dir, "model_comparison.json"),
        )

        # Generate report
        generate_markdown_report(
            comparison, eval_results,
            os.path.join(output_dir, "evaluation_report.md"),
        )

    logger.info("\n=== Evaluation Complete ===")
    logger.info(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
