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
    parser.add_argument("--models", default="base,sft,rl",
                        help="Comma-separated model names to evaluate")
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--sft-model", default="outputs/sft/best")
    parser.add_argument("--rl-model", default="outputs/rl/final")
    parser.add_argument("--sft-data", default="outputs/data/sft_train.jsonl",
                        help="SFT data path for generating test scenarios")
    parser.add_argument("--test-data", default="outputs/data/eval_test.jsonl")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--test-size", type=int, default=200)
    parser.add_argument("--test-ratio", type=float, default=0.1,
                        help="Fraction of SFT data to use as test set")
    parser.add_argument("--max-steps", type=int, default=15,
                        help="Maximum tool calls per episode")
    parser.add_argument("--prepare-test-data", action="store_true",
                        help="Only prepare test data, don't run evaluation")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)

    # Prepare test data if needed
    if args.prepare_test_data or not os.path.exists(args.test_data):
        logger.info("Preparing test scenarios from SFT data...")
        prepare_test_scenarios(
            args.sft_data, args.test_data,
            test_ratio=args.test_ratio,
        )
        if args.prepare_test_data:
            logger.info("Test data prepared. Exiting.")
            return

    model_paths = {
        "base": args.base_model,
        "sft": args.sft_model,
        "rl": args.rl_model,
    }

    models_to_eval = args.models.split(",")
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
