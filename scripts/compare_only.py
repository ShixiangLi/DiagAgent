"""
Script to compare existing evaluation results and generate a markdown report.

Usage:
    python scripts/compare_only.py --models vanilla,sft
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.evaluator import compare_models
from src.evaluation.report_generator import generate_markdown_report
from src.utils.io_utils import setup_logger, ensure_dir

logger = setup_logger("compare_only")


def main():
    parser = argparse.ArgumentParser(description="Compare existing evaluation results")
    parser.add_argument(
        "--models",
        default="vanilla,sft",
        help="Comma-separated model names (looks for outputs/evaluation/eval_<model>.json)",
    )
    parser.add_argument("--output-dir", default="outputs/evaluation")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    eval_results = []
    for model in models:
        json_path = os.path.join(output_dir, f"eval_{model}.json")
        if not os.path.exists(json_path):
            logger.error(f"Evaluation result not found: {json_path}")
            sys.exit(1)

        logger.info(f"Loading {model} results from {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            eval_results.append(json.load(f))

    if len(eval_results) > 1:
        comparison_path = os.path.join(output_dir, "model_comparison.json")
        logger.info(f"Generating comparison: {comparison_path}")
        comparison = compare_models(eval_results, comparison_path)

        report_path = os.path.join(output_dir, "evaluation_report.md")
        logger.info(f"Generating markdown report: {report_path}")
        generate_markdown_report(comparison, eval_results, report_path)
        logger.info("Comparison complete!")
    else:
        logger.warning("At least two models are required for comparison.")


if __name__ == "__main__":
    main()
