"""
Report Generator — Generate evaluation reports in Markdown format.

Produces comparative reports with tables showing metrics across models,
per-scenario-type breakdowns, and improvement analysis.
"""

import json
import os
from typing import Any, Dict, List

from src.utils.io_utils import save_json, setup_logger

logger = setup_logger(__name__)


def generate_markdown_report(
    comparison: Dict[str, Any],
    eval_results: List[Dict[str, Any]],
    output_path: str,
) -> str:
    """
    Generate a comprehensive Markdown evaluation report.

    Args:
        comparison: Output from compare_models().
        eval_results: List of per-model evaluation results.
        output_path: Path to save the Markdown report.

    Returns:
        Path to the saved report.
    """
    lines = []
    lines.append("# HVAC Fault Diagnosis Agent — Evaluation Report\n")
    lines.append(f"**Generated:** {eval_results[0].get('timestamp', 'N/A')}\n")
    lines.append(f"**Models Evaluated:** {len(eval_results)}\n")

    # Overall Metrics Table
    lines.append("## Overall Metrics Comparison\n")
    metric_names = [
        ("paper_DA", "Diagnostic Accuracy (DA)"),
        ("paper_TR", "Topology Rationality (TR)"),
        ("paper_ECR", "Evidence Closure Rate (ECR)"),
        ("paper_SE", "Search Efficiency (SE)"),
        ("paper_mean", "**Paper Mean Score**"),
    ]

    # Header
    model_names = [r["model_name"] for r in eval_results]
    header = "| Metric | " + " | ".join(model_names) + " |"
    separator = "|" + "|".join(["---"] * (len(model_names) + 1)) + "|"
    lines.append(header)
    lines.append(separator)

    for key, display_name in metric_names:
        row = f"| {display_name} |"
        for result in eval_results:
            val = result["overall_metrics"].get(key, 0)
            row += f" {val:.4f} |"
        lines.append(row)

    lines.append("")

    # Improvements Section
    if comparison.get("improvements"):
        lines.append("## Performance Improvements\n")
        for transition, improvements in comparison["improvements"].items():
            lines.append(f"### {transition.replace('_to_', ' → ').replace('_', ' ').title()}\n")

            lines.append("| Metric | Previous | Current | Change | % Change |")
            lines.append("|---|---|---|---|---|")

            for key, display_name in metric_names:
                if key in improvements:
                    imp = improvements[key]
                    pct = imp["pct_change"]
                    arrow = "↑" if pct > 0 else "↓" if pct < 0 else "→"
                    lines.append(
                        f"| {display_name} | {imp['previous']:.4f} | "
                        f"{imp['current']:.4f} | {imp['absolute_change']:+.4f} | "
                        f"{arrow} {abs(pct):.1f}% |"
                    )

            lines.append("")

    # Per-Scenario-Type Breakdown
    lines.append("## Metrics by Scenario Type\n")
    for result in eval_results:
        lines.append(f"### {result['model_name']}\n")
        type_metrics = result.get("metrics_by_scenario_type", {})

        if type_metrics:
            types = sorted(type_metrics.keys())
            header = "| Metric | " + " | ".join(types) + " |"
            sep = "|" + "|".join(["---"] * (len(types) + 1)) + "|"
            lines.append(header)
            lines.append(sep)

            for key, display_name in metric_names:
                row = f"| {display_name} |"
                for stype in types:
                    val = type_metrics[stype].get(key, 0)
                    row += f" {val:.4f} |"
                lines.append(row)

            lines.append("")

    # Per-Difficulty Breakdown
    lines.append("## Metrics by Difficulty\n")
    for result in eval_results:
        lines.append(f"### {result['model_name']}\n")
        diff_metrics = result.get("metrics_by_difficulty", {})

        if diff_metrics:
            diffs = sorted(diff_metrics.keys())
            header = "| Metric | " + " | ".join(diffs) + " |"
            sep = "|" + "|".join(["---"] * (len(diffs) + 1)) + "|"
            lines.append(header)
            lines.append(sep)

            for key, display_name in metric_names:
                row = f"| {display_name} |"
                for diff in diffs:
                    val = diff_metrics[diff].get(key, 0)
                    row += f" {val:.4f} |"
                lines.append(row)

            lines.append("")

    # Write report
    report_text = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    logger.info(f"Evaluation report saved to {output_path}")
    return output_path
