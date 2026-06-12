"""
Script 02: Train node-level fault prediction models.

Usage:
    python scripts/02_train_node_models.py [--systems chiller_plant,boiler_plant]

By default this trains the diagnostic core: one system-level oracle per
system.  Use --with-responsibility to add binary per-node responsibility
models.  Legacy per-node multi-class classifiers are slow and weak for this
project design, so they are only trained when --legacy-per-node is explicit.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.topology.topology_builder import TopologyBuilder
from src.node_models.data_loader import load_system_data
from src.node_models.model_trainer import (
    train_all_responsibility_models,
    train_all_system_models,
    train_system_oracle_model,
)
from src.utils.io_utils import setup_logger, ensure_dir, save_json

logger = setup_logger("train_node_models")

ALL_SYSTEMS = [
    "chiller_plant", "boiler_plant", "sdahu", "ddahu",
    "rtu", "fcu", "pfpu", "sfpu",
]


def _update_responsibility_summary(output_dir: str, all_results: dict) -> None:
    """Write/merge a manifest that marks responsibility artifacts complete."""
    summary_path = os.path.join(output_dir, "responsibility_training_summary.json")
    existing = {"systems": {}}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {"systems": {}}

    systems = existing.setdefault("systems", {})
    for sys_id, res in all_results.items():
        models = res.get("models", [])
        systems[sys_id] = {
            "n_components": res.get("n_components", 0),
            "n_trained": res.get("n_trained", 0),
            "n_skipped": res.get("n_skipped", 0),
            "trained_nodes": [
                m.get("node_id")
                for m in models
                if m.get("status") == "trained"
                and m.get("model_role") == "per_node_responsibility"
            ],
            "skipped_nodes": [
                {
                    "node_id": m.get("node_id"),
                    "reason": m.get("reason", ""),
                }
                for m in models
                if m.get("status") == "skipped"
                and m.get("model_role") == "per_node_responsibility"
            ],
        }

    existing["complete_systems"] = sorted(
        sys_id for sys_id in ALL_SYSTEMS if sys_id in systems
    )
    existing["complete"] = set(existing["complete_systems"]) == set(ALL_SYSTEMS)
    existing["note"] = (
        "Runtime only enables per-node responsibility fallback when complete=true."
    )
    save_json(existing, summary_path)
    logger.info(
        "Responsibility summary saved to %s (complete=%s, systems=%s/%s)",
        summary_path,
        existing["complete"],
        len(existing["complete_systems"]),
        len(ALL_SYSTEMS),
    )


def _save_training_summary(output_dir: str, all_results: dict) -> str:
    """Write a system-model summary without losing results from other systems.

    Cloud runs often retrain only a subset of weak system oracles.  The
    previous implementation overwrote the full manifest with that subset,
    which made later readiness checks and audits look incomplete even though
    the artifacts were still on disk.
    """
    summary_path = os.path.join(output_dir, "training_summary.json")
    existing = {}
    if os.path.exists(summary_path):
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                existing = loaded
        except Exception:
            existing = {}

    existing.update(all_results)
    save_json(existing, summary_path)
    return summary_path


def main():
    parser = argparse.ArgumentParser(description="Train node fault prediction models")
    parser.add_argument("--config", default="configs/topology_config.yaml")
    parser.add_argument("--data-root", default="data/lbnl")
    parser.add_argument("--output-dir", default="outputs/models")
    parser.add_argument("--systems", default=None,
                        help="Comma-separated system IDs (default: all)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Max rows per CSV file (for faster testing)")
    parser.add_argument("--sample-frac", type=float, default=0.1,
                        help="Fraction of data to sample per file (default 0.1)")
    parser.add_argument("--window-size", type=int, default=15)
    parser.add_argument("--stride", type=int, default=None,
                        help="Window stride (default: same as window-size)")
    parser.add_argument("--max-samples-per-class", type=int, default=2500,
                        help="Class-balanced cap after window extraction; 0 disables")
    parser.add_argument("--parquet-cache-dir", default=None,
                        help="Parquet cache root (default: outputs/cache/parquet)")
    parser.add_argument("--no-parquet-cache", action="store_true",
                        help="Disable Parquet cache and read raw CSV files")
    parser.add_argument("--oracle-only", action="store_true",
                        help="Only train system oracle models (skip per-node, ~80%% faster)")
    parser.add_argument("--legacy-per-node", action="store_true",
                        help="Also train legacy per-node multi-class models (slow/weak; disabled by default)")
    parser.add_argument("--with-responsibility", action="store_true",
                        help="Also train binary per-node responsibility models")
    parser.add_argument("--responsibility-only", action="store_true",
                        help="Only train binary per-node responsibility models")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()

    # Build topology first
    builder = TopologyBuilder(args.config, args.data_root)
    builder.build()

    systems = args.systems.split(",") if args.systems else ALL_SYSTEMS
    output_dir = ensure_dir(args.output_dir)
    all_results = {}

    for sys_id in systems:
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing system: {sys_id}")
        logger.info(f"{'='*60}")

        # Load data
        df, label_map = load_system_data(
            args.data_root,
            sys_id,
            max_rows_per_file=args.max_rows,
            sample_frac=args.sample_frac,
            parquet_cache_dir=args.parquet_cache_dir,
            use_parquet_cache=not args.no_parquet_cache,
        )

        if df.empty:
            logger.warning(f"No data loaded for {sys_id}, skipping")
            continue

        # Train models
        stride = args.stride or args.window_size
        if args.responsibility_only:
            results = train_all_responsibility_models(
                system_df=df,
                system_id=sys_id,
                topology_builder=builder,
                label_map=label_map,
                output_dir=output_dir,
                window_size=args.window_size,
                stride=stride,
                max_samples_per_class=args.max_samples_per_class or None,
            )
        elif args.oracle_only:
            if args.with_responsibility:
                logger.warning("--with-responsibility is ignored when --oracle-only is set")
            result = train_system_oracle_model(
                system_df=df,
                system_id=sys_id,
                topology_builder=builder,
                label_map=label_map,
                output_dir=output_dir,
                window_size=args.window_size,
                stride=stride,
                max_samples_per_class=args.max_samples_per_class or None,
            )
            results = [result]
        elif args.legacy_per_node:
            results = train_all_system_models(
                system_df=df,
                system_id=sys_id,
                topology_builder=builder,
                label_map=label_map,
                output_dir=output_dir,
                window_size=args.window_size,
                stride=stride,
                train_responsibility=args.with_responsibility,
                max_samples_per_class=args.max_samples_per_class or None,
            )
        else:
            result = train_system_oracle_model(
                system_df=df,
                system_id=sys_id,
                topology_builder=builder,
                label_map=label_map,
                output_dir=output_dir,
                window_size=args.window_size,
                stride=stride,
                max_samples_per_class=args.max_samples_per_class or None,
            )
            results = [result]
            if args.with_responsibility:
                resp_results = train_all_responsibility_models(
                    system_df=df,
                    system_id=sys_id,
                    topology_builder=builder,
                    label_map=label_map,
                    output_dir=output_dir,
                    window_size=args.window_size,
                    stride=stride,
                    max_samples_per_class=args.max_samples_per_class or None,
                )
                results.extend(resp_results)

        all_results[sys_id] = {
            "n_components": len(results),
            "n_trained": sum(1 for r in results if r.get("status") == "trained"),
            "n_skipped": sum(1 for r in results if r.get("status") == "skipped"),
            "label_map": label_map,
            "models": results,
        }

    # Save summary
    if args.responsibility_only:
        summary_path = os.path.join(output_dir, "responsibility_training_run_summary.json")
        save_json(all_results, summary_path)
    else:
        summary_path = _save_training_summary(output_dir, all_results)
    logger.info(f"\nTraining summary saved to {summary_path}")
    if args.responsibility_only or args.with_responsibility:
        _update_responsibility_summary(output_dir, all_results)

    # Print summary table
    logger.info("\n=== Training Summary ===")
    logger.info(f"{'System':<20} {'Components':<12} {'Trained':<10} {'Skipped':<10}")
    logger.info("-" * 52)
    for sys_id, res in all_results.items():
        logger.info(
            f"{sys_id:<20} {res['n_components']:<12} "
            f"{res['n_trained']:<10} {res['n_skipped']:<10}"
        )


if __name__ == "__main__":
    main()
