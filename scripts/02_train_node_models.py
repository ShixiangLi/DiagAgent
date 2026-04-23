"""
Script 02: Train node-level fault prediction models.

Usage:
    python scripts/02_train_node_models.py [--systems chiller_plant,boiler_plant] [--validate]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.topology.topology_builder import TopologyBuilder
from src.node_models.data_loader import load_system_data
from src.node_models.model_trainer import train_all_system_models
from src.utils.io_utils import setup_logger, ensure_dir, save_json

logger = setup_logger("train_node_models")

ALL_SYSTEMS = [
    "chiller_plant", "boiler_plant", "sdahu", "ddahu",
    "rtu", "fcu", "pfpu", "sfpu",
]


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
    parser.add_argument("--oracle-only", action="store_true",
                        help="Only train system oracle models (skip per-node, ~80%% faster)")
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
        )

        if df.empty:
            logger.warning(f"No data loaded for {sys_id}, skipping")
            continue

        # Train models
        stride = args.stride or args.window_size
        if args.oracle_only:
            from src.node_models.model_trainer import train_system_oracle_model
            result = train_system_oracle_model(
                system_df=df,
                system_id=sys_id,
                topology_builder=builder,
                label_map=label_map,
                output_dir=output_dir,
                window_size=args.window_size,
                stride=stride,
            )
            results = [result]
        else:
            results = train_all_system_models(
                system_df=df,
                system_id=sys_id,
                topology_builder=builder,
                label_map=label_map,
                output_dir=output_dir,
                window_size=args.window_size,
                stride=stride,
            )

        all_results[sys_id] = {
            "n_components": len(results),
            "n_trained": sum(1 for r in results if r.get("status") == "trained"),
            "n_skipped": sum(1 for r in results if r.get("status") == "skipped"),
            "label_map": label_map,
            "models": results,
        }

    # Save summary
    summary_path = os.path.join(output_dir, "training_summary.json")
    save_json(all_results, summary_path)
    logger.info(f"\nTraining summary saved to {summary_path}")

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
