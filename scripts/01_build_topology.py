"""
Script 01: Build and validate the multi-tier HVAC topology.

Usage:
    python scripts/01_build_topology.py [--validate] [--export]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.topology.topology_builder import TopologyBuilder
from src.topology.topology_tools import TopologyToolExecutor
from src.utils.io_utils import setup_logger, ensure_dir

logger = setup_logger("build_topology")


def main():
    parser = argparse.ArgumentParser(description="Build HVAC topology")
    parser.add_argument("--config", default="configs/topology_config.yaml")
    parser.add_argument("--data-root", default="data/lbnl")
    parser.add_argument("--output-dir", default="outputs/topology")
    parser.add_argument("--validate", action="store_true", help="Run validation checks")
    parser.add_argument("--export", action="store_true", help="Export topology JSON")
    args = parser.parse_args()

    # Build topology
    builder = TopologyBuilder(args.config, args.data_root)
    graph = builder.build()

    # Export
    if args.export or True:  # Always export
        output_dir = ensure_dir(args.output_dir)
        builder.export_topology_json(os.path.join(output_dir, "topology.json"))

    # Validate
    if args.validate:
        logger.info("\n=== Running Validation ===")

        # Check all 8 systems are present
        systems = builder.get_systems()
        system_ids = {s["system_id"] for s in systems}
        expected = {"chiller_plant", "boiler_plant", "sdahu", "ddahu", "rtu", "fcu", "pfpu", "sfpu"}
        missing = expected - system_ids
        if missing:
            logger.warning(f"Missing systems: {missing}")
        else:
            logger.info(f"✓ All {len(expected)} systems present")

        # Check component count per system
        for sys in systems:
            comps = builder.get_system_components(sys["system_id"])
            logger.info(f"  {sys['system_id']}: {len(comps)} components")

        # Check cross-system links exist
        cross_edges = [
            (u, v) for u, v, d in graph.edges(data=True)
            if d.get("relation") == "cross_system"
        ]
        logger.info(f"✓ {len(cross_edges)} cross-system links established")

        # Test tool executor
        executor = TopologyToolExecutor(builder)
        result = executor.execute("get_system_overview", {})
        assert result["status"] == "success", "get_system_overview failed"
        logger.info(f"✓ Tool executor working ({result['count']} systems)")

        # Test downstream/upstream queries
        if "chiller_plant" in system_ids:
            comps = builder.get_system_components("chiller_plant")
            for comp in comps:
                ds = builder.get_downstream_nodes(comp["node_id"])
                if ds:
                    logger.info(f"  {comp['node_id']} → {len(ds)} downstream nodes")

        logger.info("\n=== Validation Complete ===")

    return builder


if __name__ == "__main__":
    main()
