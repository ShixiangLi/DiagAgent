"""
Topology Builder — Construct a unified multi-tier topology graph.

Builds a NetworkX DiGraph representing the full building HVAC topology:
  Level 0: Building (single root)
  Level 1: Systems (Chiller Plant, Boiler Plant, AHU, RTU, etc.)
  Level 2: Nodes/Components (Chillers, Pumps, Coils, Fans, Dampers, VAV Boxes)
  Level 3: Sensors (individual data points)

Each node in the graph carries metadata: node_id, level, system_id,
brick_class, sensors, fault_types, and descriptive attributes.
"""

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from src.topology.ttl_parser import TTLParseResult, parse_ttl_file
from src.utils.io_utils import load_yaml, save_json, setup_logger

logger = setup_logger(__name__)

# Edge type constants
EDGE_HAS_PART = "has_part"
EDGE_FEEDS = "feeds"
EDGE_HAS_SENSOR = "has_sensor"
EDGE_CROSS_SYSTEM = "cross_system"

# Node level constants
LEVEL_BUILDING = 0
LEVEL_SYSTEM = 1
LEVEL_NODE = 2
LEVEL_SENSOR = 3


def _build_node_id(system_id: str, component_name: str) -> str:
    """Build a globally unique node ID: system_id::component_name."""
    return f"{system_id}::{component_name}"


class TopologyBuilder:
    """
    Constructs and manages the multi-tier HVAC topology graph.

    The graph is a NetworkX DiGraph where edges carry a 'relation' attribute
    indicating the type of connection (has_part, feeds, has_sensor, cross_system).
    """

    def __init__(self, config_path: str, data_root: str):
        """
        Args:
            config_path: Path to topology_config.yaml.
            data_root: Path to the data/lbnl/ directory containing TTL files.
        """
        self.config = load_yaml(config_path)
        self.data_root = data_root
        self.graph = nx.DiGraph()
        self._system_parse_results: Dict[str, TTLParseResult] = {}

    def build(self) -> nx.DiGraph:
        """Build the complete topology graph."""
        logger.info("=== Building Multi-Tier HVAC Topology ===")

        # Step 1: Create building root
        self._add_building_root()

        # Step 2: Parse each system's TTL and add to graph
        for sys_id, sys_config in self.config["systems"].items():
            ttl_path = os.path.join(self.data_root, sys_config["ttl_file"])
            if not os.path.exists(ttl_path):
                logger.warning(f"TTL file not found for {sys_id}: {ttl_path}")
                continue

            parse_result = parse_ttl_file(ttl_path, sys_id)
            self._system_parse_results[sys_id] = parse_result
            self._add_system_to_graph(sys_id, sys_config, parse_result)

        # Step 3: Add cross-system links
        self._add_cross_system_links()

        logger.info(
            f"Topology built: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
        self._log_summary()

        return self.graph

    def _add_building_root(self) -> None:
        """Add the Level 0 building root node."""
        self.graph.add_node(
            "building",
            level=LEVEL_BUILDING,
            node_type="Building",
            name="LBNL_FDD_Building",
            description="Simulated HVAC building from LBNL FDD datasets",
        )

    def _add_system_to_graph(
        self,
        sys_id: str,
        sys_config: Dict[str, Any],
        parse_result: TTLParseResult,
    ) -> None:
        """Add a parsed system and all its components/sensors to the graph."""

        # Add Level 1: System node
        system_node_id = f"system::{sys_id}"
        self.graph.add_node(
            system_node_id,
            level=LEVEL_SYSTEM,
            node_type="System",
            system_id=sys_id,
            name=sys_config["name"],
            system_type=sys_config["system_type"],
            description=sys_config.get("description", ""),
            data_dir=sys_config["data_dir"],
        )
        self.graph.add_edge(
            "building", system_node_id, relation=EDGE_HAS_PART
        )

        # Add Level 2: Component nodes
        added_components: Set[str] = set()
        for comp_name, comp in parse_result.components.items():
            comp_node_id = _build_node_id(sys_id, comp_name)
            self.graph.add_node(
                comp_node_id,
                level=LEVEL_NODE,
                node_type="Node",
                system_id=sys_id,
                name=comp_name,
                brick_class=comp.brick_class,
                sensor_names=comp.sensors,
                sensor_types=comp.sensor_types,
            )
            added_components.add(comp_name)

        # Add hasPart edges (system → root components, component → sub-components)
        for root_name in parse_result.root_components:
            root_id = _build_node_id(sys_id, root_name)
            if root_id in self.graph:
                self.graph.add_edge(
                    system_node_id, root_id, relation=EDGE_HAS_PART
                )

        for comp_name, comp in parse_result.components.items():
            parent_id = _build_node_id(sys_id, comp_name)
            for child_name in comp.children:
                child_id = _build_node_id(sys_id, child_name)
                if child_id in self.graph:
                    self.graph.add_edge(
                        parent_id, child_id, relation=EDGE_HAS_PART
                    )

        # Add feeds edges
        for source_name, target_name in parse_result.feed_edges:
            source_id = _build_node_id(sys_id, source_name)
            target_id = _build_node_id(sys_id, target_name)
            if source_id in self.graph and target_id in self.graph:
                self.graph.add_edge(
                    source_id, target_id, relation=EDGE_FEEDS
                )

        # Add Level 3: Sensor nodes
        for comp_name, comp in parse_result.components.items():
            comp_id = _build_node_id(sys_id, comp_name)
            for sensor_name in comp.sensors:
                sensor_id = _build_node_id(sys_id, sensor_name)
                sensor_class = comp.sensor_types.get(
                    sensor_name, parse_result.all_sensors.get(sensor_name, "Unknown")
                )
                if sensor_id not in self.graph:
                    self.graph.add_node(
                        sensor_id,
                        level=LEVEL_SENSOR,
                        node_type="Sensor",
                        system_id=sys_id,
                        name=sensor_name,
                        brick_class=sensor_class,
                    )
                self.graph.add_edge(
                    comp_id, sensor_id, relation=EDGE_HAS_SENSOR
                )

        logger.info(
            f"  Added system '{sys_id}': "
            f"{len(added_components)} components, "
            f"{sum(len(c.sensors) for c in parse_result.components.values())} sensors"
        )

    def _add_cross_system_links(self) -> None:
        """Add inter-system edges from config."""
        links = self.config.get("cross_system_links", [])
        added = 0

        for link in links:
            source_id = _build_node_id(link["source_system"], link["source_node"])
            target_id = _build_node_id(link["target_system"], link["target_node"])

            # Both nodes must exist in the graph
            if source_id in self.graph and target_id in self.graph:
                self.graph.add_edge(
                    source_id,
                    target_id,
                    relation=EDGE_CROSS_SYSTEM,
                    medium=link["medium"],
                    description=link.get("description", ""),
                )
                added += 1
            else:
                # Try fuzzy matching: find nodes containing the target name
                matched = False
                if source_id not in self.graph:
                    candidates = [
                        n for n in self.graph.nodes
                        if link["source_node"] in n and link["source_system"] in n
                    ]
                    if candidates:
                        source_id = candidates[0]
                if target_id not in self.graph:
                    candidates = [
                        n for n in self.graph.nodes
                        if link["target_node"] in n and link["target_system"] in n
                    ]
                    if candidates:
                        target_id = candidates[0]

                if source_id in self.graph and target_id in self.graph:
                    self.graph.add_edge(
                        source_id,
                        target_id,
                        relation=EDGE_CROSS_SYSTEM,
                        medium=link["medium"],
                        description=link.get("description", ""),
                    )
                    added += 1
                    matched = True

                if not matched:
                    logger.warning(
                        f"  Cross-system link skipped: "
                        f"{link['source_system']}.{link['source_node']} → "
                        f"{link['target_system']}.{link['target_node']}"
                    )

        logger.info(f"  Added {added}/{len(links)} cross-system links")

    def _log_summary(self) -> None:
        """Log a summary of the built topology."""
        level_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        for _, data in self.graph.nodes(data=True):
            level_counts[data.get("level", -1)] = (
                level_counts.get(data.get("level", -1), 0) + 1
            )

        edge_type_counts: Dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            rel = data.get("relation", "unknown")
            edge_type_counts[rel] = edge_type_counts.get(rel, 0) + 1

        logger.info("  Topology Summary:")
        logger.info(f"    Building nodes: {level_counts.get(0, 0)}")
        logger.info(f"    System nodes:   {level_counts.get(1, 0)}")
        logger.info(f"    Component nodes:{level_counts.get(2, 0)}")
        logger.info(f"    Sensor nodes:   {level_counts.get(3, 0)}")
        for rel, count in edge_type_counts.items():
            logger.info(f"    {rel} edges: {count}")

    # ----- Query methods for the built graph -----

    def get_systems(self) -> List[Dict[str, Any]]:
        """Return all system-level nodes."""
        return [
            {"node_id": n, **self.graph.nodes[n]}
            for n in self.graph.nodes
            if self.graph.nodes[n].get("level") == LEVEL_SYSTEM
        ]

    def get_children(self, node_id: str) -> List[Dict[str, Any]]:
        """Return child nodes connected via has_part edges."""
        children = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            if data.get("relation") == EDGE_HAS_PART:
                children.append({"node_id": target, **self.graph.nodes[target]})
        return children

    def get_sensors(self, node_id: str) -> List[Dict[str, Any]]:
        """Return sensor nodes connected via has_sensor edges."""
        sensors = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            if data.get("relation") == EDGE_HAS_SENSOR:
                sensors.append({"node_id": target, **self.graph.nodes[target]})
        return sensors

    def get_downstream_nodes(self, node_id: str) -> List[Dict[str, Any]]:
        """Return nodes downstream (fed by) this node via feeds or cross_system edges."""
        downstream = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            if data.get("relation") in (EDGE_FEEDS, EDGE_CROSS_SYSTEM):
                downstream.append({
                    "node_id": target,
                    "relation": data["relation"],
                    "medium": data.get("medium", ""),
                    **self.graph.nodes[target],
                })
        return downstream

    def get_upstream_nodes(self, node_id: str) -> List[Dict[str, Any]]:
        """Return nodes upstream of (feeding into) this node."""
        upstream = []
        for source, _, data in self.graph.in_edges(node_id, data=True):
            if data.get("relation") in (EDGE_FEEDS, EDGE_CROSS_SYSTEM):
                upstream.append({
                    "node_id": source,
                    "relation": data["relation"],
                    "medium": data.get("medium", ""),
                    **self.graph.nodes[source],
                })
        return upstream

    def get_all_component_nodes(self) -> List[Dict[str, Any]]:
        """Return all Level 2 component nodes."""
        return [
            {"node_id": n, **self.graph.nodes[n]}
            for n in self.graph.nodes
            if self.graph.nodes[n].get("level") == LEVEL_NODE
        ]

    def get_node_info(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return full info about a specific node, or None if not found."""
        if node_id not in self.graph:
            return None
        return {"node_id": node_id, **self.graph.nodes[node_id]}

    def get_system_components(self, system_id: str) -> List[Dict[str, Any]]:
        """Return all component nodes belonging to a specific system."""
        return [
            {"node_id": n, **self.graph.nodes[n]}
            for n in self.graph.nodes
            if self.graph.nodes[n].get("level") == LEVEL_NODE
            and self.graph.nodes[n].get("system_id") == system_id
        ]

    def export_topology_json(self, output_path: str) -> None:
        """Export the topology to a JSON file for inspection."""
        topology_data = {
            "nodes": [],
            "edges": [],
        }
        for node_id, data in self.graph.nodes(data=True):
            node_info = {"node_id": node_id}
            node_info.update({
                k: v for k, v in data.items()
                if isinstance(v, (str, int, float, bool, list))
            })
            # Convert dict values to serializable form
            for k, v in data.items():
                if isinstance(v, dict):
                    node_info[k] = dict(v)
            topology_data["nodes"].append(node_info)

        for source, target, data in self.graph.edges(data=True):
            edge_info = {"source": source, "target": target}
            edge_info.update(data)
            topology_data["edges"].append(edge_info)

        save_json(topology_data, output_path)
        logger.info(f"Topology exported to {output_path}")
