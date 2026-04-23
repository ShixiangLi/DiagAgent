"""
TTL Parser — Extract topology information from Brick Schema TTL files.

Parses RDF Turtle files that describe HVAC system components and their
relationships using the Brick ontology. Extracts:
  - Component hierarchy (hasPart)
  - Flow connections (feeds)
  - Sensor-to-component mappings (hasPoint)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from rdflib import Graph, Namespace, URIRef
from rdflib.namespace import RDF

from src.utils.io_utils import setup_logger

logger = setup_logger(__name__)

BRICK = Namespace("https://brickschema.org/schema/Brick#")


@dataclass
class TTLComponent:
    """A component extracted from a TTL file."""
    uri: str                            # Full URI of the component
    local_name: str                     # Short name (e.g., "Chiller_1")
    brick_class: str                    # Brick type (e.g., "AHU", "Chiller")
    children: List[str] = field(default_factory=list)    # hasPart children (local names)
    feeds: List[str] = field(default_factory=list)       # feeds targets (local names)
    sensors: List[str] = field(default_factory=list)     # hasPoint sensor names
    sensor_types: Dict[str, str] = field(default_factory=dict)  # sensor_name → brick class


@dataclass
class TTLParseResult:
    """Result of parsing a single TTL file."""
    system_id: str
    root_components: List[str]           # Top-level components (not children of others)
    components: Dict[str, TTLComponent]  # local_name → component
    all_sensors: Dict[str, str]          # sensor_name → brick sensor class
    feed_edges: List[Tuple[str, str]]    # (source, target) feed relationships


def _extract_local_name(uri: URIRef, prefixes: Dict[str, str]) -> str:
    """Extract the local name from a URI, stripping the namespace prefix."""
    uri_str = str(uri)
    for prefix_uri in prefixes.values():
        if uri_str.startswith(prefix_uri):
            return uri_str[len(prefix_uri):]
    # Fallback: split on # or /
    if "#" in uri_str:
        return uri_str.split("#")[-1]
    return uri_str.split("/")[-1]


def _is_sensor_or_point_class(brick_class: str) -> bool:
    """Check if a Brick class represents a sensor, setpoint, status, or command."""
    point_keywords = [
        "Sensor", "Setpoint", "Status", "Command", "Meter",
        "coolingCapacity", "Demand",
    ]
    return any(kw in brick_class for kw in point_keywords)


def parse_ttl_file(ttl_path: str, system_id: str) -> TTLParseResult:
    """
    Parse a Brick Schema TTL file and extract the component hierarchy.

    Args:
        ttl_path: Path to the .ttl file.
        system_id: Identifier for this system (e.g., "chiller_plant").

    Returns:
        TTLParseResult with extracted components, sensors, and relationships.
    """
    logger.info(f"Parsing TTL file for system '{system_id}': {ttl_path}")

    g = Graph()
    g.parse(ttl_path, format="turtle")

    # Collect namespace prefixes for local name extraction
    prefixes = {}
    for prefix, ns_uri in g.namespaces():
        prefixes[prefix] = str(ns_uri)

    # ---- Step 1: Identify all entities and their Brick classes ----
    entity_classes: Dict[str, str] = {}  # local_name → brick class
    for subj, _, obj in g.triples((None, RDF.type, None)):
        obj_str = str(obj)
        if "brickschema.org" in obj_str:
            local = _extract_local_name(subj, prefixes)
            brick_class = obj_str.split("#")[-1] if "#" in obj_str else obj_str.split("/")[-1]
            entity_classes[local] = brick_class

    # ---- Step 2: Classify entities as components vs sensors/points ----
    components: Dict[str, TTLComponent] = {}
    all_sensors: Dict[str, str] = {}

    for local_name, brick_class in entity_classes.items():
        if _is_sensor_or_point_class(brick_class):
            all_sensors[local_name] = brick_class
        else:
            components[local_name] = TTLComponent(
                uri=local_name,
                local_name=local_name,
                brick_class=brick_class,
            )

    # ---- Step 3: Extract hasPart relationships ----
    has_part = URIRef(str(BRICK) + "hasPart")
    child_set: Set[str] = set()

    for subj, _, obj in g.triples((None, has_part, None)):
        parent_name = _extract_local_name(subj, prefixes)
        child_name = _extract_local_name(obj, prefixes)

        if parent_name in components:
            components[parent_name].children.append(child_name)
            child_set.add(child_name)
            # If child is not yet a component, create it
            if child_name not in components and child_name not in all_sensors:
                components[child_name] = TTLComponent(
                    uri=child_name,
                    local_name=child_name,
                    brick_class=entity_classes.get(child_name, "Unknown"),
                )

    # ---- Step 4: Extract feeds relationships ----
    feeds_pred = URIRef(str(BRICK) + "feeds")
    feed_edges: List[Tuple[str, str]] = []

    for subj, _, obj in g.triples((None, feeds_pred, None)):
        source = _extract_local_name(subj, prefixes)
        target = _extract_local_name(obj, prefixes)
        feed_edges.append((source, target))

        if source in components:
            components[source].feeds.append(target)
        # Ensure feed targets exist as components
        if target not in components and target not in all_sensors:
            components[target] = TTLComponent(
                uri=target,
                local_name=target,
                brick_class=entity_classes.get(target, "HVAC_Zone"),
            )

    # ---- Step 5: Extract hasPoint relationships (sensor assignments) ----
    has_point = URIRef(str(BRICK) + "hasPoint")

    for subj, _, obj in g.triples((None, has_point, None)):
        parent_name = _extract_local_name(subj, prefixes)
        sensor_name = _extract_local_name(obj, prefixes)

        if parent_name in components:
            components[parent_name].sensors.append(sensor_name)
            sensor_class = entity_classes.get(sensor_name, "Unknown_Sensor")
            components[parent_name].sensor_types[sensor_name] = sensor_class

            # Make sure this sensor is in the global sensor map
            if sensor_name not in all_sensors:
                all_sensors[sensor_name] = sensor_class

    # ---- Step 6: Identify root components (not children of anything) ----
    root_components = [
        name for name in components
        if name not in child_set
        # Also exclude zones that are only feed targets, not structural roots
        and (components[name].children or components[name].sensors
             or not any(name == t for _, t in feed_edges))
    ]

    # If no roots found via hierarchy, use top-level typed entities
    if not root_components:
        top_types = {"Water_System", "Hot_Water_System", "AHU", "RTU"}
        root_components = [
            n for n, c in components.items()
            if c.brick_class in top_types
        ]

    logger.info(
        f"  Parsed {len(components)} components, {len(all_sensors)} sensors, "
        f"{len(feed_edges)} feed edges, {len(root_components)} roots"
    )

    return TTLParseResult(
        system_id=system_id,
        root_components=root_components,
        components=components,
        all_sensors=all_sensors,
        feed_edges=feed_edges,
    )
