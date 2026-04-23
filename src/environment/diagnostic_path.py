"""
Diagnostic Path — Generate guided diagnostic paths for fault scenarios.

For each fault scenario, builds an ordered path of nodes the agent should
traverse to find the root cause. Each node on the path carries an abnormal
hint and direction to guide the agent deeper.

Path node roles:
  - "symptom": starting point (downstream), shows observable symptoms
  - "intermediate": mid-path, shows abnormal indicators + direction
  - "root_cause": end point, returns definitive fault diagnosis
"""

import fnmatch
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from src.utils.io_utils import load_yaml, setup_logger

logger = setup_logger(__name__)


@dataclass
class PathNode:
    """A single node in the diagnostic path."""
    node_id: str
    role: str              # "symptom" | "intermediate" | "root_cause"
    system_id: str
    component_name: str
    abnormal_hint: str     # Human-readable abnormal indicator
    direction: str         # "upstream" | "downstream" | "check_component"


@dataclass
class DiagnosticPath:
    """Complete diagnostic path for a scenario."""
    nodes: List[PathNode]
    root_cause_node: str
    fault_type: str
    symptom_description: str  # What the user observes

    @property
    def path_length(self) -> int:
        return len(self.nodes)

    @property
    def symptom_node(self) -> str:
        return self.nodes[0].node_id if self.nodes else ""

    @property
    def node_ids(self) -> List[str]:
        return [n.node_id for n in self.nodes]

    def get_node_role(self, node_id: str) -> Optional[str]:
        """Return the role of a node on this path, or None if off-path."""
        for n in self.nodes:
            if n.node_id == node_id:
                return n.role
        return None

    def get_path_node(self, node_id: str) -> Optional[PathNode]:
        """Return the PathNode for a given node_id, or None if off-path."""
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None


class DiagnosticPathGenerator:
    """
    Generates diagnostic paths by combining topology traversal
    with manually-defined path templates.
    """

    def __init__(self, config_path: str = "configs/diagnostic_paths.yaml"):
        self.config = load_yaml(config_path)
        self.hint_library = self.config.get("abnormal_hints", {})
        self.templates = self.config.get("path_templates", {})
        self.settings = self.config.get("settings", {})
        self.max_path_length = self.settings.get("max_path_length", 6)
        self.min_path_length = self.settings.get("min_path_length", 2)

    def generate_path(
        self,
        fault_type: str,
        root_cause_node: str,
        root_cause_system: str,
        topology_builder,
        downstream_system: Optional[str] = None,
        rng: Optional[random.Random] = None,
    ) -> DiagnosticPath:
        """
        Generate a diagnostic path for a given fault scenario.

        Args:
            fault_type: Full fault label (e.g., "coolingtower_fouling_080").
            root_cause_node: Node ID of the root cause.
            root_cause_system: System ID containing the root cause.
            topology_builder: TopologyBuilder with built graph.
            downstream_system: For cross-system, the downstream system ID.
            rng: Random generator for diversity.

        Returns:
            DiagnosticPath with ordered nodes from symptom to root cause.
        """
        if rng is None:
            rng = random.Random(42)

        # Extract base fault category (remove intensity suffix)
        fault_category = self._extract_fault_category(fault_type)

        # Look up template
        template = self.templates.get(fault_category, {})

        # Build path nodes
        path_nodes: List[PathNode] = []

        # 1. Add downstream symptom nodes (for cross-system scenarios)
        if downstream_system and downstream_system != root_cause_system:
            symptom_nodes = self._build_downstream_symptoms(
                template, downstream_system, topology_builder, rng
            )
            path_nodes.extend(symptom_nodes)

        # 2. Add within-system intermediate nodes
        intermediate_nodes = self._build_within_system_path(
            template, root_cause_system, root_cause_node, topology_builder, rng
        )
        path_nodes.extend(intermediate_nodes)

        # 3. Add root cause node
        root_name = root_cause_node.split("::")[-1] if "::" in root_cause_node else root_cause_node
        path_nodes.append(PathNode(
            node_id=root_cause_node,
            role="root_cause",
            system_id=root_cause_system,
            component_name=root_name,
            abnormal_hint=f"Definitive fault detected: {fault_type}",
            direction="root_cause_confirmed",
        ))

        # 4. Mark the first node as "symptom" if we have more than 1 node
        if len(path_nodes) > 1:
            path_nodes[0].role = "symptom"

        # 5. Enforce path length limits
        path_nodes = self._enforce_length_limits(path_nodes)

        # 6. Build symptom description
        symptom_desc = self._build_symptom_description(path_nodes, fault_type)

        return DiagnosticPath(
            nodes=path_nodes,
            root_cause_node=root_cause_node,
            fault_type=fault_type,
            symptom_description=symptom_desc,
        )

    def _extract_fault_category(self, fault_type: str) -> str:
        """
        Extract the base fault category from a full fault label.

        Examples:
            "coolingtower_fouling_080" -> "coolingtower_fouling"
            "chiller_bias_-2" -> "chiller_bias"
            "bypass_leakage_025" -> "bypass_leakage"
        """
        # Try progressively shorter prefixes until we find a template match
        parts = fault_type.split("_")
        for end in range(len(parts), 0, -1):
            candidate = "_".join(parts[:end])
            if candidate in self.templates:
                return candidate

        # Fallback: try removing the last part (usually intensity)
        if len(parts) >= 2:
            return "_".join(parts[:-1])
        return fault_type

    def _build_downstream_symptoms(
        self,
        template: Dict,
        downstream_system: str,
        topology_builder,
        rng: random.Random,
    ) -> List[PathNode]:
        """Build symptom path nodes from downstream system."""
        nodes = []
        ds_symptoms = template.get("downstream_symptoms", [])

        # Find matching downstream symptom for this system
        matching = [s for s in ds_symptoms if s.get("system") == downstream_system]

        if matching:
            symptom_def = rng.choice(matching)
            comp_pattern = symptom_def.get("component", "*")
            hint_key = symptom_def.get("hint", "zone_temp_rising")

            # Resolve component pattern to actual node
            node_id = self._resolve_component(
                downstream_system, comp_pattern, topology_builder
            )
            if node_id:
                hint_text = self.hint_library.get(hint_key, hint_key)
                comp_name = node_id.split("::")[-1] if "::" in node_id else node_id
                nodes.append(PathNode(
                    node_id=node_id,
                    role="symptom",
                    system_id=downstream_system,
                    component_name=comp_name,
                    abnormal_hint=hint_text,
                    direction="upstream",
                ))
        else:
            # Auto-generate: pick a component from the downstream system
            ds_components = topology_builder.get_system_components(downstream_system)
            if ds_components:
                comp = rng.choice(ds_components)
                nodes.append(PathNode(
                    node_id=comp["node_id"],
                    role="symptom",
                    system_id=downstream_system,
                    component_name=comp.get("name", ""),
                    abnormal_hint="Operating parameters are outside normal ranges. The root cause may be in an upstream system.",
                    direction="upstream",
                ))

        return nodes

    def _build_within_system_path(
        self,
        template: Dict,
        system_id: str,
        root_cause_node: str,
        topology_builder,
        rng: random.Random,
    ) -> List[PathNode]:
        """Build intermediate path nodes within the root cause system."""
        nodes = []
        within_path = template.get("within_system_path", [])

        for step_def in within_path:
            comp_pattern = step_def.get("component", "*")
            hint_key = step_def.get("hint", "")
            direction = step_def.get("direction", "upstream")

            node_id = self._resolve_component(
                system_id, comp_pattern, topology_builder
            )
            if node_id and node_id != root_cause_node:
                hint_text = self.hint_library.get(hint_key, hint_key)
                comp_name = node_id.split("::")[-1] if "::" in node_id else node_id
                nodes.append(PathNode(
                    node_id=node_id,
                    role="intermediate",
                    system_id=system_id,
                    component_name=comp_name,
                    abnormal_hint=hint_text,
                    direction=direction,
                ))

        return nodes

    def _resolve_component(
        self,
        system_id: str,
        pattern: str,
        topology_builder,
    ) -> Optional[str]:
        """
        Resolve a component name pattern to an actual node_id.

        Supports wildcards like "Chiller_*", "Cooling_Tower_*".
        """
        components = topology_builder.get_system_components(system_id)
        for comp in components:
            name = comp.get("name", "")
            if fnmatch.fnmatch(name, pattern):
                return comp["node_id"]
        # Fallback: partial match
        for comp in components:
            name = comp.get("name", "")
            base_pattern = pattern.replace("_*", "").replace("*", "")
            if base_pattern and base_pattern in name:
                return comp["node_id"]
        return None

    def _enforce_length_limits(self, nodes: List[PathNode]) -> List[PathNode]:
        """Trim path to max length, keeping symptom + root_cause."""
        if len(nodes) <= self.max_path_length:
            return nodes

        # Keep first (symptom) and last (root_cause), trim middle
        keep = [nodes[0]]
        middle = nodes[1:-1]
        # Evenly sample from middle to fit limit
        n_middle = self.max_path_length - 2
        if n_middle > 0 and middle:
            step = max(1, len(middle) // n_middle)
            keep.extend(middle[::step][:n_middle])
        keep.append(nodes[-1])
        return keep

    def _build_symptom_description(
        self,
        path_nodes: List[PathNode],
        fault_type: str,
    ) -> str:
        """Build user-facing symptom description from the first path node."""
        if not path_nodes:
            return "An HVAC fault has been detected. Please investigate."

        first = path_nodes[0]
        return first.abnormal_hint


def generate_no_fault_path(
    system_id: str,
    topology_builder,
    rng: Optional[random.Random] = None,
) -> DiagnosticPath:
    """
    Generate a diagnostic path for a no-fault scenario.

    All nodes return Normal. Path covers 2-3 components for verification.
    """
    if rng is None:
        rng = random.Random(42)

    components = topology_builder.get_system_components(system_id)
    if not components:
        return DiagnosticPath(
            nodes=[], root_cause_node="none",
            fault_type="Normal", symptom_description="Routine check requested.",
        )

    # Pick 2-3 components to verify
    n_check = min(rng.randint(2, 3), len(components))
    selected = rng.sample(components, n_check)

    path_nodes = []
    for comp in selected:
        path_nodes.append(PathNode(
            node_id=comp["node_id"],
            role="intermediate",  # All return Normal
            system_id=system_id,
            component_name=comp.get("name", ""),
            abnormal_hint="All parameters within normal operating ranges.",
            direction="check_component",
        ))

    return DiagnosticPath(
        nodes=path_nodes,
        root_cause_node="none",
        fault_type="Normal",
        symptom_description="Routine diagnostic check requested. Please verify system health.",
    )
