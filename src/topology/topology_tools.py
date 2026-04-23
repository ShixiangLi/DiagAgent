"""
Topology Tools — Agent-callable tool definitions and implementations.

Defines the JSON tool schemas the LLM agent can invoke and wraps the
TopologyBuilder query methods into a standardized tool execution interface.
"""

from typing import Any, Dict, List, Optional

from src.utils.io_utils import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Tool Schema Definitions (for the agent's system prompt)
# ============================================================================

TOPOLOGY_TOOL_SCHEMAS = [
    {
        "name": "get_system_overview",
        "description": (
            "Get an overview of all HVAC systems in the building. Returns a list "
            "of top-level systems with their IDs, names, types, and descriptions. "
            "Use this as the first step to understand the building layout."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_node_children",
        "description": (
            "Get the child components of a given node in the topology hierarchy. "
            "For a system node, returns its major components (e.g., chillers, pumps). "
            "For a component, returns its sub-components. "
            "Use this to drill down into a system's internal structure."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": (
                        "The ID of the node to get children for. "
                        "Format: 'system::SystemID' for systems or "
                        "'system_id::ComponentName' for components."
                    ),
                },
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "get_downstream_nodes",
        "description": (
            "Get nodes that are downstream of (fed by) a given node. This follows "
            "the thermal/fluid/air flow direction. For example, a chiller plant's "
            "downstream includes AHU cooling coils. Use this to trace how a fault "
            "in one component might propagate to other parts of the system."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "The ID of the node to find downstream connections for.",
                },
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "get_upstream_nodes",
        "description": (
            "Get nodes that are upstream of (feeding into) a given node. This traces "
            "back along the thermal/fluid/air flow direction. For example, an AHU "
            "cooling coil's upstream is the chiller plant. Use this to trace the "
            "root cause of a fault back to its source."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "The ID of the node to find upstream connections for.",
                },
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "get_node_sensors",
        "description": (
            "Get the list of sensors associated with a given component node. "
            "Returns sensor names and their types (e.g., temperature, pressure, "
            "flow, status). Use this to understand what measurements are available "
            "for diagnosing a specific component."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "The ID of the component node to get sensors for.",
                },
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "get_related_systems",
        "description": (
            "Get all systems that are connected to a given system via cross-system "
            "links (e.g., chilled water or hot water connections). Returns both "
            "upstream and downstream connected systems with the connection medium."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "system_id": {
                    "type": "string",
                    "description": (
                        "The system identifier (e.g., 'chiller_plant', 'sdahu', 'boiler_plant')."
                    ),
                },
            },
            "required": ["system_id"],
        },
    },
]


# ============================================================================
# Tool Execution Class
# ============================================================================

class TopologyToolExecutor:
    """
    Executes topology tool calls using a TopologyBuilder instance.

    This class bridges the agent's tool call JSON and the actual topology
    graph queries, returning structured results.
    """

    def __init__(self, topology_builder):
        """
        Args:
            topology_builder: An initialized TopologyBuilder with a built graph.
        """
        self.tb = topology_builder
        self._tool_map = {
            "get_system_overview": self._get_system_overview,
            "get_node_children": self._get_node_children,
            "get_downstream_nodes": self._get_downstream_nodes,
            "get_upstream_nodes": self._get_upstream_nodes,
            "get_node_sensors": self._get_node_sensors,
            "get_related_systems": self._get_related_systems,
        }

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a topology tool call.

        Args:
            tool_name: Name of the tool to execute.
            arguments: Dict of arguments for the tool.

        Returns:
            Dict with the tool result, including a 'status' key.
        """
        if tool_name not in self._tool_map:
            return {
                "status": "error",
                "error": f"Unknown topology tool: {tool_name}",
            }
        try:
            return self._tool_map[tool_name](arguments)
        except Exception as e:
            logger.error(f"Tool execution error ({tool_name}): {e}")
            return {"status": "error", "error": str(e)}

    def _get_system_overview(self, args: Dict) -> Dict:
        systems = self.tb.get_systems()
        result = []
        for sys in systems:
            result.append({
                "system_id": sys.get("system_id", ""),
                "name": sys.get("name", ""),
                "type": sys.get("system_type", ""),
                "description": sys.get("description", ""),
                "node_id": sys.get("node_id", ""),
            })
        return {"status": "success", "systems": result, "count": len(result)}

    def _get_node_children(self, args: Dict) -> Dict:
        node_id = args.get("node_id", "")
        if not node_id:
            return {"status": "error", "error": "node_id is required"}

        children = self.tb.get_children(node_id)
        result = []
        for ch in children:
            result.append({
                "node_id": ch.get("node_id", ""),
                "name": ch.get("name", ""),
                "type": ch.get("brick_class", ch.get("node_type", "")),
                "level": ch.get("level", -1),
            })
        return {"status": "success", "parent_node": node_id, "children": result}

    def _get_downstream_nodes(self, args: Dict) -> Dict:
        node_id = args.get("node_id", "")
        if not node_id:
            return {"status": "error", "error": "node_id is required"}

        downstream = self.tb.get_downstream_nodes(node_id)
        result = []
        for d in downstream:
            result.append({
                "node_id": d.get("node_id", ""),
                "name": d.get("name", ""),
                "relation": d.get("relation", ""),
                "medium": d.get("medium", ""),
                "system_id": d.get("system_id", ""),
            })
        return {"status": "success", "source_node": node_id, "downstream": result}

    def _get_upstream_nodes(self, args: Dict) -> Dict:
        node_id = args.get("node_id", "")
        if not node_id:
            return {"status": "error", "error": "node_id is required"}

        upstream = self.tb.get_upstream_nodes(node_id)
        result = []
        for u in upstream:
            result.append({
                "node_id": u.get("node_id", ""),
                "name": u.get("name", ""),
                "relation": u.get("relation", ""),
                "medium": u.get("medium", ""),
                "system_id": u.get("system_id", ""),
            })
        return {"status": "success", "target_node": node_id, "upstream": result}

    def _get_node_sensors(self, args: Dict) -> Dict:
        node_id = args.get("node_id", "")
        if not node_id:
            return {"status": "error", "error": "node_id is required"}

        sensors = self.tb.get_sensors(node_id)
        result = []
        for s in sensors:
            result.append({
                "sensor_id": s.get("node_id", ""),
                "name": s.get("name", ""),
                "type": s.get("brick_class", ""),
            })
        return {"status": "success", "component_node": node_id, "sensors": result}

    def _get_related_systems(self, args: Dict) -> Dict:
        system_id = args.get("system_id", "")
        if not system_id:
            return {"status": "error", "error": "system_id is required"}

        system_node = f"system::{system_id}"
        if system_node not in self.tb.graph:
            return {"status": "error", "error": f"System '{system_id}' not found"}

        # Find all components in this system that have cross-system edges
        components = self.tb.get_system_components(system_id)

        upstream_systems = []
        downstream_systems = []
        for comp in components:
            comp_id = comp["node_id"]
            # Check outgoing cross-system edges
            for _, target, data in self.tb.graph.out_edges(comp_id, data=True):
                if data.get("relation") == "cross_system":
                    target_sys = self.tb.graph.nodes[target].get("system_id", "")
                    downstream_systems.append({
                        "system_id": target_sys,
                        "connected_via": comp_id,
                        "target_component": target,
                        "medium": data.get("medium", ""),
                    })
            # Check incoming cross-system edges
            for source, _, data in self.tb.graph.in_edges(comp_id, data=True):
                if data.get("relation") == "cross_system":
                    source_sys = self.tb.graph.nodes[source].get("system_id", "")
                    upstream_systems.append({
                        "system_id": source_sys,
                        "connected_via": source,
                        "target_component": comp_id,
                        "medium": data.get("medium", ""),
                    })

        return {
            "status": "success",
            "system_id": system_id,
            "upstream_connections": upstream_systems,
            "downstream_connections": downstream_systems,
        }
