"""
Tool Executor — Unified tool call parser and router.

Provides a single entry point for executing any agent tool call,
routing between topology tools and prediction tools.
"""

import json
from typing import Any, Dict, Optional

from src.topology.topology_tools import TopologyToolExecutor, TOPOLOGY_TOOL_SCHEMAS
from src.node_models.prediction_tools import PredictionToolExecutor, PREDICTION_TOOL_SCHEMAS
from src.utils.io_utils import setup_logger

logger = setup_logger(__name__)

# All available tool schemas (for system prompts)
ALL_TOOL_SCHEMAS = TOPOLOGY_TOOL_SCHEMAS + PREDICTION_TOOL_SCHEMAS


class UnifiedToolExecutor:
    """
    Unified tool execution interface routing to topology or prediction tools.
    """

    def __init__(
        self,
        topo_executor: TopologyToolExecutor,
        pred_executor: PredictionToolExecutor,
    ):
        self.topo_executor = topo_executor
        self.pred_executor = pred_executor

        # Wire health_provider: give topology access to prediction for anomaly scores
        self.topo_executor._health_provider = self.pred_executor

        # Build unified tool name → executor mapping
        self._tool_router: Dict[str, Any] = {}
        for schema in TOPOLOGY_TOOL_SCHEMAS:
            self._tool_router[schema["name"]] = self.topo_executor
        for schema in PREDICTION_TOOL_SCHEMAS:
            self._tool_router[schema["name"]] = self.pred_executor

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call by routing to the correct executor."""
        executor = self._tool_router.get(tool_name)
        if executor is None:
            return {
                "status": "error",
                "error": (
                    f"Unknown tool '{tool_name}'. Available tools: "
                    f"{list(self._tool_router.keys())}"
                ),
            }
        return executor.execute(tool_name, arguments)

    def set_scenario_state(self, state) -> None:
        """Set the current scenario state on the prediction executor."""
        self.pred_executor.set_scenario_state(state)

    def get_tool_schemas(self) -> list:
        """Return all tool schemas for the system prompt."""
        return ALL_TOOL_SCHEMAS

    def get_tool_schemas_str(self) -> str:
        """Return tool schemas as a formatted JSON string for prompts."""
        return json.dumps(ALL_TOOL_SCHEMAS, indent=2)

    def get_available_tools(self) -> list:
        """Return list of available tool names."""
        return list(self._tool_router.keys())

