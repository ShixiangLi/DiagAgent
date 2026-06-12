"""
Tool Executor — Unified tool call parser and router.

Provides a single entry point for executing any agent tool call,
routing between topology tools and prediction tools.
"""

import json
from typing import Any, Dict

from src.topology.topology_tools import TopologyToolExecutor, TOPOLOGY_TOOL_SCHEMAS
from src.node_models.prediction_tools import (
    AGENT_PREDICTION_TOOL_SCHEMAS,
    PredictionToolExecutor,
    PREDICTION_TOOL_SCHEMAS,
)
from src.utils.io_utils import setup_logger

logger = setup_logger(__name__)

STATUS_SUMMARY_TOOL = "get_node_status_summary"


def build_tool_schemas(expose_status_summary: bool = False) -> list:
    """Return tool schemas for the active tool-exposure policy."""
    prediction_schemas = (
        PREDICTION_TOOL_SCHEMAS
        if expose_status_summary
        else AGENT_PREDICTION_TOOL_SCHEMAS
    )
    return TOPOLOGY_TOOL_SCHEMAS + prediction_schemas


# Default agent-visible tool schemas for system prompts.
ALL_TOOL_SCHEMAS = build_tool_schemas(expose_status_summary=False)


class UnifiedToolExecutor:
    """
    Unified tool execution interface routing to topology or prediction tools.
    """

    def __init__(
        self,
        topo_executor: TopologyToolExecutor,
        pred_executor: PredictionToolExecutor,
        include_system_health: bool = True,
        expose_status_summary: bool = False,
    ):
        self.topo_executor = topo_executor
        self.pred_executor = pred_executor
        self.include_system_health = bool(include_system_health)
        self.expose_status_summary = bool(expose_status_summary)
        self._tool_schemas = build_tool_schemas(self.expose_status_summary)

        # Optionally give topology access to prediction for anomaly scores.
        self._bind_topology_providers()

        # Build unified tool name → executor mapping
        self._tool_router: Dict[str, Any] = {}
        for schema in TOPOLOGY_TOOL_SCHEMAS:
            self._tool_router[schema["name"]] = self.topo_executor
        for schema in PREDICTION_TOOL_SCHEMAS:
            if (
                schema.get("name") == STATUS_SUMMARY_TOOL
                and not self.expose_status_summary
            ):
                continue
            self._tool_router[schema["name"]] = self.pred_executor

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool call by routing to the correct executor."""
        executor = self._tool_router.get(tool_name)
        if executor is None:
            if tool_name == STATUS_SUMMARY_TOOL and not self.expose_status_summary:
                return {
                    "status": "error",
                    "error": (
                        "Tool 'get_node_status_summary' is disabled in the "
                        "main topology-diagnosis experiment. Use topology "
                        "navigation tools and diagnose_node instead."
                    ),
                }
            return {
                "status": "error",
                "error": (
                    f"Unknown tool '{tool_name}'. Available tools: "
                    f"{list(self._tool_router.keys())}"
                ),
            }
        if executor is self.topo_executor:
            self._bind_topology_providers()
        return executor.execute(tool_name, arguments)

    def set_scenario_state(self, state) -> None:
        """Set the current scenario state on the prediction executor."""
        self.pred_executor.set_scenario_state(state)
        self._bind_topology_providers()

    def _bind_topology_providers(self) -> None:
        """Bind the shared topology executor to this unified executor.

        Some generation scripts keep separate agent-visible and recoverability
        executors that share one TopologyToolExecutor.  Rebinding before
        topology calls prevents get_node_sensors from reading a stale provider
        or another executor's scenario state.
        """
        self.topo_executor._health_provider = (
            self.pred_executor if self.include_system_health else None
        )
        # Always allow get_node_sensors to attach current scenario readings.
        # This does not expose hidden system health or root-cause labels; it
        # only makes the sensor-verification tool useful in the active episode.
        self.topo_executor._sensor_provider = self.pred_executor

    def get_tool_schemas(self) -> list:
        """Return all tool schemas for the system prompt."""
        return self._tool_schemas

    def get_tool_schemas_str(self) -> str:
        """Return tool schemas as a formatted JSON string for prompts."""
        return json.dumps(self._tool_schemas, indent=2)

    def get_available_tools(self) -> list:
        """Return list of available tool names."""
        return list(self._tool_router.keys())
