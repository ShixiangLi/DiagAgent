"""
Prediction Tools — Agent-callable tool definitions for node fault diagnosis.

Defines the JSON tool schemas the LLM agent uses to diagnose node status
and wraps the ModelRegistry prediction interface.
"""

from typing import Any, Dict, List, Optional

from src.utils.io_utils import setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Tool Schema Definitions
# ============================================================================

PREDICTION_TOOL_SCHEMAS = [
    {
        "name": "diagnose_node",
        "description": (
            "Diagnose the current status of a specific component node by running "
            "its fault prediction model on the current sensor data. Returns the "
            "predicted status (Normal, Abnormal, or Fault), fault type if faulty, "
            "confidence score, and relevant sensor readings. An 'Abnormal' status "
            "indicates the node shows anomalous readings but is not the root cause; "
            "check the 'suggested_direction' field to trace further. Use this to "
            "check whether a specific component is operating normally, showing "
            "symptoms, or experiencing a definitive fault."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": (
                        "The ID of the component node to diagnose. "
                        "Format: 'system_id::ComponentName' "
                        "(e.g., 'chiller_plant::Chiller_1')."
                    ),
                },
            },
            "required": ["node_id"],
        },
    },
    {
        "name": "get_node_status_summary",
        "description": (
            "Get a summary of the diagnostic status of all component nodes "
            "within a specific system. Returns each node's status and confidence. "
            "Use this for a quick overview of which components in a system might "
            "be faulty before drilling into specific nodes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "system_id": {
                    "type": "string",
                    "description": (
                        "The system identifier to scan "
                        "(e.g., 'chiller_plant', 'sdahu', 'boiler_plant')."
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

class PredictionToolExecutor:
    """
    Executes prediction tool calls using the ModelRegistry and environment state.

    This class bridges the agent's tool call JSON and the actual model predictions,
    using the current fault scenario's sensor data.
    """

    def __init__(self, model_registry, topology_builder, scenario_state=None):
        """
        Args:
            model_registry: Initialized ModelRegistry with loaded models.
            topology_builder: TopologyBuilder with the built graph.
            scenario_state: Current FaultScenarioState providing sensor data.
        """
        self.registry = model_registry
        self.topology = topology_builder
        self.scenario_state = scenario_state
        self._tool_map = {
            "diagnose_node": self._diagnose_node,
            "get_node_status_summary": self._get_node_status_summary,
        }

    def set_scenario_state(self, state) -> None:
        """Update the current scenario state (sensor data context)."""
        self.scenario_state = state

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a prediction tool call."""
        if tool_name not in self._tool_map:
            return {"status": "error", "error": f"Unknown prediction tool: {tool_name}"}
        try:
            return self._tool_map[tool_name](arguments)
        except Exception as e:
            logger.error(f"Prediction tool error ({tool_name}): {e}")
            return {"status": "error", "error": str(e)}

    def _diagnose_node(self, args: Dict) -> Dict:
        node_id = args.get("node_id", "")
        if not node_id:
            return {"status": "error", "error": "node_id is required"}

        # Extract system_id from node_id (format: "system_id::ComponentName")
        parts = node_id.split("::")
        system_id = parts[0] if len(parts) >= 2 else ""

        # Strategy: prefer system oracle → fallback to per-node model
        oracle_id = f"{system_id}::oracle" if system_id else ""

        # Try system oracle first (much higher accuracy)
        if oracle_id and self.registry.has_model(oracle_id):
            prediction = self._predict_with_oracle(oracle_id, node_id, system_id)
            if prediction is not None:
                return prediction

        # Fallback to per-node model
        if self.registry.has_model(node_id):
            return self._predict_with_node_model(node_id)

        # No model at all
        node_info = self.topology.get_node_info(node_id)
        if node_info is None:
            return {
                "status": "error",
                "error": f"Node '{node_id}' not found in topology",
            }
        return {
            "status": "unknown",
            "message": (
                f"No diagnostic model available for node '{node_id}' "
                f"(type: {node_info.get('brick_class', 'unknown')}). "
                "Try diagnosing its parent component or checking its sensors."
            ),
            "node_info": {
                "name": node_info.get("name", ""),
                "type": node_info.get("brick_class", ""),
                "system": node_info.get("system_id", ""),
            },
        }

    def _predict_with_oracle(self, oracle_id: str, target_node_id: str, system_id: str) -> Optional[Dict]:
        """
        Use the system oracle to diagnose a specific node.

        Supports two modes:
          1. Path-aware mode: If the scenario has a diagnostic_path, return
             Normal/Abnormal/Fault based on the node's role in the path.
          2. Legacy mode: Use oracle prediction + fault-node mapping.
        """
        if self.scenario_state is None:
            return None

        # Path-aware mode: use diagnostic path for routing
        scenario = self.scenario_state.scenario
        if hasattr(scenario, 'diagnostic_path') and scenario.diagnostic_path is not None:
            return self._predict_path_aware(target_node_id, system_id, scenario)

        # Legacy mode: oracle prediction + fault-node mapping
        return self._predict_oracle_legacy(oracle_id, target_node_id, system_id)

    def _predict_path_aware(
        self, target_node_id: str, system_id: str, scenario
    ) -> Dict:
        """
        Path-aware prediction: uses diagnostic path for status routing,
        but calls the real Oracle model for confidence/probability scores.

        - Off-path → Normal (with real Oracle confidence if available)
        - Symptom/Intermediate → Abnormal with directional hint
        - Root cause → Fault with specific fault type and real confidence
        """
        path = scenario.diagnostic_path
        path_node = path.get_path_node(target_node_id)
        node_info = self.topology.get_node_info(target_node_id)
        node_name = node_info.get("name", "") if node_info else ""

        # Try to get real Oracle prediction for confidence scores
        oracle_id = f"{system_id}::oracle"
        real_confidence = None
        real_probabilities = None
        if self.scenario_state is not None:
            features = self.scenario_state.get_node_features(oracle_id)
            if features is not None and self.registry.has_model(oracle_id):
                try:
                    prediction = self.registry.predict(oracle_id, features)
                    if prediction.get("status") != "error":
                        real_confidence = round(prediction.get("confidence", 0.85), 4)
                        real_probabilities = prediction.get("probabilities")
                except Exception:
                    pass

        base_result = {
            "node_id": target_node_id,
            "node_name": node_name,
            "system_id": system_id,
            "model_source": "system_oracle",
        }

        # Add real sensor readings from scenario state
        if self.scenario_state is not None:
            sensor_readings = self.scenario_state.get_sensor_readings(target_node_id)
            base_result["sensor_readings"] = dict(list(sensor_readings.items())[:10])

        if path_node is None:
            # Off-path: Normal
            base_result.update({
                "status": "Normal",
                "fault_type": "None",
                "confidence": real_confidence if real_confidence else 0.90,
            })
        elif path_node.role == "root_cause":
            # Root cause: definitive Fault
            base_result.update({
                "status": "Fault",
                "fault_type": scenario.fault_type,
                "confidence": real_confidence if real_confidence else 0.93,
            })
            if real_probabilities:
                base_result["probabilities"] = real_probabilities
        else:
            # Symptom or intermediate: Abnormal with hint
            base_result.update({
                "status": "Abnormal",
                "fault_type": "None",
                "confidence": real_confidence if real_confidence else 0.78,
                "abnormal_indicators": path_node.abnormal_hint,
                "suggested_direction": path_node.direction,
            })

        return base_result

    def _predict_oracle_legacy(
        self, oracle_id: str, target_node_id: str, system_id: str
    ) -> Optional[Dict]:
        """Legacy oracle prediction using fault-node mapping."""
        features = self.scenario_state.get_node_features(oracle_id)
        if features is None:
            features = self.scenario_state.get_node_features(f"{system_id}::oracle")

        if features is None:
            return None

        prediction = self.registry.predict(oracle_id, features)
        if prediction.get("status") == "error":
            return None

        # Load fault-node mapping
        oracle_meta = self.registry.get_metadata(oracle_id)
        fault_node_map = {}
        if oracle_meta:
            import os
            map_path = os.path.join(
                os.path.dirname(oracle_meta.get("model_path", "")),
                "fault_node_map.json"
            )
            if os.path.exists(map_path):
                from src.utils.io_utils import load_json
                fault_node_map = load_json(map_path)

        fault_type = prediction.get("fault_type", "None")
        responsible_node = fault_node_map.get(fault_type, target_node_id)

        node_info = self.topology.get_node_info(target_node_id)

        if prediction["status"] == "Normal" or responsible_node != target_node_id:
            prediction_result = {
                "status": "Normal",
                "fault_type": "None",
                "confidence": round(prediction.get("confidence", 0.85), 4),
                "predicted_class": 0,
            }
        else:
            prediction_result = {
                "status": "Fault",
                "fault_type": fault_type,
                "confidence": round(prediction.get("confidence", 0.85), 4),
                "predicted_class": prediction.get("predicted_class", 1),
                "probabilities": prediction.get("probabilities", {}),
            }

        prediction_result["node_id"] = target_node_id
        prediction_result["node_name"] = node_info.get("name", "") if node_info else ""
        prediction_result["system_id"] = system_id
        prediction_result["model_source"] = "system_oracle"

        if self.scenario_state is not None:
            sensor_readings = self.scenario_state.get_sensor_readings(target_node_id)
            prediction_result["sensor_readings"] = dict(list(sensor_readings.items())[:10])

        return prediction_result

    def _predict_with_node_model(self, node_id: str) -> Dict:
        """Predict using the per-node model (fallback)."""
        if self.scenario_state is None:
            return {
                "status": "error",
                "error": "No scenario state available for prediction",
            }

        features = self.scenario_state.get_node_features(node_id)
        if features is not None:
            prediction = self.registry.predict(node_id, features)
        else:
            sensor_data = self.scenario_state.get_sensor_readings(node_id)
            prediction = self.registry.predict_from_sensor_data(node_id, sensor_data)

        # Enrich with node info
        node_info = self.topology.get_node_info(node_id)
        prediction["node_id"] = node_id
        prediction["node_name"] = node_info.get("name", "") if node_info else ""
        prediction["system_id"] = node_info.get("system_id", "") if node_info else ""
        prediction["model_source"] = "per_node"

        # Add sensor readings summary
        if self.scenario_state is not None:
            sensor_readings = self.scenario_state.get_sensor_readings(node_id)
            top_sensors = dict(list(sensor_readings.items())[:10])
            prediction["sensor_readings"] = top_sensors

        return prediction

    def _get_node_status_summary(self, args: Dict) -> Dict:
        system_id = args.get("system_id", "")
        if not system_id:
            return {"status": "error", "error": "system_id is required"}

        components = self.topology.get_system_components(system_id)
        if not components:
            return {
                "status": "error",
                "error": f"No components found for system '{system_id}'",
            }

        node_statuses = []
        for comp in components:
            nid = comp["node_id"]
            if self.registry.has_model(nid) and self.scenario_state is not None:
                features = self.scenario_state.get_node_features(nid)
                if features is not None:
                    pred = self.registry.predict(nid, features)
                    node_statuses.append({
                        "node_id": nid,
                        "name": comp.get("name", ""),
                        "status": pred.get("status", "unknown"),
                        "fault_type": pred.get("fault_type", "None"),
                        "confidence": pred.get("confidence", 0.0),
                    })
                else:
                    node_statuses.append({
                        "node_id": nid,
                        "name": comp.get("name", ""),
                        "status": "no_data",
                    })
            else:
                node_statuses.append({
                    "node_id": nid,
                    "name": comp.get("name", ""),
                    "status": "no_model",
                })

        return {
            "status": "success",
            "system_id": system_id,
            "node_statuses": node_statuses,
            "total_nodes": len(node_statuses),
            "faulty_nodes": sum(1 for n in node_statuses if n.get("status") == "Fault"),
        }
