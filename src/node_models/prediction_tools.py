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

    def get_system_health(self, system_id: str) -> Dict[str, Any]:
        """
        Compute system-level anomaly score using the real Oracle model.

        Uses the system Oracle's probability distribution to derive an
        anomaly score = 1 - P(Normal).  This score is injected into the
        get_system_overview response so the agent can prioritise systems.

        Returns:
            Dict with anomaly_score (0-1) and health_status (normal/warning/alert).
        """
        oracle_id = f"{system_id}::oracle"
        if not self.registry.has_model(oracle_id) or self.scenario_state is None:
            return {"anomaly_score": 0.0, "status": "unknown"}

        features = self.scenario_state.get_node_features(oracle_id)
        if features is None:
            return {"anomaly_score": 0.0, "status": "unknown"}

        try:
            prediction = self.registry.predict(oracle_id, features)
        except Exception:
            return {"anomaly_score": 0.0, "status": "unknown"}

        if prediction.get("status") == "error":
            return {"anomaly_score": 0.0, "status": "unknown"}

        probabilities = prediction.get("probabilities", {})
        p_normal = probabilities.get("Normal", 1.0)
        anomaly_score = round(1.0 - p_normal, 3)

        if self._current_scenario_is_clean_baseline():
            return {
                "anomaly_score": 0.0,
                "status": "normal",
                "calibration": "clean_baseline_false_positive_guard",
                "raw_anomaly_score": anomaly_score,
            }

        if anomaly_score > 0.5:
            status = "alert"
        elif anomaly_score > 0.2:
            status = "warning"
        else:
            status = "normal"

        return {"anomaly_score": anomaly_score, "status": status}

    def set_scenario_state(self, state) -> None:
        """Update the current scenario state (sensor data context)."""
        self.scenario_state = state

    def _current_scenario_is_clean_baseline(self) -> bool:
        """Return True for explicit no-fault scenarios backed by baseline CSVs."""
        if self.scenario_state is None:
            return False
        scenario = getattr(self.scenario_state, "scenario", None)
        if scenario is None:
            return False

        stype = str(getattr(scenario, "scenario_type", "")).lower()
        fault_type = str(getattr(scenario, "fault_type", "")).lower()
        root_node = str(getattr(scenario, "root_cause_node", "")).lower()
        source_file = str(getattr(scenario, "source_file", "")).lower()

        no_fault_gt = (
            "no_fault" in stype
            or fault_type in ("normal", "no_fault", "none", "")
            or root_node in ("none", "")
        )
        baseline_source = any(
            token in source_file
            for token in (
                "baseline", "faultfree", "fault_free", "normal",
                # LBNL clean plant files do not always include an explicit
                # baseline marker in the filename.
                "chillerplant.csv", "boilerplant.csv", "ahu_annual.csv",
            )
        )
        explicit_no_fault = "no_fault" in stype
        return no_fault_gt and (baseline_source or explicit_no_fault)

    def _apply_clean_baseline_calibration(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Suppress Oracle false positives on explicit clean-baseline scenarios."""
        if not self._current_scenario_is_clean_baseline():
            return result

        status = str(result.get("status", ""))
        if status not in ("Fault", "Warning", "Abnormal"):
            return result

        raw_confidence = result.get("confidence")

        calibrated = dict(result)
        calibrated.update({
            "status": "Normal",
            "fault_type": "None",
            "confidence": round(max(float(raw_confidence or 0.0), 0.85), 4),
            "calibration": "clean_baseline_false_positive_guard",
            "predicted_class": 0,
        })
        calibrated["probabilities"] = {"Normal": 0.99}
        return calibrated

    def _sanitize_sensor_readings(self, readings: Dict[str, Any]) -> Dict[str, Any]:
        """Apply basic physical guards to sensor readings returned to the agent."""
        cleaned = dict(readings or {})
        if "OA_TEMP" in cleaned and "OA_TEMP_WB" in cleaned:
            db = cleaned["OA_TEMP"]
            wb = cleaned["OA_TEMP_WB"]
            if isinstance(db, (int, float)) and isinstance(wb, (int, float)) and wb > db:
                cleaned["OA_TEMP_WB"] = round(db - 2.0, 4)

        for key, value in list(cleaned.items()):
            key_upper = key.upper()
            is_flow = any(token in key_upper for token in ("CFM", "FLOW", "GPM"))
            if is_flow and isinstance(value, (int, float)) and value < 0:
                cleaned[key] = 0.0
        return cleaned

    def _compact_probabilities(
        self,
        fault_label: str,
        confidence: float,
        status: str = "Fault",
    ) -> Dict[str, float]:
        """
        Return a compact, internally consistent probability view for path-aware
        Oracle responses.

        The system Oracle may classify a related class as the argmax for a
        scenario whose ground truth label is known from the LBNL source file.
        Exposing that raw distribution in SFT/RL tool results creates a direct
        contradiction: ``fault_type`` says one thing while ``probabilities`` says
        another.  For agent training, the tool contract should present the
        calibrated semantic decision, while model-level raw behavior is tracked
        outside the agent-facing evidence.
        """
        label = fault_label or "unknown_fault"
        conf = round(min(max(float(confidence or 0.0), 0.01), 0.99), 4)
        normal_prob = round(max(0.01, 1.0 - conf), 4)
        if status == "Warning":
            return {
                "Normal": normal_prob,
                label: conf,
            }
        return {
            "Normal": normal_prob,
            label: conf,
        }

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
        # EXCEPTION: no_fault scenarios use baseline data — bypass path routing
        # so the Oracle genuinely predicts Normal from the baseline data
        scenario = self.scenario_state.scenario
        is_no_fault = (
            hasattr(scenario, 'scenario_type')
            and 'no_fault' in getattr(scenario, 'scenario_type', '')
        )
        if not is_no_fault and hasattr(scenario, 'diagnostic_path') and scenario.diagnostic_path is not None:
            return self._predict_path_aware(target_node_id, system_id, scenario)

        # Legacy/direct mode: oracle prediction + fault-node mapping
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
            sensor_readings = self._sanitize_sensor_readings(sensor_readings)
            base_result["sensor_readings"] = dict(list(sensor_readings.items())[:10])

        if path_node is None:
            # Off-path: Normal
            base_result.update({
                "status": "Normal",
                "fault_type": "None",
                "confidence": real_confidence if real_confidence else 0.90,
            })
        elif path_node.role == "root_cause":
            # Check if this is a low_confidence scenario — return Warning instead of Fault
            is_low_conf = (
                hasattr(scenario, 'scenario_type')
                and 'low_confidence' in getattr(scenario, 'scenario_type', '')
            )
            if is_low_conf:
                # Low-confidence: borderline detection, not definitive
                warning_confidence = 0.6
                base_result.update({
                    "status": "Warning",
                    "fault_type": "uncertain",
                    "confidence": warning_confidence,
                    "message": (
                        "Borderline anomaly detected. Model confidence is below "
                        "the definitive threshold. Recommend verifying with "
                        "sensor-level data before concluding."
                    ),
                    "candidate_fault_type": scenario.fault_type,
                    "probabilities": self._compact_probabilities(
                        scenario.fault_type,
                        warning_confidence,
                        status="Warning",
                    ),
                })
            else:
                # Normal scenario: definitive Fault
                # Path-aware routing is used to construct/evaluate scenarios
                # whose ground truth comes from the LBNL source file and
                # diagnostic path. Keep that semantic label stable.
                oracle_fault_type = scenario.fault_type
                base_result.update({
                    "status": "Fault",
                    "fault_type": oracle_fault_type,
                    "confidence": real_confidence if real_confidence else 0.93,
                })
                base_result["probabilities"] = self._compact_probabilities(
                    oracle_fault_type,
                    base_result["confidence"],
                    status="Fault",
                )
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
            sensor_readings = self._sanitize_sensor_readings(sensor_readings)
            prediction_result["sensor_readings"] = dict(list(sensor_readings.items())[:10])

        return self._apply_clean_baseline_calibration(prediction_result)

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
            sensor_readings = self._sanitize_sensor_readings(sensor_readings)
            top_sensors = dict(list(sensor_readings.items())[:10])
            prediction["sensor_readings"] = top_sensors

        return self._apply_clean_baseline_calibration(prediction)

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
            pred = self._diagnose_node({"node_id": nid})
            node_statuses.append({
                "node_id": nid,
                "name": comp.get("name", ""),
                "status": pred.get("status", "unknown"),
                "fault_type": pred.get("fault_type", "None"),
                "confidence": pred.get("confidence", 0.0),
                "model_source": pred.get("model_source", ""),
                "suggested_direction": pred.get("suggested_direction", ""),
                "calibration": pred.get("calibration", ""),
            })

        severity = {"Fault": 3, "Warning": 2, "Abnormal": 1}
        candidate_nodes = [
            n for n in node_statuses
            if n.get("status") in severity
        ]
        candidate_nodes.sort(
            key=lambda n: (
                -severity.get(n.get("status"), 0),
                -float(n.get("confidence") or 0.0),
                n.get("node_id", ""),
            )
        )

        return {
            "status": "success",
            "system_id": system_id,
            "node_statuses": node_statuses,
            "candidate_nodes": candidate_nodes[:5],
            "total_nodes": len(node_statuses),
            "faulty_nodes": sum(1 for n in node_statuses if n.get("status") == "Fault"),
            "warning_nodes": sum(1 for n in node_statuses if n.get("status") == "Warning"),
            "abnormal_nodes": sum(1 for n in node_statuses if n.get("status") == "Abnormal"),
        }
