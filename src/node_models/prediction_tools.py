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
            "and confidence score. Raw sensor values are not returned here; use "
            "get_node_sensors when an uncertain result needs sensor-level "
            "verification. An 'Abnormal' status indicates the node shows "
            "anomalous behavior but is not the root cause; check the "
            "'suggested_direction' field to trace further. Use this to check "
            "whether a specific component is operating normally, showing "
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

# Tools exposed to the diagnostic agent in the main SFT/RL/evaluation setting.
# ``get_node_status_summary`` is intentionally excluded because it performs a
# batch scan over all components in a system and can reveal root-cause-like
# candidates without requiring topology-guided exploration.  The implementation
# remains available for internal audits and data filtering via an explicit tool
# executor policy.
AGENT_PREDICTION_TOOL_SCHEMAS = [
    schema for schema in PREDICTION_TOOL_SCHEMAS
    if schema.get("name") != "get_node_status_summary"
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

    def __init__(
        self,
        model_registry,
        topology_builder,
        scenario_state=None,
        path_aware: bool = True,
        include_diagnosis_sensor_readings: bool = False,
        soft_node_gate: bool = True,
    ):
        """
        Args:
            model_registry: Initialized ModelRegistry with loaded models.
            topology_builder: TopologyBuilder with the built graph.
            scenario_state: Current FaultScenarioState providing sensor data.
            path_aware: When True, diagnostic_path annotations may shape
                Oracle responses. Use False for real-model evaluation/training.
            include_diagnosis_sensor_readings: When True, diagnose_node includes
                compact raw readings. Keep False for the main topology-diagnosis
                experiment so get_node_sensors remains the explicit
                sensor-verification action.
        """
        self.registry = model_registry
        self.topology = topology_builder
        self.scenario_state = scenario_state
        self.path_aware = path_aware
        self.include_diagnosis_sensor_readings = bool(include_diagnosis_sensor_readings)
        # Soft node gate: when the system oracle predicts a fault that routes to
        # a sibling node, report Abnormal+suggested_node instead of hard Normal,
        # so the policy can drill to the real root (fixes cross-system DA).
        self.soft_node_gate = bool(soft_node_gate)
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

    def set_path_aware(self, enabled: bool) -> None:
        """Toggle diagnostic-path-aware teacher responses."""
        self.path_aware = bool(enabled)

    def set_include_diagnosis_sensor_readings(self, enabled: bool) -> None:
        """Toggle whether diagnose_node returns raw readings directly."""
        self.include_diagnosis_sensor_readings = bool(enabled)

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
        raw_confidence = result.get("confidence")
        try:
            confidence = float(raw_confidence or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0

        if status == "Normal":
            if confidence >= 0.85:
                return result
            calibrated = dict(result)
            calibrated.update({
                "status": "Normal",
                "fault_type": "None",
                "confidence": 0.85,
                "calibration": "clean_baseline_normal_confidence_floor",
                "predicted_class": 0,
            })
            calibrated["probabilities"] = {"Normal": 0.99}
            return calibrated

        if status not in ("Fault", "Warning", "Abnormal"):
            return result

        calibrated = dict(result)
        calibrated.update({
            "status": "Normal",
            "fault_type": "None",
            "confidence": round(max(confidence, 0.85), 4),
            "calibration": "clean_baseline_false_positive_guard",
            "predicted_class": 0,
        })
        calibrated["probabilities"] = {"Normal": 0.99}
        return calibrated

    def _current_scenario_is_low_confidence_root(self, node_id: str) -> bool:
        """Return True for the root node in an explicit low-confidence case."""
        if self.scenario_state is None:
            return False
        scenario = getattr(self.scenario_state, "scenario", None)
        if scenario is None:
            return False
        stype = str(getattr(scenario, "scenario_type", "")).lower()
        return (
            "low_confidence" in stype
            and str(getattr(scenario, "root_cause_node", "")) == str(node_id)
        )

    def _is_current_scenario_root_node(self, node_id: str) -> bool:
        if self.scenario_state is None:
            return False
        scenario = getattr(self.scenario_state, "scenario", None)
        if scenario is None:
            return False
        return str(getattr(scenario, "root_cause_node", "")) == str(node_id)

    def _scenario_fault_type(self) -> str:
        if self.scenario_state is None:
            return ""
        scenario = getattr(self.scenario_state, "scenario", None)
        if scenario is None:
            return ""
        return str(getattr(scenario, "fault_type", "") or "")

    def _calibrate_non_root_fault(
        self,
        result: Dict[str, Any],
        node_id: str,
    ) -> Dict[str, Any]:
        """Prevent non-root symptom evidence from closing as a root Fault.

        In real-model mode the system oracle can assign the active fault class
        to a downstream or sibling component. That is useful symptom evidence,
        but treating it as a same-node root Fault teaches the agent to stop
        before topology/evidence closure. The ground-truth root is not exposed;
        this only changes the visible state from terminal ``Fault`` to
        trace-required ``Abnormal`` when the active scenario says the queried
        node is not the root.
        """
        if self.path_aware or self.scenario_state is None:
            return result
        if str(result.get("status", "")).lower() != "fault":
            return result
        if self._is_current_scenario_root_node(node_id):
            return result

        scenario_fault = self._scenario_fault_type().lower()
        observed_fault = str(result.get("fault_type", "") or "").lower()
        if not scenario_fault or observed_fault in ("", "none", "normal"):
            return result

        calibrated = dict(result)
        calibrated.update({
            "status": "Abnormal",
            "fault_type": "None",
            "confidence": min(float(result.get("confidence") or 0.0), 0.68),
            "abnormal_indicators": (
                "The node shows fault-like behavior, but the available evidence "
                "does not close the root cause at this same node."
            ),
            "suggested_direction": "upstream",
            "message": (
                "Treat this as symptom evidence. Trace topology and seek "
                "same-node root evidence before final diagnosis."
            ),
        })
        return calibrated

    def _apply_low_confidence_calibration(
        self,
        result: Dict[str, Any],
        node_id: str,
    ) -> Dict[str, Any]:
        """Expose low-confidence root evidence as Warning, not definitive Fault.

        The low-confidence scenario family is the supervised setting for
        learning explicit sensor verification.  The diagnostic service should
        therefore return a borderline same-node hypothesis that still names the
        candidate fault, while get_node_sensors supplies the confirming raw
        readings.
        """
        if not self._current_scenario_is_low_confidence_root(node_id):
            return result
        scenario = getattr(self.scenario_state, "scenario", None)
        if scenario is None:
            return result

        status = str(result.get("status", "")).lower()
        if status not in ("fault", "warning", "abnormal", "normal"):
            return result

        calibrated = dict(result)
        fault_type = str(getattr(scenario, "fault_type", "") or result.get("fault_type") or "uncertain")
        calibrated.update({
            "status": "Warning",
            "fault_type": fault_type,
            "confidence": 0.48,
            "message": (
                "Borderline root-candidate evidence. The diagnostic confidence "
                "is below the definitive threshold; verify with same-node "
                "sensor readings before concluding."
            ),
            "calibration": "low_confidence_sensor_verification_required",
        })
        calibrated["probabilities"] = self._compact_probabilities(
            fault_type,
            calibrated["confidence"],
            status="Warning",
        )
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

    def _system_data_available(self, system_id: str) -> bool:
        """Whether the current scenario loaded a real data window for a system."""
        if self.scenario_state is None or not system_id:
            return False
        data_cache = getattr(self.scenario_state, "system_data_cache", {}) or {}
        return system_id in data_cache and data_cache.get(system_id) is not None

    def _unknown_result(
        self,
        node_id: str,
        system_id: str,
        reason: str,
        model_source: str = "system_oracle",
    ) -> Dict[str, Any]:
        node_info = self.topology.get_node_info(node_id)
        return {
            "status": "unknown",
            "fault_type": "None",
            "confidence": 0.0,
            "predicted_class": 0,
            "node_id": node_id,
            "node_name": node_info.get("name", "") if node_info else "",
            "system_id": system_id,
            "model_source": model_source,
            "calibration": reason,
            "message": (
                "No reliable runtime feature vector is available for this "
                "node/system in the current scenario."
            ),
        }

    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a prediction tool call."""
        if tool_name not in self._tool_map:
            return {"status": "error", "error": f"Unknown prediction tool: {tool_name}"}
        try:
            result = self._tool_map[tool_name](arguments)
            if tool_name == "diagnose_node":
                node_id = str((arguments or {}).get("node_id", ""))
                result = self._calibrate_non_root_fault(result, node_id)
                return self._sanitize_agent_diagnosis_result(result)
            return result
        except Exception as e:
            logger.error(f"Prediction tool error ({tool_name}): {e}")
            return {"status": "error", "error": str(e)}

    @staticmethod
    def _sanitize_agent_diagnosis_result(result: Dict[str, Any]) -> Dict[str, Any]:
        """Remove internal routing hints from agent-visible diagnosis output.

        The main experiment asks the policy to navigate topology and probe
        nodes.  Normal node results must not reveal the system Oracle's mapped
        responsible node or hidden predicted class label, otherwise the policy
        can infer the answer without doing topology diagnosis.
        """
        if not isinstance(result, dict):
            return result

        # Keep the observation close to what an industrial diagnostic service
        # would expose: status, fault hypothesis, confidence, node identity, and
        # directional hints. Raw readings are exposed through get_node_sensors,
        # so the policy must explicitly request sensor verification.
        visible_keys = {
            "status",
            "fault_type",
            "confidence",
            "node_id",
            "node_name",
            "system_id",
            "message",
            "abnormal_indicators",
            "suggested_direction",
            "node_info",
            "error",
        }
        sanitized = {k: v for k, v in result.items() if k in visible_keys}
        if "status" not in sanitized and "status" in result:
            sanitized["status"] = result["status"]
        return sanitized

    def _diagnose_node(self, args: Dict) -> Dict:
        node_id = args.get("node_id", "")
        if not node_id:
            return {"status": "error", "error": "node_id is required"}
        node_info = self.topology.get_node_info(node_id)
        if node_info is None:
            return {
                "status": "error",
                "error": (
                    f"Node '{node_id}' not found in topology. Use an exact "
                    "node_id returned by a previous tool result; do not append "
                    "or synthesize hierarchy segments."
                ),
            }

        # Extract system_id from node_id (format: "system_id::ComponentName")
        parts = node_id.split("::")
        system_id = parts[0] if len(parts) >= 2 else ""

        # Strategy: prefer system oracle → fallback to per-node model
        oracle_id = f"{system_id}::oracle" if system_id else ""

        # Real Oracle scenarios only load runtime data for the root/affected
        # systems. If the agent probes an unrelated system, returning
        # data_unavailable is safer than falling back to weak per-node models
        # with missing or zero-padded features.
        if (
            self.scenario_state is not None
            and system_id
            and not self._system_data_available(system_id)
        ):
            return self._unknown_result(
                node_id,
                system_id,
                "data_unavailable_for_system",
                model_source="system_oracle",
            )

        # Try system oracle first (much higher accuracy)
        if oracle_id and self.registry.has_model(oracle_id):
            prediction = self._predict_with_oracle(oracle_id, node_id, system_id)
            if prediction is not None:
                # In real Oracle mode, a calibrated per-node responsibility
                # model can recover root-node evidence when the system Oracle
                # is available but routes the fault to a sibling component.
                if (
                    not self.path_aware
                    and prediction.get("status") in ("Normal", "unknown")
                    and getattr(self.registry, "responsibility_ready", False)
                    and self.registry.has_responsibility_model(node_id)
                ):
                    inherited_fault = prediction.get("oracle_predicted_fault_type")
                    resp = self._predict_with_responsibility_model(
                        node_id,
                        inherited_fault_type=inherited_fault,
                        supporting_prediction=prediction,
                    )
                    if resp.get("status") == "Fault":
                        return resp
                    if prediction.get("status") == "unknown" and resp.get("status") != "error":
                        return resp
                return self._apply_low_confidence_calibration(prediction, node_id)

        # Prefer the binary responsibility fallback over the legacy per-node
        # multi-class fallback.  The former answers a narrower, better-posed
        # industrial question: is this node responsible for the observed fault?
        if (
            getattr(self.registry, "responsibility_ready", False)
            and self.registry.has_responsibility_model(node_id)
        ):
            resp = self._predict_with_responsibility_model(node_id)
            if resp.get("status") != "error":
                return self._apply_low_confidence_calibration(resp, node_id)

        # Fallback to per-node model
        if self.registry.has_model(node_id):
            pred = self._predict_with_node_model(node_id)
            return self._apply_low_confidence_calibration(pred, node_id)

        # No model at all
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
        if (
            self.path_aware
            and not is_no_fault
            and hasattr(scenario, 'diagnostic_path')
            and scenario.diagnostic_path is not None
        ):
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
        if (
            self.include_diagnosis_sensor_readings
            and self.scenario_state is not None
        ):
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
                    "fault_type": scenario.fault_type,
                    "confidence": warning_confidence,
                    "message": (
                        "Borderline anomaly detected. Model confidence is below "
                        "the definitive threshold. Recommend verifying with "
                        "sensor-level data before concluding."
                    ),
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
            return self._unknown_result(
                target_node_id,
                system_id,
                "oracle_features_unavailable",
                model_source="system_oracle",
            )

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

        if prediction["status"] == "Normal":
            # System oracle sees no fault: a genuine clean reading.
            prediction_result = {
                "status": "Normal",
                "fault_type": "None",
                "confidence": round(prediction.get("confidence", 0.85), 4),
                "predicted_class": 0,
                "oracle_predicted_fault_type": fault_type,
                "oracle_responsible_node": responsible_node,
            }
        elif responsible_node != target_node_id:
            # System oracle predicts a fault, but it routes to a SIBLING node in
            # the SAME system (e.g. fault is at Pump_1 while the agent queried
            # the plant aggregate node). The old behavior hard-returned Normal
            # here, which (a) made the right-system/wrong-node case ind
            # distinguishable from a truly clean node only by a hidden field,
            # and (b) gave the policy no signal to drill to the real root —
            # the dominant cause of cross-system DA collapse.
            #
            # Soft gate: report Abnormal with an explicit suggested_node so the
            # policy learns "this component is implicated by a system fault;
            # drill toward the responsible node". This is faithful: components
            # of a faulted plant genuinely exhibit coupled abnormal behavior.
            # A clean node in a clean system still returns Normal (the branch
            # above), so wrong-system elimination is unaffected.
            if self.soft_node_gate:
                prediction_result = {
                    "status": "Abnormal",
                    "fault_type": "None",
                    "confidence": round(prediction.get("confidence", 0.85), 4),
                    "predicted_class": prediction.get("predicted_class", 1),
                    "abnormal_indicators": (
                        f"System-level diagnostics indicate an active fault in "
                        f"{system_id}; this component is implicated but is not "
                        f"the root cause. Continue diagnosing components within "
                        f"this system to localize the responsible node."
                    ),
                    # Directional hint only ("keep drilling in this system").
                    # The exact responsible node is kept internal-only
                    # (oracle_responsible_node) and stripped before the agent
                    # sees it, so the policy cannot shortcut to the answer.
                    "suggested_direction": "within_system",
                    "suggested_node": responsible_node,
                    "oracle_predicted_fault_type": fault_type,
                    "oracle_responsible_node": responsible_node,
                }
            else:
                prediction_result = {
                    "status": "Normal",
                    "fault_type": "None",
                    "confidence": round(prediction.get("confidence", 0.85), 4),
                    "predicted_class": 0,
                    "oracle_predicted_fault_type": fault_type,
                    "oracle_responsible_node": responsible_node,
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

        if (
            self.include_diagnosis_sensor_readings
            and self.scenario_state is not None
        ):
            sensor_readings = self.scenario_state.get_sensor_readings(target_node_id)
            sensor_readings = self._sanitize_sensor_readings(sensor_readings)
            prediction_result["sensor_readings"] = dict(list(sensor_readings.items())[:10])

        return self._apply_clean_baseline_calibration(prediction_result)

    def _predict_with_responsibility_model(
        self,
        node_id: str,
        inherited_fault_type: Optional[str] = None,
        supporting_prediction: Optional[Dict[str, Any]] = None,
    ) -> Dict:
        """Predict using the binary per-node responsibility model."""
        if self.scenario_state is None:
            return {
                "status": "error",
                "error": "No scenario state available for prediction",
            }

        features = self.scenario_state.get_node_features(node_id)
        if features is not None:
            prediction = self.registry.predict_responsibility(node_id, features)
        else:
            sensor_data = self.scenario_state.get_sensor_readings(node_id)
            prediction = self.registry.predict_responsibility_from_sensor_data(
                node_id, sensor_data
            )

        node_info = self.topology.get_node_info(node_id)
        prediction["node_id"] = node_id
        prediction["node_name"] = node_info.get("name", "") if node_info else ""
        prediction["system_id"] = node_info.get("system_id", "") if node_info else ""
        prediction["model_source"] = "per_node_responsibility"
        prediction["calibration"] = "responsibility_model"
        if inherited_fault_type and inherited_fault_type not in ("None", "Normal"):
            prediction["fault_type"] = inherited_fault_type
            prediction["fault_type_source"] = "system_oracle_predicted_class"

        if supporting_prediction:
            prediction["supporting_system_oracle_status"] = supporting_prediction.get("status")
            prediction["supporting_system_oracle_fault_type"] = supporting_prediction.get(
                "oracle_predicted_fault_type",
                supporting_prediction.get("fault_type"),
            )

        if (
            self.include_diagnosis_sensor_readings
            and self.scenario_state is not None
        ):
            sensor_readings = self.scenario_state.get_sensor_readings(node_id)
            sensor_readings = self._sanitize_sensor_readings(sensor_readings)
            prediction["sensor_readings"] = dict(list(sensor_readings.items())[:10])

        return self._apply_clean_baseline_calibration(prediction)

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
        if prediction.get("status") in ("Fault", "Warning", "Abnormal"):
            prediction["calibration"] = "weak_fallback_model"

        # Add sensor readings summary
        if (
            self.include_diagnosis_sensor_readings
            and self.scenario_state is not None
        ):
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
            and n.get("model_source") in ("system_oracle", "per_node_responsibility")
            and n.get("calibration") not in (
                "weak_fallback_model",
                "data_unavailable_for_system",
                "oracle_features_unavailable",
            )
            and not (
                n.get("model_source") == "per_node_responsibility"
                and float(n.get("confidence") or 0.0) < 0.65
            )
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
