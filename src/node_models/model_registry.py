"""
Model Registry — Central registry for trained node fault prediction models.

Provides a unified interface for loading and querying node models, with:
  - Lazy loading and caching of LightGBM models
  - Prediction with confidence scores
  - Model metadata access
"""

import os
import json
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import lightgbm as lgb

from src.utils.io_utils import load_json, setup_logger

logger = setup_logger(__name__)


class ModelRegistry:
    """
    Central registry mapping node_id → trained model.

    Scans a model directory, loads metadata, and provides prediction interface.
    """

    def __init__(self, models_base_dir: str):
        """
        Args:
            models_base_dir: Base directory containing per-system/per-node model folders.
        """
        self.models_base_dir = models_base_dir
        self._metadata_cache: Dict[str, Dict] = {}  # node_id → metadata
        self._model_cache: Dict[str, lgb.Booster] = {}  # node_id → loaded model
        self._responsibility_metadata_cache: Dict[str, Dict] = {}
        self._responsibility_model_cache: Dict[str, lgb.Booster] = {}
        self.responsibility_ready = self._responsibility_manifest_complete()
        self._scan_models()

    def _responsibility_manifest_complete(self) -> bool:
        """Whether responsibility models were trained as a complete artifact set."""
        manifest_path = os.path.join(
            self.models_base_dir, "responsibility_training_summary.json"
        )
        if not os.path.exists(manifest_path):
            return False
        try:
            manifest = load_json(manifest_path)
        except Exception:
            return False
        return bool(manifest.get("complete", False))

    def _scan_models(self) -> None:
        """Scan the model directory and build the metadata index."""
        if not os.path.isdir(self.models_base_dir):
            logger.warning(f"Model directory not found: {self.models_base_dir}")
            return

        count = 0
        for system_dir in os.listdir(self.models_base_dir):
            sys_path = os.path.join(self.models_base_dir, system_dir)
            if not os.path.isdir(sys_path):
                continue

            for node_dir in os.listdir(sys_path):
                meta_path = os.path.join(sys_path, node_dir, "metadata.json")
                if os.path.exists(meta_path):
                    try:
                        meta = load_json(meta_path)
                        node_id = meta.get("node_id", node_dir.replace("__", "::"))
                        # Fix relative model path to absolute
                        if not os.path.isabs(meta.get("model_path", "")):
                            meta["model_path"] = os.path.join(
                                sys_path, node_dir, "model.txt"
                            )
                        self._metadata_cache[node_id] = meta
                        count += 1
                    except Exception as e:
                        logger.warning(f"Failed to load metadata: {meta_path}: {e}")

                resp_meta_path = os.path.join(
                    sys_path, node_dir, "responsibility_metadata.json"
                )
                if os.path.exists(resp_meta_path):
                    try:
                        resp_meta = load_json(resp_meta_path)
                        node_id = resp_meta.get("node_id", node_dir.replace("__", "::"))
                        if not os.path.isabs(resp_meta.get("model_path", "")):
                            resp_meta["model_path"] = os.path.join(
                                sys_path, node_dir, "responsibility_model.txt"
                            )
                        self._responsibility_metadata_cache[node_id] = resp_meta
                    except Exception as e:
                        logger.warning(
                            "Failed to load responsibility metadata: "
                            f"{resp_meta_path}: {e}"
                        )

        logger.info(
            f"Model registry: {count} models indexed from {self.models_base_dir} "
            f"({len(self._responsibility_metadata_cache)} responsibility models, "
            f"ready={self.responsibility_ready})"
        )

    def _load_model(self, node_id: str) -> Optional[lgb.Booster]:
        """Lazily load a LightGBM model into memory."""
        if node_id in self._model_cache:
            return self._model_cache[node_id]

        meta = self._metadata_cache.get(node_id)
        if meta is None:
            return None

        model_path = meta.get("model_path", "")
        if not os.path.exists(model_path):
            logger.error(f"Model file not found: {model_path}")
            return None

        try:
            model = lgb.Booster(model_file=model_path)
            self._model_cache[node_id] = model
            return model
        except Exception as e:
            logger.error(f"Failed to load model for {node_id}: {e}")
            return None

    def has_model(self, node_id: str) -> bool:
        """Check if a model exists for the given node."""
        return node_id in self._metadata_cache

    def get_metadata(self, node_id: str) -> Optional[Dict]:
        """Get metadata for a node's model."""
        return self._metadata_cache.get(node_id)

    def has_responsibility_model(self, node_id: str) -> bool:
        """Check if a binary responsibility model exists for the node."""
        return node_id in self._responsibility_metadata_cache

    def get_responsibility_metadata(self, node_id: str) -> Optional[Dict]:
        """Get metadata for a node responsibility model."""
        return self._responsibility_metadata_cache.get(node_id)

    def get_responsibility_node_ids(self) -> List[str]:
        """Return all node IDs with responsibility models."""
        return list(self._responsibility_metadata_cache.keys())

    def get_all_node_ids(self) -> List[str]:
        """Return all node IDs with trained models."""
        return list(self._metadata_cache.keys())

    def get_feature_cols(self, node_id: str) -> List[str]:
        """Get the list of feature columns required by a node's model."""
        meta = self._metadata_cache.get(node_id, {})
        return meta.get("feature_cols", [])

    def get_responsibility_feature_cols(self, node_id: str) -> List[str]:
        """Get feature columns required by a responsibility model."""
        meta = self._responsibility_metadata_cache.get(node_id, {})
        return meta.get("feature_cols", [])

    def _load_responsibility_model(self, node_id: str) -> Optional[lgb.Booster]:
        """Lazily load a responsibility LightGBM model."""
        if node_id in self._responsibility_model_cache:
            return self._responsibility_model_cache[node_id]

        meta = self._responsibility_metadata_cache.get(node_id)
        if meta is None:
            return None

        model_path = meta.get("model_path", "")
        if not os.path.exists(model_path):
            logger.error(f"Responsibility model file not found: {model_path}")
            return None

        try:
            model = lgb.Booster(model_file=model_path)
            self._responsibility_model_cache[node_id] = model
            return model
        except Exception as e:
            logger.error(f"Failed to load responsibility model for {node_id}: {e}")
            return None

    def predict(
        self,
        node_id: str,
        features: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Run prediction for a node given pre-computed features.

        Args:
            node_id: The node to predict for.
            features: 1D or 2D numpy array of feature values matching
                      the node's feature_cols order.

        Returns:
            Dict with: status, fault_type, confidence, all_probabilities.
        """
        meta = self._metadata_cache.get(node_id)
        if meta is None:
            return {
                "status": "error",
                "error": f"No model found for node '{node_id}'",
            }

        model = self._load_model(node_id)
        if model is None:
            return {
                "status": "error",
                "error": f"Failed to load model for node '{node_id}'",
            }

        # Ensure 2D input
        if features.ndim == 1:
            features = features.reshape(1, -1)

        try:
            raw_pred = model.predict(features)
        except Exception as e:
            return {"status": "error", "error": f"Prediction failed: {e}"}

        is_binary = meta.get("is_binary", False)
        reverse_map = meta.get("reverse_label_map", {})
        temperature = float(meta.get("calibration_temperature", 1.0) or 1.0)

        if is_binary:
            # raw_pred is probability of class 1
            prob = float(raw_pred[0]) if raw_pred.ndim == 1 else float(raw_pred[0, 0])
            if abs(temperature - 1.0) > 1e-6:
                # Re-temperature via pseudo-logit on the 2-class distribution.
                import math as _math
                p1 = min(max(prob, 1e-12), 1.0 - 1e-12)
                margin = _math.log(p1 / (1.0 - p1))
                prob = 1.0 / (1.0 + _math.exp(-margin / temperature))
            predicted_class = 1 if prob > 0.5 else 0
            confidence = prob if predicted_class == 1 else (1 - prob)
            probabilities = {
                reverse_map.get("0", "Normal"): 1 - prob,
                reverse_map.get("1", "Fault"): prob,
            }
        else:
            # raw_pred is (1, n_classes) probability matrix
            probs = raw_pred[0] if raw_pred.ndim == 2 else raw_pred
            if abs(temperature - 1.0) > 1e-6:
                from src.node_models.calibration import _softmax_with_temperature
                # Recover pseudo-logits from softmax probs, then re-temperature.
                logp = np.log(np.clip(probs, 1e-12, None))
                probs = _softmax_with_temperature(logp.reshape(1, -1), temperature)[0]
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])
            probabilities = {}
            for i, p in enumerate(probs):
                label = reverse_map.get(str(i), f"class_{i}")
                probabilities[label] = float(p)

        fault_type = reverse_map.get(str(predicted_class), f"class_{predicted_class}")
        is_normal = fault_type == "Normal" or predicted_class == 0

        return {
            "status": "Normal" if is_normal else "Fault",
            "fault_type": fault_type if not is_normal else "None",
            "confidence": round(confidence, 4),
            "predicted_class": predicted_class,
            "probabilities": probabilities,
            "calibration_temperature": round(temperature, 4),
        }

    def predict_from_sensor_data(
        self,
        node_id: str,
        sensor_data: Dict[str, float],
        min_feature_coverage: float = 0.60,
    ) -> Dict[str, Any]:
        """
        Predict node status from raw sensor readings (single time point).

        This is a convenience method for the tool executor; for windowed
        predictions, use the feature_engineer + predict pipeline.

        Args:
            node_id: Node to predict for.
            sensor_data: Dict of sensor_name → value.

        Returns:
            Prediction result dict.
        """
        feature_cols = self.get_feature_cols(node_id)
        if not feature_cols:
            return {
                "status": "error",
                "error": f"No feature columns defined for node '{node_id}'",
            }

        # Build feature vector from sensor data.  A low coverage vector is not a
        # reliable diagnostic input: silently padding most features with zeros
        # creates false positives when the agent probes an unrelated system.
        # For single-point prediction, use mean features (no window stats)
        features = []
        matched = 0
        for col in feature_cols:
            # Try to extract the base sensor name from the feature column name
            # Feature columns follow pattern: SENSOR_NAME_stat (e.g., CHL_SW_TEMP_1_mean)
            base = col.rsplit("_", 1)[0] if "_" in col else col
            if col in sensor_data:
                features.append(sensor_data[col])
                matched += 1
            elif base in sensor_data:
                features.append(sensor_data[base])
                matched += 1
            else:
                features.append(0.0)

        coverage = matched / max(len(feature_cols), 1)
        if coverage < min_feature_coverage:
            return {
                "status": "unknown",
                "fault_type": "None",
                "confidence": 0.0,
                "error": (
                    "Insufficient sensor coverage for per-node prediction: "
                    f"{matched}/{len(feature_cols)} features available"
                ),
                "feature_coverage": round(coverage, 4),
            }

        result = self.predict(node_id, np.array(features, dtype=np.float32))
        result["feature_coverage"] = round(coverage, 4)
        return result

    def predict_responsibility(
        self,
        node_id: str,
        features: np.ndarray,
    ) -> Dict[str, Any]:
        """Predict whether a node is the responsible root cause."""
        meta = self._responsibility_metadata_cache.get(node_id)
        if meta is None:
            return {
                "status": "error",
                "error": f"No responsibility model found for node '{node_id}'",
            }

        model = self._load_responsibility_model(node_id)
        if model is None:
            return {
                "status": "error",
                "error": f"Failed to load responsibility model for node '{node_id}'",
            }

        if features.ndim == 1:
            features = features.reshape(1, -1)

        try:
            raw_pred = model.predict(features)
        except Exception as e:
            return {"status": "error", "error": f"Prediction failed: {e}"}

        prob = float(raw_pred[0]) if raw_pred.ndim == 1 else float(raw_pred[0, 0])
        threshold = float(meta.get("responsibility_threshold", 0.60))
        responsible = prob >= threshold
        confidence = prob if responsible else 1.0 - prob
        return {
            "status": "Fault" if responsible else "Normal",
            "fault_type": "responsible_node" if responsible else "None",
            "confidence": round(confidence, 4),
            "responsibility_probability": round(prob, 4),
            "responsibility_threshold": round(threshold, 4),
            "predicted_class": 1 if responsible else 0,
            "probabilities": {
                "Normal": round(1.0 - prob, 4),
                "Responsible": round(prob, 4),
            },
        }

    def predict_responsibility_from_sensor_data(
        self,
        node_id: str,
        sensor_data: Dict[str, float],
        min_feature_coverage: float = 0.60,
    ) -> Dict[str, Any]:
        """Predict node responsibility from raw sensor readings."""
        feature_cols = self.get_responsibility_feature_cols(node_id)
        if not feature_cols:
            return {
                "status": "error",
                "error": f"No responsibility feature columns for node '{node_id}'",
            }

        features = []
        matched = 0
        for col in feature_cols:
            base = col.rsplit("_", 1)[0] if "_" in col else col
            if col in sensor_data:
                features.append(sensor_data[col])
                matched += 1
            elif base in sensor_data:
                features.append(sensor_data[base])
                matched += 1
            else:
                features.append(0.0)

        coverage = matched / max(len(feature_cols), 1)
        if coverage < min_feature_coverage:
            return {
                "status": "unknown",
                "fault_type": "None",
                "confidence": 0.0,
                "error": (
                    "Insufficient sensor coverage for responsibility prediction: "
                    f"{matched}/{len(feature_cols)} features available"
                ),
                "feature_coverage": round(coverage, 4),
            }

        result = self.predict_responsibility(node_id, np.array(features, dtype=np.float32))
        result["feature_coverage"] = round(coverage, 4)
        return result
