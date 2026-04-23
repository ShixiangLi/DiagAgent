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
        self._scan_models()

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

        logger.info(f"Model registry: {count} models indexed from {self.models_base_dir}")

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

    def get_all_node_ids(self) -> List[str]:
        """Return all node IDs with trained models."""
        return list(self._metadata_cache.keys())

    def get_feature_cols(self, node_id: str) -> List[str]:
        """Get the list of feature columns required by a node's model."""
        meta = self._metadata_cache.get(node_id, {})
        return meta.get("feature_cols", [])

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

        if is_binary:
            # raw_pred is probability of class 1
            prob = float(raw_pred[0]) if raw_pred.ndim == 1 else float(raw_pred[0, 0])
            predicted_class = 1 if prob > 0.5 else 0
            confidence = prob if predicted_class == 1 else (1 - prob)
            probabilities = {
                reverse_map.get("0", "Normal"): 1 - prob,
                reverse_map.get("1", "Fault"): prob,
            }
        else:
            # raw_pred is (1, n_classes) probability matrix
            probs = raw_pred[0] if raw_pred.ndim == 2 else raw_pred
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
        }

    def predict_from_sensor_data(
        self,
        node_id: str,
        sensor_data: Dict[str, float],
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

        # Build feature vector from sensor data
        # For single-point prediction, use mean features (no window stats)
        features = []
        for col in feature_cols:
            # Try to extract the base sensor name from the feature column name
            # Feature columns follow pattern: SENSOR_NAME_stat (e.g., CHL_SW_TEMP_1_mean)
            base = col.rsplit("_", 1)[0] if "_" in col else col
            if col in sensor_data:
                features.append(sensor_data[col])
            elif base in sensor_data:
                features.append(sensor_data[base])
            else:
                features.append(0.0)

        return self.predict(node_id, np.array(features, dtype=np.float32))
