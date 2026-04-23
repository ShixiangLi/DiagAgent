"""
Model Trainer — Train node-level fault prediction classifiers.

Trains one LightGBM multi-class classifier per topology node (component)
using sensor data assigned to that node. Models are trained on the full
dataset to act as "oracle" predictors for the diagnostic agent environment.

Handles:
  - Class imbalance via SMOTE + class weights
  - Automatic fallback to binary classification for nodes with limited faults
  - Model serialization with metadata
"""

import os
import json
import pickle
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score
import lightgbm as lgb

from src.utils.io_utils import ensure_dir, save_json, setup_logger

logger = setup_logger(__name__)


def _compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Compute balanced class weights inversely proportional to frequency."""
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {}
    for cls, cnt in zip(classes, counts):
        weights[int(cls)] = total / (len(classes) * cnt)
    return weights


def train_node_model(
    features: pd.DataFrame,
    feature_cols: List[str],
    label_col: str = "fault_label",
    label_map: Optional[Dict[str, int]] = None,
    node_id: str = "unknown",
    output_dir: str = "models",
    use_smote: bool = True,
    test_size: float = 0.15,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Train a LightGBM classifier for a single node.

    Args:
        features: DataFrame with feature columns and fault_label.
        feature_cols: List of feature column names to use.
        label_col: Column containing integer fault labels.
        label_map: Optional mapping of label strings to integers.
        node_id: Identifier for this node (used in filenames).
        output_dir: Directory to save the trained model.
        use_smote: Whether to apply SMOTE oversampling for minority classes.
        test_size: Fraction of data for validation.
        random_state: Random seed.

    Returns:
        Dict with model path, metrics, and metadata.
    """
    if features.empty or not feature_cols:
        logger.warning(f"No features/data for node {node_id}, skipping")
        return {"node_id": node_id, "status": "skipped", "reason": "no_data"}

    X = features[feature_cols].values.astype(np.float32)
    y = features[label_col].values.astype(np.int32)

    # Filter out classes with fewer than 2 samples (can't stratify them)
    unique_classes, class_counts = np.unique(y, return_counts=True)
    valid_classes = unique_classes[class_counts >= 2]
    if len(valid_classes) < 2:
        logger.warning(f"Node {node_id}: fewer than 2 valid classes, skipping")
        return {"node_id": node_id, "status": "skipped", "reason": "insufficient_classes"}

    mask = np.isin(y, valid_classes)
    X, y = X[mask], y[mask]

    # Re-map labels to contiguous integers
    label_remap = {old: new for new, old in enumerate(sorted(valid_classes))}
    y = np.array([label_remap[v] for v in y], dtype=np.int32)

    n_classes = len(valid_classes)

    # Build reverse label map (only for valid classes)
    if label_map is None:
        label_map = {str(i): i for i in valid_classes}
    reverse_label_map = {v: k for k, v in label_map.items()}
    # Update reverse map for remapped labels
    remapped_reverse = {}
    for old_label, new_label in label_remap.items():
        old_name = reverse_label_map.get(old_label, str(old_label))
        remapped_reverse[new_label] = old_name
    reverse_label_map = remapped_reverse

    # Decide: multi-class or binary
    is_binary = n_classes == 2

    # Dynamically adjust test_size to ensure enough samples per class
    min_test_samples = max(n_classes, 2)
    actual_test_size = max(test_size, min_test_samples / len(y) + 0.01)
    actual_test_size = min(actual_test_size, 0.5)  # Never more than 50%

    # Split train/val
    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=actual_test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        # Fallback: split without stratification
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=actual_test_size, random_state=random_state
        )

    # Apply SMOTE if requested and we have enough samples
    if use_smote and n_classes >= 2:
        try:
            from imblearn.over_sampling import SMOTE
            min_class_count = min(np.bincount(y_train)[np.bincount(y_train) > 0])
            if min_class_count >= 6:
                k = min(5, min_class_count - 1)
                smote = SMOTE(random_state=random_state, k_neighbors=k)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                logger.info(f"  SMOTE applied: {len(X_train)} training samples")
        except Exception as e:
            logger.warning(f"  SMOTE failed for {node_id}: {e}")

    # Compute class weights
    class_weights = _compute_class_weights(y_train)
    sample_weights = np.array([class_weights[int(label)] for label in y_train])

    # LightGBM parameters
    params = {
        "objective": "binary" if is_binary else "multiclass",
        "metric": "binary_logloss" if is_binary else "multi_logloss",
        "num_class": 1 if is_binary else n_classes,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "max_depth": 8,
        "min_child_samples": min(20, max(1, len(X_train) // (n_classes * 5))),
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "n_jobs": -1,
        "random_state": random_state,
    }

    train_set = lgb.Dataset(X_train, y_train, weight=sample_weights)
    val_set = lgb.Dataset(X_val, y_val, reference=train_set)

    # Train with early stopping
    model = lgb.train(
        params,
        train_set,
        num_boost_round=500,
        valid_sets=[val_set],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    # Evaluate
    if is_binary:
        y_pred_prob = model.predict(X_val)
        y_pred = (y_pred_prob > 0.5).astype(int)
    else:
        y_pred_prob = model.predict(X_val)
        y_pred = y_pred_prob.argmax(axis=1)

    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average="weighted")

    all_labels_in_data = sorted(set(y_val) | set(y_pred))
    target_names = [reverse_label_map.get(i, str(i)) for i in all_labels_in_data]
    report = classification_report(
        y_val, y_pred, target_names=target_names, output_dict=True,
        labels=all_labels_in_data, zero_division=0,
    )

    logger.info(
        f"  Node {node_id}: accuracy={accuracy:.4f}, "
        f"weighted-F1={f1:.4f}, classes={n_classes}"
    )

    # Save model
    safe_node_id = node_id.replace("::", "__").replace("/", "_")
    model_dir = ensure_dir(os.path.join(output_dir, safe_node_id))
    model_path = os.path.join(model_dir, "model.txt")
    model.save_model(model_path)

    # Save metadata
    metadata = {
        "node_id": node_id,
        "status": "trained",
        "model_path": model_path,
        "n_classes": n_classes,
        "is_binary": is_binary,
        "label_map": label_map,
        "reverse_label_map": {str(k): v for k, v in reverse_label_map.items()},
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "n_train_samples": len(X_train),
        "n_val_samples": len(X_val),
        "accuracy": accuracy,
        "weighted_f1": f1,
        "classification_report": report,
    }
    save_json(metadata, os.path.join(model_dir, "metadata.json"))

    return metadata


def train_system_oracle_model(
    system_df: pd.DataFrame,
    system_id: str,
    topology_builder,
    label_map: Dict[str, int],
    output_dir: str = "models",
    window_size: int = 15,
    stride: int = 15,
) -> Dict[str, Any]:
    """
    Train a single system-level oracle model using ALL sensors.

    This is the primary diagnostic model — it sees all sensors in the system
    simultaneously, giving it much higher accuracy than per-node models.
    The prediction is then routed to the appropriate component node via
    a fault-type-to-node mapping.

    Args:
        system_df: Full system DataFrame with all sensor data and labels.
        system_id: System identifier.
        topology_builder: TopologyBuilder instance.
        label_map: Mapping of fault label strings to integer codes.
        output_dir: Base directory for saving models.
        window_size: Feature aggregation window.
        stride: Window step size.

    Returns:
        Training result dict with model path, metrics, and fault-node mapping.
    """
    from src.node_models.feature_engineer import build_node_features

    # Gather ALL sensors across ALL components in this system
    components = topology_builder.get_system_components(system_id)
    all_sensors = []
    for comp in components:
        all_sensors.extend(comp.get("sensor_names", []))
    all_sensors = list(set(all_sensors))  # deduplicate

    if not all_sensors:
        logger.warning(f"No sensors found for system {system_id}")
        return {"node_id": f"{system_id}::oracle", "status": "skipped", "reason": "no_sensors"}

    logger.info(
        f"  Training system oracle for '{system_id}' "
        f"({len(all_sensors)} sensors across {len(components)} components)"
    )

    # Build features using ALL sensors
    feat_df, feature_cols = build_node_features(
        system_df, all_sensors,
        window_size=window_size, stride=stride,
    )

    # Fallback: if topology sensor names don't match CSV columns (e.g., FCU where
    # TTL uses semantic names but CSV has abbreviated column names), use ALL
    # numeric columns from the DataFrame as features instead.
    if (feat_df.empty or not feature_cols) and not system_df.empty:
        logger.warning(
            f"  Topology sensor names didn't match CSV columns for '{system_id}'. "
            f"Falling back to all numeric columns."
        )
        exclude_cols = {"fault_label", "fault_label_str", "source_file", "Datetime"}
        numeric_cols = [
            c for c in system_df.select_dtypes(include=["number"]).columns
            if c not in exclude_cols
        ]
        if numeric_cols:
            feat_df, feature_cols = build_node_features(
                system_df, numeric_cols,
                window_size=window_size, stride=stride,
            )
            if feature_cols:
                logger.info(
                    f"  Fallback succeeded: {len(feature_cols)} features from "
                    f"{len(numeric_cols)} numeric columns"
                )

    if feat_df.empty or not feature_cols:
        return {"node_id": f"{system_id}::oracle", "status": "skipped", "reason": "no_valid_features"}

    # Train the system oracle
    oracle_node_id = f"{system_id}::oracle"
    result = train_node_model(
        features=feat_df,
        feature_cols=feature_cols,
        label_map=label_map,
        node_id=oracle_node_id,
        output_dir=os.path.join(output_dir, system_id),
    )

    # Build fault-type → root-cause-node mapping
    # This lets us route system-level predictions to specific component nodes
    fault_node_map = _build_fault_node_mapping(system_id, components, label_map)
    result["fault_node_map"] = fault_node_map

    # Save the mapping alongside the model
    if result.get("status") == "trained":
        safe_id = oracle_node_id.replace("::", "__").replace("/", "_")
        map_path = os.path.join(output_dir, system_id, safe_id, "fault_node_map.json")
        save_json(fault_node_map, map_path)

    return result


def _build_fault_node_mapping(
    system_id: str,
    components: List[Dict],
    label_map: Dict[str, int],
) -> Dict[str, str]:
    """
    Build a mapping from fault_type → most likely root cause node_id.

    Uses keyword matching to associate each fault type with the component
    that is most likely its root cause.
    """
    from src.environment.fault_scenario import _infer_root_cause_node

    fault_node_map = {}
    for fault_label, label_idx in label_map.items():
        if fault_label == "Normal":
            fault_node_map[fault_label] = "none"
        else:
            # Extract just the fault type (remove intensity suffix)
            node_id = _infer_root_cause_node(fault_label, system_id, components)
            fault_node_map[fault_label] = node_id

    return fault_node_map


def train_all_system_models(
    system_df: pd.DataFrame,
    system_id: str,
    topology_builder,
    label_map: Dict[str, int],
    output_dir: str = "models",
    window_size: int = 15,
    stride: int = 15,
    max_rows_per_file: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Train models for a system: one system-level oracle + per-node models.

    The system oracle uses ALL sensors and is the primary high-accuracy
    predictor. Per-node models are secondary and use only local sensors.

    Args:
        system_df: Full system DataFrame with all sensor data and labels.
        system_id: System identifier.
        topology_builder: TopologyBuilder instance with built graph.
        label_map: Mapping of fault label strings to integer codes.
        output_dir: Base directory for saving models.
        window_size: Feature aggregation window.
        stride: Window step size.

    Returns:
        List of training result dicts (oracle + per-node).
    """
    from src.node_models.feature_engineer import build_node_features

    results = []

    # ================================================================
    # 1. Train SYSTEM-LEVEL ORACLE (high accuracy, all sensors)
    # ================================================================
    oracle_result = train_system_oracle_model(
        system_df, system_id, topology_builder, label_map,
        output_dir=output_dir,
        window_size=window_size, stride=stride,
    )
    results.append(oracle_result)

    if oracle_result.get("status") == "trained":
        logger.info(
            f"  System oracle '{system_id}::oracle': "
            f"accuracy={oracle_result.get('accuracy', 0):.4f}, "
            f"weighted-F1={oracle_result.get('weighted_f1', 0):.4f}"
        )

    # ================================================================
    # 2. Train PER-NODE models (lower accuracy, local sensors only)
    # ================================================================
    components = topology_builder.get_system_components(system_id)
    logger.info(
        f"Training per-node models for {len(components)} components in '{system_id}'"
    )

    for comp in components:
        node_id = comp["node_id"]
        sensor_names = comp.get("sensor_names", [])

        if not sensor_names:
            logger.debug(f"  Skipping {node_id}: no sensors assigned")
            results.append({
                "node_id": node_id,
                "status": "skipped",
                "reason": "no_sensors",
            })
            continue

        logger.info(
            f"  Training model for {node_id} "
            f"({len(sensor_names)} sensors: {sensor_names[:5]}...)"
        )

        # Build features for this node
        feat_df, feature_cols = build_node_features(
            system_df, sensor_names,
            window_size=window_size, stride=stride,
        )

        if feat_df.empty or not feature_cols:
            results.append({
                "node_id": node_id,
                "status": "skipped",
                "reason": "no_valid_features",
            })
            continue

        # Train model
        result = train_node_model(
            features=feat_df,
            feature_cols=feature_cols,
            label_map=label_map,
            node_id=node_id,
            output_dir=os.path.join(output_dir, system_id),
        )
        results.append(result)

    trained = sum(1 for r in results if r.get("status") == "trained")
    logger.info(
        f"System '{system_id}': {trained}/{len(results)} models trained "
        f"(1 oracle + {trained - 1} per-node)"
    )
    return results

