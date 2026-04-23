"""
Feature Engineer — Extract per-node features from sensor time-series data.

Creates windowed statistical features from raw 1-minute sensor data:
  - Window aggregates: mean, std, min, max, range, slope
  - Cross-sensor features: temperature differentials, setpoint deviations
  - Temporal features: hour-of-day, day-of-week encoding
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.io_utils import setup_logger

logger = setup_logger(__name__)


def compute_window_features(
    df: pd.DataFrame,
    sensor_cols: List[str],
    window_size: int = 15,
    stride: int = 15,
) -> pd.DataFrame:
    """
    Compute windowed statistical features from raw time-series data.

    Args:
        df: DataFrame with sensor columns and a datetime-parseable first column.
        sensor_cols: List of sensor column names to extract features from.
        window_size: Window size in minutes (number of rows at 1-min frequency).
        stride: Step size between windows in minutes.

    Returns:
        DataFrame where each row represents one time window with statistical
        features for each sensor.
    """
    # Filter to only existing sensor columns
    available = [c for c in sensor_cols if c in df.columns]
    if not available:
        logger.warning("No matching sensor columns found in DataFrame")
        return pd.DataFrame()

    # Extract numeric data for the relevant sensors
    sensor_data = df[available].apply(pd.to_numeric, errors="coerce")

    # Parse timestamp if available (for temporal features)
    time_col = df.columns[0]
    timestamps = pd.to_datetime(df[time_col], errors="coerce")

    features_list = []
    n_rows = len(sensor_data)

    for start in range(0, n_rows - window_size + 1, stride):
        end = start + window_size
        window = sensor_data.iloc[start:end]

        feat = {}

        # Timestamp of the window center
        window_center = start + window_size // 2
        if window_center < len(timestamps) and pd.notna(timestamps.iloc[window_center]):
            ts = timestamps.iloc[window_center]
            feat["hour_of_day"] = ts.hour
            feat["day_of_week"] = ts.dayofweek
            feat["is_occupied"] = 1 if (6 <= ts.hour <= 20 and ts.dayofweek < 6) else 0
            feat["timestamp"] = str(ts)
        else:
            feat["hour_of_day"] = 0
            feat["day_of_week"] = 0
            feat["is_occupied"] = 0
            feat["timestamp"] = ""

        # Per-sensor statistical features
        for col in available:
            vals = window[col].dropna()
            prefix = col

            if len(vals) == 0:
                feat[f"{prefix}_mean"] = np.nan
                feat[f"{prefix}_std"] = 0.0
                feat[f"{prefix}_min"] = np.nan
                feat[f"{prefix}_max"] = np.nan
                feat[f"{prefix}_range"] = 0.0
                feat[f"{prefix}_slope"] = 0.0
                continue

            feat[f"{prefix}_mean"] = vals.mean()
            feat[f"{prefix}_std"] = vals.std() if len(vals) > 1 else 0.0
            feat[f"{prefix}_min"] = vals.min()
            feat[f"{prefix}_max"] = vals.max()
            feat[f"{prefix}_range"] = vals.max() - vals.min()

            # Slope via linear regression on window indices
            if len(vals) > 2:
                x = np.arange(len(vals), dtype=float)
                coeffs = np.polyfit(x, vals.values, 1)
                feat[f"{prefix}_slope"] = coeffs[0]
            else:
                feat[f"{prefix}_slope"] = 0.0

        features_list.append(feat)

    result = pd.DataFrame(features_list)
    return result


def extract_cross_sensor_features(
    df: pd.DataFrame,
    sensor_pairs: Optional[List[Tuple[str, str]]] = None,
) -> pd.DataFrame:
    """
    Add cross-sensor derived features (differentials, ratios).

    Auto-detects supply/return temperature pairs, setpoint deviations, etc.

    Args:
        df: Feature DataFrame (output of compute_window_features).
        sensor_pairs: Optional explicit pairs of (sensor_a_mean, sensor_b_mean)
            columns to compute differentials for.

    Returns:
        DataFrame with additional cross-sensor feature columns appended.
    """
    result = df.copy()
    mean_cols = [c for c in df.columns if c.endswith("_mean")]

    # Auto-detect temperature differentials (supply - return)
    for col in mean_cols:
        col_base = col.replace("_mean", "")

        # Supply-Return temperature differential
        if "SW_TEMP" in col_base or "Supply" in col_base:
            # Look for matching return temperature
            rw_candidates = [
                c for c in mean_cols
                if ("RW_TEMP" in c or "Return" in c)
                and c != col
                # Match the same component prefix if possible
                and col_base.split("_")[0] in c
            ]
            for rw_col in rw_candidates:
                diff_name = f"{col_base}_minus_{rw_col.replace('_mean', '')}"
                result[diff_name] = df[col] - df[rw_col]

        # Setpoint deviation
        if "SPT" not in col_base and "Setpoint" not in col_base:
            spt_candidates = [
                c for c in mean_cols
                if ("SPT" in c or "Setpoint" in c or "Set_Point" in c)
                and col_base.split("_")[0] in c.replace("_mean", "")
            ]
            for spt_col in spt_candidates:
                dev_name = f"{col_base}_spt_deviation"
                result[dev_name] = df[col] - df[spt_col]

    # User-specified pairs
    if sensor_pairs:
        for col_a, col_b in sensor_pairs:
            if col_a in df.columns and col_b in df.columns:
                result[f"{col_a}_minus_{col_b}"] = df[col_a] - df[col_b]

    new_cols = len(result.columns) - len(df.columns)
    if new_cols > 0:
        logger.debug(f"Added {new_cols} cross-sensor features")

    return result


def build_node_features(
    system_df: pd.DataFrame,
    node_sensors: List[str],
    window_size: int = 15,
    stride: int = 15,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the full feature matrix for a single node.

    Combines windowed stats with cross-sensor features and returns the
    feature column names (excluding metadata and label columns).

    Args:
        system_df: Full system DataFrame with fault_label column.
        node_sensors: List of sensor column names assigned to this node.
        window_size: Aggregation window in minutes.
        stride: Window step size in minutes.

    Returns:
        Tuple of (feature DataFrame with labels, list of feature column names).
    """
    # Group by source file to maintain per-file fault labels
    all_features = []
    for source_file, group_df in system_df.groupby("source_file"):
        fault_label = group_df["fault_label"].iloc[0]
        fault_label_str = group_df["fault_label_str"].iloc[0]

        # Compute windowed features
        feat_df = compute_window_features(
            group_df.reset_index(drop=True),
            node_sensors,
            window_size=window_size,
            stride=stride,
        )

        if feat_df.empty:
            continue

        # Add cross-sensor features
        feat_df = extract_cross_sensor_features(feat_df)

        # Attach labels
        feat_df["fault_label"] = fault_label
        feat_df["fault_label_str"] = fault_label_str
        feat_df["source_file"] = source_file

        all_features.append(feat_df)

    if not all_features:
        return pd.DataFrame(), []

    combined = pd.concat(all_features, ignore_index=True)

    # Identify feature columns (exclude metadata and labels)
    meta_cols = {"fault_label", "fault_label_str", "source_file", "timestamp"}
    feature_cols = [
        c for c in combined.columns
        if c not in meta_cols and combined[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]
    ]

    # Drop features that are all NaN
    valid_features = []
    for col in feature_cols:
        if combined[col].notna().sum() > len(combined) * 0.1:
            valid_features.append(col)
        else:
            combined = combined.drop(columns=[col])

    # Fill remaining NaN with column median
    for col in valid_features:
        if combined[col].isna().any():
            combined[col] = combined[col].fillna(combined[col].median())

    logger.info(
        f"  Built features: {len(combined)} windows, "
        f"{len(valid_features)} features from {len(node_sensors)} sensors"
    )

    return combined, valid_features
