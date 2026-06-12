"""
Feature Engineer — Extract per-node features from sensor time-series data.

Creates windowed statistical features from raw 1-minute sensor data:
  - Window aggregates: mean, std, min, max, range, slope
  - Cross-sensor features: temperature differentials, setpoint deviations
  - Temporal features: hour-of-day, day-of-week encoding
"""

import re
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.io_utils import setup_logger

logger = setup_logger(__name__)


SENSOR_COLUMN_ALIASES: Dict[str, List[str]] = {
    # Common Brick semantic names used by the FCU TTL versus LBNL FCU CSV names.
    "Mode_Command": ["FCU_CTRL", "CTRL", "MODE"],
    "Speed_Status": ["FCU_SPD", "FAN_CTRL", "FAN_SPEED"],
    "Zone_Air_Temperature_Sensor": ["RM_TEMP", "RMTEMP", "ZONE_TEMP"],
    "Zone_Air_Cooling_Temperature_Setpoint": ["RMCLGSPT", "CLGSPT", "CLG_SPT"],
    "Zone_Air_Heating_Temperature_Setpoint": ["RMHTGSPT", "HTGSPT", "HTG_SPT"],
    "Supply_Air_Temperature_Sensor": ["SA_TEMP", "DAT", "FCU_DAT"],
    "Discharge_Air_Temperature_Sensor": ["DAT", "FCU_DAT", "SA_TEMP"],
    "Return_Air_Temperature_Sensor": ["RA_TEMP", "RAT", "FCU_RAT"],
    "Mixed_Air_Temperature_Sensor": ["MA_TEMP", "MAT", "FCU_MAT"],
    "Outside_Air_Temperature_Sensor": ["OA_TEMP", "OAT", "FCU_OAT"],
    "Cooling_Valve_Command": ["CVLV", "FCU_CVLV"],
    "Heating_Valve_Command": ["HVLV", "FCU_HVLV"],
    "Damper_Command": ["DMPR", "FCU_DMPR"],
    "Discharge_Air_Flow_Sensor": ["DA_CFM", "FCU_DA_CFM"],
    "Outside_Air_Flow_Sensor": ["OA_CFM", "FCU_OA_CFM"],
}


def _norm_col_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _parse_timestamps_fast(series: pd.Series) -> pd.Series:
    """Parse common LBNL timestamp formats without pandas mixed-format slow path."""
    if np.issubdtype(series.dtype, np.datetime64):
        return pd.to_datetime(series, errors="coerce")

    sample = ""
    for value in series.head(20):
        if pd.notna(value) and str(value).strip():
            sample = str(value).strip()
            break

    fmt = None
    if re.match(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$", sample):
        fmt = "%Y-%m-%d %H:%M:%S"
    elif re.match(r"^\d{1,2}/\d{1,2}/\d{4} \d{2}:\d{2}$", sample):
        fmt = "%m/%d/%Y %H:%M"

    if fmt:
        return pd.to_datetime(series, format=fmt, errors="coerce")
    return pd.to_datetime(series, errors="coerce")


def resolve_sensor_columns(
    df: pd.DataFrame,
    sensor_cols: List[str],
) -> List[str]:
    """Resolve topology sensor names to actual CSV columns.

    Most LBNL TTL files use names that match CSV columns directly.  A few
    systems, especially FCU, use Brick semantic point names in TTL and compact
    engineering abbreviations in CSV.  This resolver preserves exact matches
    first, then applies curated aliases and conservative normalized matching.
    """
    resolved: List[str] = []
    seen = set()
    columns = list(df.columns)
    col_by_norm = {_norm_col_name(c): c for c in columns}

    def _add(col: str) -> bool:
        if col in df.columns and col not in seen:
            resolved.append(col)
            seen.add(col)
            return True
        return False

    for sensor in sensor_cols:
        if _add(sensor):
            continue

        sensor_local = str(sensor).split("::")[-1]
        if _add(sensor_local):
            continue

        norm_sensor = _norm_col_name(sensor_local)
        if norm_sensor in col_by_norm and _add(col_by_norm[norm_sensor]):
            continue

        aliases = SENSOR_COLUMN_ALIASES.get(sensor_local, [])
        aliases += SENSOR_COLUMN_ALIASES.get(str(sensor), [])
        for alias in aliases:
            if _add(alias):
                break
            norm_alias = _norm_col_name(alias)
            if norm_alias in col_by_norm and _add(col_by_norm[norm_alias]):
                break
        else:
            # Conservative final pass: only accept substring matches for
            # reasonably specific identifiers to avoid mapping generic words
            # like "sensor" or "status" to unrelated columns.
            if len(norm_sensor) >= 6:
                for col in columns:
                    norm_col = _norm_col_name(col)
                    if norm_sensor in norm_col or norm_col in norm_sensor:
                        if _add(col):
                            break

    return resolved


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
    # Filter/resolve to existing sensor columns.
    available = resolve_sensor_columns(df, sensor_cols)
    if not available:
        logger.warning(
            "No matching sensor columns found in DataFrame "
            "(requested=%s, available_sample=%s)",
            sensor_cols[:8],
            list(df.columns[:8]),
        )
        return pd.DataFrame()

    # Extract numeric data for the relevant sensors.  Keep this as a compact
    # NumPy matrix because wide systems such as DDAHU otherwise spend most of
    # their time in pandas/object conversion and temporary DataFrame blocks.
    raw_sensor_data = df[available]
    try:
        values = raw_sensor_data.to_numpy(
            dtype=np.float32, na_value=np.nan, copy=False
        )
    except TypeError:
        try:
            values = raw_sensor_data.to_numpy(dtype=np.float32, copy=False)
        except (TypeError, ValueError):
            values = raw_sensor_data.apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=np.float32, copy=True)
    except ValueError:
        values = raw_sensor_data.apply(
            pd.to_numeric, errors="coerce"
        ).to_numpy(dtype=np.float32, copy=True)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    # Parse timestamp if available (for temporal features)
    time_col = df.columns[0]
    timestamps = _parse_timestamps_fast(df[time_col])

    n_rows = len(values)
    if n_rows < window_size:
        return pd.DataFrame()

    starts = np.arange(0, n_rows - window_size + 1, stride, dtype=np.int64)
    if len(starts) == 0:
        return pd.DataFrame()

    # Vectorized window aggregation without materializing a giant
    # (n_windows, window_size, n_sensors) tensor.  This keeps the same feature
    # schema as the original implementation but loops over the small window
    # length only.  It is much faster and less memory-hungry for wide systems
    # such as DDAHU.
    n_windows = len(starts)
    n_sensors = len(available)
    shape = (n_windows, n_sensors)
    count = np.zeros(shape, dtype=np.float64)
    sum_y = np.zeros(shape, dtype=np.float64)
    sum_y2 = np.zeros(shape, dtype=np.float64)
    min_vals = np.full(shape, np.inf, dtype=np.float64)
    max_vals = np.full(shape, -np.inf, dtype=np.float64)
    sum_x = np.zeros(shape, dtype=np.float64)
    sum_x2 = np.zeros(shape, dtype=np.float64)
    sum_xy = np.zeros(shape, dtype=np.float64)

    for offset in range(window_size):
        vals = values[starts + offset, :]
        finite = np.isfinite(vals)
        clean = np.where(finite, vals, 0.0).astype(np.float64, copy=False)
        count += finite
        sum_y += clean
        sum_y2 += clean * clean
        min_vals = np.minimum(min_vals, np.where(finite, vals, np.inf))
        max_vals = np.maximum(max_vals, np.where(finite, vals, -np.inf))
        sum_x += finite * offset
        sum_x2 += finite * (offset * offset)
        sum_xy += clean * offset

    mean = np.divide(
        sum_y,
        count,
        out=np.full_like(sum_y, np.nan, dtype=np.float64),
        where=count > 0,
    )
    var = np.divide(
        sum_y2 - np.divide(sum_y * sum_y, count, out=np.zeros_like(sum_y), where=count > 0),
        count - 1,
        out=np.zeros_like(sum_y, dtype=np.float64),
        where=count > 1,
    )
    var = np.maximum(var, 0.0)
    std = np.sqrt(var)

    min_vals[count == 0] = np.nan
    max_vals[count == 0] = np.nan
    ranges = max_vals - min_vals
    ranges[count == 0] = 0.0

    denom = count * sum_x2 - sum_x * sum_x
    slope = np.divide(
        count * sum_xy - sum_x * sum_y,
        denom,
        out=np.zeros_like(sum_y, dtype=np.float64),
        where=(count > 2) & (np.abs(denom) > 1e-12),
    )

    feature_data: Dict[str, np.ndarray] = {}

    centers = starts + window_size // 2
    center_ts = timestamps.iloc[centers].reset_index(drop=True)
    valid_ts = center_ts.notna()
    feature_data["hour_of_day"] = np.where(valid_ts, center_ts.dt.hour, 0).astype(np.int16)
    feature_data["day_of_week"] = np.where(valid_ts, center_ts.dt.dayofweek, 0).astype(np.int16)
    occupied = (
        valid_ts
        & center_ts.dt.hour.between(6, 20)
        & (center_ts.dt.dayofweek < 6)
    )
    feature_data["is_occupied"] = occupied.astype(np.int8).to_numpy()

    for j, col in enumerate(available):
        prefix = col
        feature_data[f"{prefix}_mean"] = mean[:, j].astype(np.float32)
        feature_data[f"{prefix}_std"] = std[:, j].astype(np.float32)
        feature_data[f"{prefix}_min"] = min_vals[:, j].astype(np.float32)
        feature_data[f"{prefix}_max"] = max_vals[:, j].astype(np.float32)
        feature_data[f"{prefix}_range"] = ranges[:, j].astype(np.float32)
        feature_data[f"{prefix}_slope"] = slope[:, j].astype(np.float32)

    return pd.DataFrame(feature_data)


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
    derived = {}
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
                derived[diff_name] = df[col] - df[rw_col]

        # Setpoint deviation
        if "SPT" not in col_base and "Setpoint" not in col_base:
            spt_candidates = [
                c for c in mean_cols
                if ("SPT" in c or "Setpoint" in c or "Set_Point" in c)
                and col_base.split("_")[0] in c.replace("_mean", "")
            ]
            for spt_col in spt_candidates:
                dev_name = f"{col_base}_spt_deviation"
                derived[dev_name] = df[col] - df[spt_col]

    # User-specified pairs
    if sensor_pairs:
        for col_a, col_b in sensor_pairs:
            if col_a in df.columns and col_b in df.columns:
                derived[f"{col_a}_minus_{col_b}"] = df[col_a] - df[col_b]

    if derived:
        logger.debug(f"Added {len(derived)} cross-sensor features")
        return pd.concat([df.copy(), pd.DataFrame(derived, index=df.index)], axis=1)

    return df.copy()


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
    resolved_sensors = resolve_sensor_columns(system_df, node_sensors)
    if not resolved_sensors:
        logger.warning(
            "No matching sensor columns found in DataFrame "
            "(requested=%s, available_sample=%s)",
            node_sensors[:8],
            list(system_df.columns[:8]),
        )
        return pd.DataFrame(), []
    if resolved_sensors != node_sensors:
        logger.info(
            "  Resolved %d/%d topology sensors to CSV columns: %s",
            len(resolved_sensors),
            len(node_sensors),
            resolved_sensors[:8],
        )

    all_features = []
    for source_file, group_df in system_df.groupby("source_file", sort=False):
        fault_label = group_df["fault_label"].iloc[0]
        fault_label_str = group_df["fault_label_str"].iloc[0]
        n_windows = max(0, (len(group_df) - window_size) // stride + 1)
        started = time.time()
        logger.info(
            "  Feature source %s: rows=%d, windows=%d, sensors=%d",
            source_file,
            len(group_df),
            n_windows,
            len(resolved_sensors),
        )

        # Compute windowed features
        feat_df = compute_window_features(
            group_df.reset_index(drop=True),
            resolved_sensors,
            window_size=window_size,
            stride=stride,
        )

        if feat_df.empty:
            continue
        logger.info(
            "  Feature source %s complete: %d windows in %.1fs",
            source_file,
            len(feat_df),
            time.time() - started,
        )

        # Add cross-sensor features
        feat_df = extract_cross_sensor_features(feat_df)

        # Attach metadata in one concat to keep the feature frame contiguous.
        metadata = pd.DataFrame(
            {
                "fault_label": fault_label,
                "fault_label_str": fault_label_str,
                "source_file": source_file,
            },
            index=feat_df.index,
        )
        feat_df = pd.concat([feat_df.copy(), metadata], axis=1)

        all_features.append(feat_df)

    if not all_features:
        return pd.DataFrame(), []

    combined = pd.concat(all_features, ignore_index=True).copy()

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
        f"{len(valid_features)} features from {len(resolved_sensors)} sensors"
    )

    return combined, valid_features
