"""
Data Loader — Load and preprocess LBNL_FDD CSV data for node model training.

Handles:
  - Mapping CSV filenames to fault types and intensities
  - Loading fault-free and faulted data per system
  - NaN handling and basic cleaning
  - Creating labeled datasets (Normal vs fault classes)
"""

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.io_utils import setup_logger

logger = setup_logger(__name__)


@dataclass
class FaultFileInfo:
    """Metadata about a single fault CSV file."""
    filepath: str
    filename: str
    system_id: str
    fault_type: str          # e.g., "coolingtower_bias", "coi_stuck"
    fault_intensity: str     # e.g., "-2", "075", "Severe"
    is_fault_free: bool
    label: str               # Human-readable label for this fault scenario


# ============================================================================
# System-specific filename parsers
# ============================================================================

def _parse_chiller_filename(filename: str) -> Tuple[str, str, bool]:
    """Parse ChillerPlant CSV filenames to extract fault type and intensity."""
    if filename == "ChillerPlant.csv":
        return "normal", "none", True

    # Pattern: ChillerPlant_{component}_{faulttype}_{intensity}.csv
    name = filename.replace("ChillerPlant_", "").replace(".csv", "")
    parts = name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1], False
    return name, "unknown", False


def _parse_boiler_filename(filename: str) -> Tuple[str, str, bool]:
    """Parse BoilerPlant CSV filenames."""
    if filename == "BoilerPlant.csv":
        return "normal", "none", True
    name = filename.replace("BoilerPlant_", "").replace(".csv", "")
    parts = name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1], False
    return name, "unknown", False


def _parse_sdahu_filename(filename: str) -> Tuple[str, str, bool]:
    """Parse SDAHU CSV filenames."""
    if filename == "AHU_annual.csv":
        return "normal", "none", True
    name = filename.replace("_annual", "").replace(".csv", "")
    # Pattern: {fault}_{intensity}
    parts = name.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1], False
    return name, "unknown", False


def _parse_ddahu_filename(filename: str) -> Tuple[str, str, bool]:
    """Parse Dual-Duct AHU CSV filenames."""
    if filename == "DualDuct_FaultFree.csv":
        return "normal", "none", True
    name = filename.replace("DualDuct_", "").replace(".csv", "").replace("_", " ")
    # Try to extract intensity (last token if numeric or percentage)
    parts = name.rsplit(" ", 1)
    if len(parts) == 2 and (parts[1].replace("+", "").replace("-", "").replace(".", "")
                             .replace("inwg", "").replace("C", "").replace("%", "").isdigit()
                             or parts[1] in ("Minor", "Moderate", "Severe")):
        return parts[0].replace(" ", "_"), parts[1], False
    return name.replace(" ", "_"), "default", False


def _parse_fcu_filename(filename: str) -> Tuple[str, str, bool]:
    """Parse FCU CSV filenames."""
    if filename == "FCU_FaultFree.csv":
        return "normal", "none", True
    name = filename.replace("FCU_", "").replace(".csv", "")
    # Patterns: FCU_FaultType_Intensity or FCU_FaultType
    # Handle sensor bias: SensorBias_RMTemp_+2C
    if "SensorBias" in name:
        return name, "default", False
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and (parts[1].replace("%", "").replace(".", "").isdigit()
                             or parts[1] in ("Minor", "Moderate", "Severe")):
        return parts[0], parts[1], False
    return name, "default", False


def _parse_fpu_filename(filename: str, prefix: str) -> Tuple[str, str, bool]:
    """Parse FPU (PFPU/SFPU) CSV filenames."""
    if "FaultFree" in filename:
        return "normal", "none", True
    name = filename.replace(f"{prefix}_", "").replace(".csv", "")
    parts = name.rsplit("_", 1)
    if len(parts) == 2 and (parts[1].replace("%", "").replace("+", "").replace("-", "")
                             .replace("CFM", "").replace("C", "").replace(".", "").isdigit()
                             or parts[1] in ("Minor", "Moderate", "Severe")):
        return parts[0], parts[1], False
    return name, "default", False


def _parse_rtu_sim_filename(filename: str) -> Tuple[str, str, bool]:
    """Parse simulated RTU CSV filenames."""
    if filename == "RTU_sim_baseline.csv":
        return "normal", "none", True
    name = filename.replace("RTU_sim_", "").replace(".csv", "")
    # Pattern: faulttype + intensity (e.g., condfouling10)
    match = re.match(r"([a-z]+)(\d+)", name)
    if match:
        return match.group(1), match.group(2), False
    return name, "default", False


# ============================================================================
# Main Data Loader
# ============================================================================

# Maps system_id to (data subdirectory name, filename parser, CSV subdirectory)
SYSTEM_DATA_MAP = {
    "chiller_plant": {
        "subdir": "",
        "parser": _parse_chiller_filename,
        "csv_pattern": "ChillerPlant*.csv",
    },
    "boiler_plant": {
        "subdir": "LBNL_FDD_Dataset_Boiler_Plant",
        "parser": _parse_boiler_filename,
        "csv_pattern": "BoilerPlant*.csv",
    },
    "sdahu": {
        "subdir": "LBNL_FDD_Dataset_SDAHU",
        "parser": _parse_sdahu_filename,
        "csv_pattern": "*_annual*.csv",
    },
    "ddahu": {
        "subdir": "LBNL_FDD_Data_Sets_DDAHU",
        "parser": _parse_ddahu_filename,
        "csv_pattern": "DualDuct_*.csv",
    },
    "rtu": {
        "subdir": "Simulated RTU",
        "parser": _parse_rtu_sim_filename,
        "csv_pattern": "RTU_sim_*.csv",
    },
    "fcu": {
        "subdir": "LBNL_FDD_Dataset_FCU",
        "parser": _parse_fcu_filename,
        "csv_pattern": "FCU_*.csv",
    },
    "pfpu": {
        "subdir": "LBNL_FDD_Data_Sets_PFPU",
        "parser": lambda f: _parse_fpu_filename(f, "PFPU"),
        "csv_pattern": "PFPU_*.csv",
    },
    "sfpu": {
        "subdir": "LBNL_FDD_Data_Sets_SFPU",
        "parser": lambda f: _parse_fpu_filename(f, "SFPU"),
        "csv_pattern": "SFPU_*.csv",
    },
}


def discover_fault_files(data_root: str, system_id: str) -> List[FaultFileInfo]:
    """
    Discover all CSV files for a system and parse their fault metadata.

    Args:
        data_root: Path to data/lbnl/ directory.
        system_id: System identifier (e.g., "chiller_plant").

    Returns:
        List of FaultFileInfo objects.
    """
    from src.utils.io_utils import load_yaml

    # Determine the system's data directory
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "configs", "topology_config.yaml"
    )
    config = load_yaml(config_path)
    sys_config = config["systems"].get(system_id, {})
    base_dir = os.path.join(data_root, sys_config.get("data_dir", ""))

    sys_map = SYSTEM_DATA_MAP.get(system_id)
    if sys_map is None:
        logger.warning(f"No data map for system '{system_id}'")
        return []

    csv_dir = os.path.join(base_dir, sys_map["subdir"]) if sys_map["subdir"] else base_dir

    if not os.path.isdir(csv_dir):
        logger.warning(f"CSV directory not found: {csv_dir}")
        return []

    files = []
    for fname in sorted(os.listdir(csv_dir)):
        if not fname.endswith(".csv"):
            continue

        fpath = os.path.join(csv_dir, fname)
        fault_type, intensity, is_ff = sys_map["parser"](fname)

        label = "Normal" if is_ff else f"{fault_type}_{intensity}"

        files.append(FaultFileInfo(
            filepath=fpath,
            filename=fname,
            system_id=system_id,
            fault_type=fault_type,
            fault_intensity=intensity,
            is_fault_free=is_ff,
            label=label,
        ))

    logger.info(
        f"Discovered {len(files)} CSV files for '{system_id}' "
        f"({sum(1 for f in files if f.is_fault_free)} fault-free, "
        f"{sum(1 for f in files if not f.is_fault_free)} faulted)"
    )
    return files


def load_system_data(
    data_root: str,
    system_id: str,
    max_rows_per_file: Optional[int] = None,
    sample_frac: Optional[float] = None,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load all CSV files for a system into a single labeled DataFrame.

    Args:
        data_root: Path to data/lbnl/ directory.
        system_id: System identifier.
        max_rows_per_file: If set, limit rows loaded per file.
        sample_frac: If set, randomly sample this fraction from each file.
        random_state: Random seed for sampling.

    Returns:
        Tuple of (DataFrame with all data + 'fault_label' column,
                  dict mapping label strings to integer codes).
    """
    files = discover_fault_files(data_root, system_id)
    if not files:
        return pd.DataFrame(), {}

    dfs = []
    label_map = {"Normal": 0}
    label_counter = 1

    for finfo in files:
        logger.info(f"  Loading {finfo.filename} ({finfo.label})...")
        try:
            df = pd.read_csv(
                finfo.filepath,
                nrows=max_rows_per_file,
                low_memory=False,
            )
        except Exception as e:
            logger.error(f"  Failed to load {finfo.filename}: {e}")
            continue

        if sample_frac and sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=random_state)

        # Assign fault label
        if finfo.label not in label_map:
            label_map[finfo.label] = label_counter
            label_counter += 1
        df["fault_label"] = label_map[finfo.label]
        df["fault_label_str"] = finfo.label
        df["source_file"] = finfo.filename

        dfs.append(df)

    if not dfs:
        return pd.DataFrame(), {}

    combined = pd.concat(dfs, ignore_index=True)

    # Basic cleaning: handle NaN
    # Drop columns that are entirely NaN
    nan_cols = combined.columns[combined.isna().all()]
    if len(nan_cols) > 0:
        logger.info(f"  Dropping {len(nan_cols)} all-NaN columns")
        combined = combined.drop(columns=nan_cols)

    # Forward-fill short NaN gaps, drop remaining
    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    combined[numeric_cols] = combined[numeric_cols].ffill(limit=5)

    logger.info(
        f"Loaded {len(combined)} rows for '{system_id}' with "
        f"{len(label_map)} classes: {label_map}"
    )
    return combined, label_map
