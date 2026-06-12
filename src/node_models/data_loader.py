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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils.io_utils import setup_logger

logger = setup_logger(__name__)

DEFAULT_PARQUET_CACHE_DIR = os.environ.get(
    "DIAGAGENT_PARQUET_CACHE",
    "outputs/cache/parquet",
)


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


def get_parquet_cache_path(
    finfo: FaultFileInfo,
    cache_dir: Optional[str] = None,
) -> str:
    """Return the deterministic Parquet cache path for a raw LBNL CSV file."""
    root = Path(cache_dir or DEFAULT_PARQUET_CACHE_DIR)
    return str(root / finfo.system_id / f"{Path(finfo.filename).stem}.parquet")


def _is_parquet_cache_fresh(finfo: FaultFileInfo, parquet_path: str) -> bool:
    """A cache is fresh if it exists and is not older than its source CSV."""
    if not os.path.exists(parquet_path):
        return False
    trust_cache = os.environ.get("DIAGAGENT_TRUST_PARQUET_CACHE", "").lower()
    if trust_cache in {"1", "true", "yes"}:
        return True
    if not os.path.exists(finfo.filepath):
        # Cache-only training deployments intentionally omit raw CSV files.
        return True
    try:
        return os.path.getmtime(parquet_path) >= os.path.getmtime(finfo.filepath)
    except OSError:
        return False


def get_fault_file_read_source(
    finfo: FaultFileInfo,
    cache_dir: Optional[str] = None,
    use_parquet_cache: bool = True,
) -> Tuple[str, str]:
    """Return the storage source that read_fault_file will try first."""
    disable_env = os.environ.get("DIAGAGENT_DISABLE_PARQUET", "").lower()
    parquet_enabled = use_parquet_cache and disable_env not in {"1", "true", "yes"}
    parquet_path = get_parquet_cache_path(finfo, cache_dir)
    if parquet_enabled and _is_parquet_cache_fresh(finfo, parquet_path):
        return "parquet", parquet_path
    return "csv", finfo.filepath


def get_fault_file_row_count(
    finfo: FaultFileInfo,
    cache_dir: Optional[str] = None,
    use_parquet_cache: bool = True,
) -> int:
    """Return the number of data rows in a fault file without loading it all."""
    source, source_path = get_fault_file_read_source(
        finfo,
        cache_dir=cache_dir,
        use_parquet_cache=use_parquet_cache,
    )
    if source == "parquet":
        try:
            import pyarrow.parquet as pq

            return int(pq.ParquetFile(source_path).metadata.num_rows)
        except Exception:
            return int(pd.read_parquet(source_path).shape[0])

    rows = 0
    with open(source_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            rows += chunk.count(b"\n")
    # CSV row count excludes the header.  If the final line has no trailing
    # newline this may under-count by one; the conservative value is safer for
    # window sampling.
    return max(0, rows - 1)


def _parquet_stub_source_path(system_id: str, filename: str) -> str:
    """Return a stable placeholder source path for cache-only deployments."""
    return os.path.join(
        "parquet_cache_only",
        system_id,
        f"{Path(filename).stem}.csv",
    )


def _discover_fault_files_from_parquet_cache(
    system_id: str,
    csv_pattern: str,
    parser,
) -> List[FaultFileInfo]:
    """Discover file metadata from Parquet cache when raw CSV files are absent."""
    cache_dir = Path(DEFAULT_PARQUET_CACHE_DIR) / system_id
    if not cache_dir.is_dir():
        return []

    pattern = "^" + re.escape(csv_pattern).replace(r"\*", ".*").replace(r"\?", ".") + "$"
    regex = re.compile(pattern, re.IGNORECASE)
    files: List[FaultFileInfo] = []
    for parquet_path in sorted(cache_dir.glob("*.parquet")):
        fname = f"{parquet_path.stem}.csv"
        if not regex.match(fname):
            continue
        fault_type, intensity, is_ff = parser(fname)
        label = "Normal" if is_ff else f"{fault_type}_{intensity}"
        files.append(FaultFileInfo(
            filepath=_parquet_stub_source_path(system_id, fname),
            filename=fname,
            system_id=system_id,
            fault_type=fault_type,
            fault_intensity=intensity,
            is_fault_free=is_ff,
            label=label,
        ))
    if files:
        logger.info(
            "Discovered %s Parquet cache files for '%s' "
            "(%s fault-free, %s faulted)",
            len(files),
            system_id,
            sum(1 for f in files if f.is_fault_free),
            sum(1 for f in files if not f.is_fault_free),
        )
    return files


def _read_parquet(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    """Read a Parquet file, honoring nrows without materializing the full file."""
    if nrows is None:
        return pd.read_parquet(path)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        parquet_file = pq.ParquetFile(path)
        batches = []
        remaining = max(int(nrows), 0)
        if remaining == 0:
            return pd.DataFrame()
        for batch in parquet_file.iter_batches(batch_size=min(remaining, 65536)):
            if remaining <= 0:
                break
            if batch.num_rows > remaining:
                batch = batch.slice(0, remaining)
            batches.append(batch)
            remaining -= batch.num_rows
        if not batches:
            return pd.DataFrame()
        return pa.Table.from_batches(batches).to_pandas()
    except Exception:
        # Fallback for environments where pyarrow batch APIs differ.
        return pd.read_parquet(path).head(nrows)


def read_fault_file(
    finfo: FaultFileInfo,
    nrows: Optional[int] = None,
    cache_dir: Optional[str] = None,
    use_parquet_cache: bool = True,
    numeric_only: bool = False,
) -> pd.DataFrame:
    """Read one LBNL fault file, preferring the Parquet cache when available.

    The Parquet cache is a lossless storage optimization. It must not change
    row order, sampling, labels, or feature engineering behavior.
    """
    df: pd.DataFrame
    source, source_path = get_fault_file_read_source(
        finfo,
        cache_dir=cache_dir,
        use_parquet_cache=use_parquet_cache,
    )

    if source == "parquet":
        try:
            df = _read_parquet(source_path, nrows=nrows)
            logger.debug("  Loaded Parquet cache: %s", source_path)
        except Exception as exc:
            logger.warning(
                "  Failed to load Parquet cache for %s, falling back to CSV: %s",
                finfo.filename,
                exc,
            )
            df = pd.read_csv(finfo.filepath, nrows=nrows, low_memory=False)
    else:
        df = pd.read_csv(finfo.filepath, nrows=nrows, low_memory=False)

    if numeric_only:
        df = df.select_dtypes(include=["number"])
    return df


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

    files = []
    if os.path.isdir(csv_dir):
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
    if not files:
        files = _discover_fault_files_from_parquet_cache(
            system_id,
            sys_map["csv_pattern"],
            sys_map["parser"],
        )
        if not files:
            logger.warning(f"CSV files and Parquet cache not found for {system_id}: {csv_dir}")
            return []

    logger.info(
        f"Discovered {len(files)} data files for '{system_id}' "
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
    parquet_cache_dir: Optional[str] = None,
    use_parquet_cache: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Load all CSV files for a system into a single labeled DataFrame.

    Args:
        data_root: Path to data/lbnl/ directory.
        system_id: System identifier.
        max_rows_per_file: If set, limit rows loaded per file.
        sample_frac: If set, randomly sample this fraction from each file.
        random_state: Random seed for sampling.
        parquet_cache_dir: Optional Parquet cache root.
        use_parquet_cache: Prefer cached Parquet files when available.

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
        source, source_path = get_fault_file_read_source(
            finfo,
            cache_dir=parquet_cache_dir,
            use_parquet_cache=use_parquet_cache,
        )
        logger.info(
            "  Loading %s (%s) via %s: %s",
            finfo.filename,
            finfo.label,
            source,
            source_path,
        )
        try:
            df = read_fault_file(
                finfo,
                nrows=max_rows_per_file,
                cache_dir=parquet_cache_dir,
                use_parquet_cache=use_parquet_cache,
            )
        except Exception as e:
            logger.error(f"  Failed to load {finfo.filename}: {e}")
            continue

        if sample_frac and sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=random_state)

        # Assign fault metadata in one concat.  Some LBNL CSVs have many
        # columns; repeated scalar inserts fragment the frame badly and slow
        # server-side model training.
        if finfo.label not in label_map:
            label_map[finfo.label] = label_counter
            label_counter += 1
        metadata = pd.DataFrame(
            {
                "fault_label": label_map[finfo.label],
                "fault_label_str": finfo.label,
                "source_file": finfo.filename,
            },
            index=df.index,
        )
        df = pd.concat([df.copy(), metadata], axis=1)

        dfs.append(df)

    if not dfs:
        return pd.DataFrame(), {}

    combined = pd.concat(dfs, ignore_index=True).copy()

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
