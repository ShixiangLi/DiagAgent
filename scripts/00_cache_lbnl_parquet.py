"""
Build a Parquet cache for LBNL_FDD raw CSV files.

The cache is a lossless IO optimization used by model training, data
generation, RL rollout state loading, and evaluation. It does not sample,
resample, relabel, or change feature engineering.
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.node_models.data_loader import (
    DEFAULT_PARQUET_CACHE_DIR,
    FaultFileInfo,
    discover_fault_files,
    get_parquet_cache_path,
)
from src.utils.io_utils import ensure_dir, save_json, setup_logger

logger = setup_logger("cache_lbnl_parquet")

ALL_SYSTEMS = [
    "chiller_plant",
    "boiler_plant",
    "sdahu",
    "ddahu",
    "rtu",
    "fcu",
    "pfpu",
    "sfpu",
]


def _cache_is_fresh(finfo: FaultFileInfo, parquet_path: str) -> bool:
    if not os.path.exists(parquet_path):
        return False
    try:
        return os.path.getmtime(parquet_path) >= os.path.getmtime(finfo.filepath)
    except OSError:
        return False


def _convert_one(
    finfo: FaultFileInfo,
    cache_dir: str,
    compression: str,
    overwrite: bool,
    validate: bool,
) -> dict:
    parquet_path = get_parquet_cache_path(finfo, cache_dir)
    if not overwrite and _cache_is_fresh(finfo, parquet_path):
        return {
            "system_id": finfo.system_id,
            "filename": finfo.filename,
            "status": "skipped_fresh",
            "parquet_path": parquet_path,
        }

    started = time.time()
    out_path = Path(parquet_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")

    df = pd.read_csv(finfo.filepath, low_memory=False)
    n_rows, n_cols = df.shape
    df.to_parquet(
        tmp_path,
        engine="pyarrow",
        compression=compression,
        index=False,
    )
    os.replace(tmp_path, out_path)

    if validate:
        cached = pd.read_parquet(out_path)
        if cached.shape != (n_rows, n_cols):
            raise RuntimeError(
                f"Shape mismatch for {finfo.filename}: "
                f"csv={(n_rows, n_cols)} parquet={cached.shape}"
            )

    return {
        "system_id": finfo.system_id,
        "filename": finfo.filename,
        "status": "converted",
        "rows": int(n_rows),
        "columns": int(n_cols),
        "source_path": finfo.filepath,
        "parquet_path": parquet_path,
        "elapsed_sec": round(time.time() - started, 3),
        "source_size_mb": round(os.path.getsize(finfo.filepath) / 1024 / 1024, 3),
        "parquet_size_mb": round(os.path.getsize(out_path) / 1024 / 1024, 3),
        "compression": compression,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Cache LBNL CSV files as Parquet")
    parser.add_argument("--data-root", default="data/lbnl")
    parser.add_argument("--cache-dir", default=DEFAULT_PARQUET_CACHE_DIR)
    parser.add_argument("--systems", default=None, help="Comma-separated system IDs")
    parser.add_argument("--compression", default="zstd", choices=["zstd", "snappy", "gzip", "brotli", "none"])
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument(
        "--max-files-per-system",
        type=int,
        default=None,
        help="Debug/smoke option. Converts only the first N files per system.",
    )
    args = parser.parse_args()

    compression = None if args.compression == "none" else args.compression
    systems = args.systems.split(",") if args.systems else ALL_SYSTEMS
    ensure_dir(args.cache_dir)

    files = []
    for system_id in systems:
        discovered = discover_fault_files(args.data_root, system_id)
        if args.max_files_per_system is not None:
            discovered = discovered[: args.max_files_per_system]
        files.extend(discovered)

    logger.info(
        "Converting %s CSV files to Parquet cache: %s",
        len(files),
        args.cache_dir,
    )
    logger.info(
        "Options: compression=%s, workers=%s, overwrite=%s, validate=%s",
        args.compression,
        args.workers,
        args.overwrite,
        args.validate,
    )

    results = []
    errors = []
    max_workers = max(1, int(args.workers or 1))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(
                _convert_one,
                finfo,
                args.cache_dir,
                compression,
                args.overwrite,
                args.validate,
            ): finfo
            for finfo in files
        }
        for future in as_completed(future_map):
            finfo = future_map[future]
            try:
                result = future.result()
                results.append(result)
                logger.info(
                    "  [%s] %s/%s",
                    result["status"],
                    finfo.system_id,
                    finfo.filename,
                )
            except Exception as exc:
                err = {
                    "system_id": finfo.system_id,
                    "filename": finfo.filename,
                    "status": "error",
                    "error": str(exc),
                }
                errors.append(err)
                logger.error("  [error] %s/%s: %s", finfo.system_id, finfo.filename, exc)

    summary = {
        "cache_dir": args.cache_dir,
        "total_files": len(files),
        "converted": sum(1 for r in results if r.get("status") == "converted"),
        "skipped_fresh": sum(1 for r in results if r.get("status") == "skipped_fresh"),
        "errors": len(errors),
        "systems": systems,
        "results": sorted(results, key=lambda r: (r.get("system_id", ""), r.get("filename", ""))),
        "error_details": errors,
    }
    summary_path = os.path.join(args.cache_dir, "manifest.json")
    save_json(summary, summary_path)
    logger.info("Parquet cache summary saved to %s", summary_path)
    if errors:
        logger.error("Parquet cache completed with %s errors", len(errors))
        return 1
    logger.info("Parquet cache ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
