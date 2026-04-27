"""Repair no-fault scenario metadata in generated training artifacts.

This is a one-time migration for datasets produced by older code that cloned
fault scenarios into ``na_no_fault`` samples by only changing the fault label.
Those records keep faulted CSVs and root-cause paths, which breaks live-Oracle
RL. The script rewrites no-fault scenarios to use true baseline templates from
the same system and normalizes RL ground truth.
"""

import copy
import json
import os
import sys
from collections import Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)


DATA_DIR = Path("outputs/data")
ALL_SCENARIOS = DATA_DIR / "all_scenarios.json"
RL_SPLITS = [
    DATA_DIR / "rl_train.jsonl",
    DATA_DIR / "rl_val.jsonl",
    DATA_DIR / "rl_test.jsonl",
]
SFT_SPLITS = [
    DATA_DIR / "sft_train.jsonl",
]


def _is_no_fault(record):
    return (
        "no_fault" in str(record.get("scenario_type", ""))
        or str(record.get("fault_type", "")).lower() in ("normal", "no_fault")
        or str(record.get("root_cause_node", "")).lower() in ("none", "")
    )


def _is_clean_no_fault_template(record):
    path = record.get("diagnostic_path") or {}
    return (
        record.get("scenario_type") == "no_fault"
        and str(record.get("root_cause_node", "")).lower() in ("none", "")
        and str(record.get("fault_type", "")).lower() in ("normal", "no_fault")
        and path
    )


def _repair_scenario(record, template):
    repaired = dict(record)
    start = int(repaired.get("time_window_start") or template.get("time_window_start") or 0)
    t_start = int(template.get("time_window_start") or 0)
    t_end = int(template.get("time_window_end") or 0)
    window = max(1, t_end - t_start)
    if window <= 1:
        window = 15

    repaired["root_cause_node"] = "none"
    repaired["fault_type"] = "Normal"
    repaired["fault_intensity"] = "none"
    repaired["affected_systems"] = [template["root_cause_system"]]
    repaired["source_file"] = template.get("source_file", repaired.get("source_file", ""))
    repaired["time_window_start"] = start
    repaired["time_window_end"] = start + window
    repaired["difficulty"] = "easy"
    repaired["optimal_path"] = copy.deepcopy(template.get("optimal_path", []))

    path = copy.deepcopy(template.get("diagnostic_path") or {})
    if path:
        path["root_cause_node"] = "none"
        path["fault_type"] = "Normal"
        for node in path.get("nodes", []):
            if node.get("role") == "root_cause":
                node["role"] = "intermediate"
            node["abnormal_hint"] = "All parameters within normal operating ranges."
            node["direction"] = "check_component"
        repaired["diagnostic_path"] = path

    return repaired


def repair_all_scenarios():
    if not ALL_SCENARIOS.exists():
        print(f"[SKIP] {ALL_SCENARIOS} not found")
        return {}, Counter()

    scenarios = json.loads(ALL_SCENARIOS.read_text(encoding="utf-8"))
    templates = {}
    for record in scenarios:
        if _is_clean_no_fault_template(record):
            templates.setdefault(record["root_cause_system"], record)

    counts = Counter()
    repaired_by_id = {}
    output = []
    for record in scenarios:
        if record.get("scenario_type") == "na_no_fault":
            template = templates.get(record.get("root_cause_system"))
            if template:
                record = _repair_scenario(record, template)
                counts["na_no_fault_repaired"] += 1
            else:
                counts["na_no_fault_missing_template"] += 1
        elif record.get("scenario_type") == "no_fault":
            # Normalize the explicit baseline records too.
            record["root_cause_node"] = "none"
            record["fault_type"] = "Normal"
            record["fault_intensity"] = "none"
            counts["no_fault_normalized"] += 1

        repaired_by_id[record["scenario_id"]] = record
        output.append(record)

    ALL_SCENARIOS.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] repaired {ALL_SCENARIOS}: {dict(counts)}")
    return repaired_by_id, counts


def repair_rl_split(path, repaired_by_id):
    if not path.exists():
        print(f"[SKIP] {path} not found")
        return Counter()

    counts = Counter()
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            sid = entry.get("id") or entry.get("metadata", {}).get("scenario_id")
            scenario = repaired_by_id.get(sid)
            stype = entry.get("metadata", {}).get("scenario_type", "")
            gt = entry.setdefault("ground_truth", {})
            if "no_fault" in stype or _is_no_fault(gt):
                gt["root_cause_node"] = "none"
                gt["fault_type"] = "Normal"
                gt["fault_intensity"] = "none"
                if scenario:
                    gt["root_cause_system"] = scenario.get(
                        "root_cause_system", gt.get("root_cause_system", "")
                    )
                    gt["affected_systems"] = scenario.get(
                        "affected_systems", gt.get("affected_systems", [])
                    )
                    gt["optimal_path_length"] = (
                        (scenario.get("diagnostic_path") or {}).get("path_length")
                        or gt.get("optimal_path_length", 2)
                    )
                    entry.setdefault("metadata", {})["source_file"] = scenario.get(
                        "source_file", entry.get("metadata", {}).get("source_file", "")
                    )
                counts["no_fault_gt_repaired"] += 1
            rows.append(entry)

    with path.open("w", encoding="utf-8") as f:
        for entry in rows:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[OK] repaired {path}: {dict(counts)}")
    return counts


def repair_sft_split(path):
    if not path.exists():
        print(f"[SKIP] {path} not found")
        return Counter()

    counts = Counter()
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            entry = json.loads(line)
            meta = entry.setdefault("metadata", {})
            stype = meta.get("scenario_type", "")
            gt = meta.setdefault("ground_truth", {})
            if "no_fault" in stype or _is_no_fault(gt):
                gt["root_cause_node"] = "none"
                gt["fault_type"] = "no_fault"
                gt["fault_intensity"] = "none"
                counts["no_fault_gt_repaired"] += 1
            rows.append(entry)

    with path.open("w", encoding="utf-8") as f:
        for entry in rows:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"[OK] repaired {path}: {dict(counts)}")
    return counts


def main():
    repaired_by_id, _ = repair_all_scenarios()
    if repaired_by_id:
        for split in RL_SPLITS:
            repair_rl_split(split, repaired_by_id)
    for split in SFT_SPLITS:
        repair_sft_split(split)


if __name__ == "__main__":
    main()
