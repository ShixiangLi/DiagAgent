"""Local readiness checks for cloud SFT/RL training.

This script is intentionally CPU/local-data oriented. It does not run full
SFT or RL training. It validates the assets that must be synced to the cloud:
topology, Oracle model files, scenario data, SFT/RL datasets, reward metrics,
and historical training behavior.

Usage:
    python scripts/local_training_readiness.py
    python scripts/local_training_readiness.py --require-oracle-runtime
"""

import argparse
import importlib.util
import json
import logging
import os
import re
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)


DATA_DIR = Path("outputs/data")
TOPOLOGY_JSON = Path("outputs/topology/topology.json")
MODELS_DIR = Path("outputs/models")

EXPECTED_TOPOLOGY = {
    "nodes": 624,
    "edges": 671,
    "levels": {0: 1, 1: 8, 2: 129, 3: 486},
    "cross_system_edges": 10,
}

EXPECTED_ALL_SCENARIOS = {
    "single_system": 1230,
    "cross_system": 200,
    "a_single_system": 1230,
    "a_cross_system": 200,
    "na_low_confidence": 1230,
    "a_low_confidence": 1230,
    "na_no_fault": 1230,
    "no_fault": 24,
}

EXPECTED_RL_SPLITS = {
    "rl_train.jsonl": 5259,
    "rl_val.jsonl": 657,
    "rl_test.jsonl": 658,
}

LOCAL_RUNTIME_DEPS = [
    "numpy",
    "pandas",
    "sklearn",
    "networkx",
    "yaml",
    "rdflib",
    "imblearn",
    "lightgbm",
]

CLOUD_LLM_DEPS = [
    "torch",
    "transformers",
    "peft",
    "datasets",
    "accelerate",
]


class Reporter:
    def __init__(self) -> None:
        self.failures: List[str] = []
        self.warnings: List[str] = []

    def header(self, title: str) -> None:
        print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")

    def pass_(self, message: str) -> None:
        print(f"  [PASS] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"  [WARN] {message}")

    def fail(self, message: str) -> None:
        self.failures.append(message)
        print(f"  [FAIL] {message}")

    def check(self, condition: bool, ok: str, bad: str, fatal: bool = True) -> None:
        if condition:
            self.pass_(ok)
        elif fatal:
            self.fail(bad)
        else:
            self.warn(bad)


def _has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _is_baseline_file(name: str) -> bool:
    lower = str(name or "").lower()
    return any(token in lower for token in ("baseline", "faultfree", "fault_free"))


def _fault_free_files() -> set:
    """Return known fault-free CSVs as (system_id, filename) pairs."""
    try:
        from src.node_models.data_loader import discover_fault_files
        from src.utils.io_utils import load_yaml

        logging.getLogger("src.node_models.data_loader").setLevel(logging.WARNING)
        config = load_yaml("configs/topology_config.yaml")
        pairs = set()
        for system_id in config.get("systems", {}):
            for info in discover_fault_files("data/lbnl", system_id):
                if info.is_fault_free:
                    pairs.add((system_id, info.filename))
        return pairs
    except Exception:
        return set()


def _is_fault_free_source(system_id: str, filename: str, fault_free_files: set) -> bool:
    if not filename:
        return True
    if fault_free_files:
        return (system_id, filename) in fault_free_files
    return _is_baseline_file(filename)


def _is_no_fault_gt(gt: Dict[str, Any], scenario_type: str = "") -> bool:
    node = str(gt.get("root_cause_node", "")).lower()
    fault = str(gt.get("fault_type", "")).lower()
    return (
        "no_fault" in str(scenario_type).lower()
        or node in ("none", "")
        or fault in ("normal", "no_fault", "no fault", "none", "")
    )


def _clean_no_fault_gt(gt: Dict[str, Any]) -> bool:
    node = str(gt.get("root_cause_node", "")).lower()
    fault = str(gt.get("fault_type", "")).lower()
    intensity = str(gt.get("fault_intensity", "")).lower()
    return (
        node in ("none", "")
        and fault in ("normal", "no_fault", "no fault")
        and intensity in ("none", "")
    )


def check_dependencies(r: Reporter, require_oracle_runtime: bool) -> None:
    r.header("1. Dependency Boundary")
    missing_local = [m for m in LOCAL_RUNTIME_DEPS if not _has_module(m)]
    missing_cloud = [m for m in CLOUD_LLM_DEPS if not _has_module(m)]

    r.check(
        not missing_local,
        "Local topology/Oracle dependencies are importable",
        "Missing local topology/Oracle dependencies: " + ", ".join(missing_local),
        fatal=require_oracle_runtime,
    )
    if missing_cloud:
        r.warn(
            "Cloud-only LLM dependencies missing locally: "
            + ", ".join(missing_cloud)
        )
    else:
        r.pass_("Cloud LLM dependency set is importable in this environment")


def check_topology(r: Reporter) -> None:
    r.header("2. Topology Artifact")
    if not TOPOLOGY_JSON.exists():
        r.fail(f"Topology artifact missing: {TOPOLOGY_JSON}")
        return

    topology = _load_json(TOPOLOGY_JSON)
    nodes = topology.get("nodes", [])
    edges = topology.get("edges", [])
    level_counts = Counter(n.get("level") for n in nodes)
    relation_counts = Counter(e.get("relation") for e in edges)

    r.check(
        len(nodes) == EXPECTED_TOPOLOGY["nodes"],
        f"Topology nodes = {len(nodes)}",
        f"Topology nodes = {len(nodes)}, expected {EXPECTED_TOPOLOGY['nodes']}",
    )
    r.check(
        len(edges) == EXPECTED_TOPOLOGY["edges"],
        f"Topology edges = {len(edges)}",
        f"Topology edges = {len(edges)}, expected {EXPECTED_TOPOLOGY['edges']}",
    )
    r.check(
        dict(level_counts) == EXPECTED_TOPOLOGY["levels"],
        f"Level counts = {dict(level_counts)}",
        f"Level counts = {dict(level_counts)}, expected {EXPECTED_TOPOLOGY['levels']}",
    )
    r.check(
        relation_counts.get("cross_system", 0) == EXPECTED_TOPOLOGY["cross_system_edges"],
        f"Cross-system edges = {relation_counts.get('cross_system', 0)}",
        (
            f"Cross-system edges = {relation_counts.get('cross_system', 0)}, "
            f"expected {EXPECTED_TOPOLOGY['cross_system_edges']}"
        ),
    )


def _oracle_meta_rows() -> List[Tuple[Path, Dict[str, Any]]]:
    rows = []
    for meta_path in MODELS_DIR.rglob("metadata.json"):
        try:
            rows.append((meta_path, _load_json(meta_path)))
        except Exception:
            rows.append((meta_path, {}))
    return rows


def check_oracle_models(r: Reporter, require_runtime: bool) -> None:
    r.header("3. Oracle Model Artifacts")
    if not MODELS_DIR.exists():
        r.fail(f"Model directory missing: {MODELS_DIR}")
        return

    rows = _oracle_meta_rows()
    r.check(
        len(rows) == 85,
        "Model metadata count = 85",
        f"Model metadata count = {len(rows)}, expected 85",
    )

    missing_model = []
    missing_features = []
    oracle_f1 = []
    oracle_rows = []
    for meta_path, meta in rows:
        model_path = meta.get("model_path")
        if model_path:
            candidate = Path(model_path)
            if not candidate.is_absolute():
                candidate = meta_path.parent / "model.txt"
        else:
            candidate = meta_path.parent / "model.txt"
        if not candidate.exists():
            missing_model.append(str(candidate))
        if not meta.get("feature_cols"):
            missing_features.append(str(meta_path))
        node_id = meta.get("node_id", meta_path.parent.name.replace("__", "::"))
        if node_id.endswith("::oracle"):
            oracle_rows.append((node_id, meta))
            if isinstance(meta.get("weighted_f1"), (int, float)):
                oracle_f1.append(float(meta["weighted_f1"]))

    r.check(
        not missing_model,
        "Every metadata entry has a model.txt file",
        f"Missing model files: {missing_model[:5]}",
    )
    r.check(
        not missing_features,
        "Every metadata entry has feature_cols",
        f"Missing feature_cols in: {missing_features[:5]}",
    )
    r.check(
        len(oracle_rows) == 8,
        "System Oracle count = 8",
        f"System Oracle count = {len(oracle_rows)}, expected 8",
    )
    if oracle_f1:
        min_f1 = min(oracle_f1)
        mean_f1 = sum(oracle_f1) / len(oracle_f1)
        r.check(
            min_f1 >= 0.60 and mean_f1 >= 0.85,
            f"System Oracle F1 min={min_f1:.3f}, mean={mean_f1:.3f}",
            f"System Oracle F1 min={min_f1:.3f}, mean={mean_f1:.3f}; inspect weak oracles",
            fatal=False,
        )

    per_node_f1 = [
        float(meta["weighted_f1"])
        for _, meta in rows
        if isinstance(meta.get("weighted_f1"), (int, float))
        and not str(meta.get("node_id", "")).endswith("::oracle")
    ]
    if per_node_f1:
        r.warn(
            "Per-node fallback models are weak: "
            f"median F1={statistics.median(per_node_f1):.3f}, "
            f"mean F1={sum(per_node_f1) / len(per_node_f1):.3f}. "
            "Training/eval should prefer system Oracle routing."
        )

    if _has_module("lightgbm"):
        try:
            import numpy as np
            from src.node_models.model_registry import ModelRegistry

            registry = ModelRegistry(str(MODELS_DIR))
            load_errors = []
            for node_id, meta in oracle_rows:
                n_features = len(meta.get("feature_cols", []))
                pred = registry.predict(node_id, np.zeros(n_features, dtype=np.float32))
                if pred.get("status") == "error":
                    load_errors.append((node_id, pred.get("error")))
            r.check(
                not load_errors,
                "LightGBM runtime smoke loaded all 8 system Oracles",
                f"Oracle runtime smoke errors: {load_errors[:3]}",
                fatal=require_runtime,
            )
        except Exception as exc:
            r.check(
                False,
                "",
                f"Oracle runtime smoke failed: {type(exc).__name__}: {exc}",
                fatal=require_runtime,
            )
    else:
        r.check(
            False,
            "",
            "Skipped LightGBM runtime smoke because lightgbm is not installed",
            fatal=require_runtime,
        )


def check_scenarios(r: Reporter) -> None:
    r.header("4. Scenario Bank")
    path = DATA_DIR / "all_scenarios.json"
    if not path.exists():
        r.fail(f"Scenario bank missing: {path}")
        return

    scenarios = _load_json(path)
    type_counts = Counter(s.get("scenario_type") for s in scenarios)
    fault_free_files = _fault_free_files()
    r.check(
        len(scenarios) == sum(EXPECTED_ALL_SCENARIOS.values()),
        f"Scenario bank size = {len(scenarios)}",
        f"Scenario bank size = {len(scenarios)}, expected {sum(EXPECTED_ALL_SCENARIOS.values())}",
    )
    r.check(
        dict(type_counts) == EXPECTED_ALL_SCENARIOS,
        f"Scenario distribution = {dict(type_counts)}",
        f"Scenario distribution = {dict(type_counts)}, expected {EXPECTED_ALL_SCENARIOS}",
    )

    bad_no_fault = []
    for s in scenarios:
        stype = s.get("scenario_type", "")
        if "no_fault" not in stype:
            continue
        path_nodes = (s.get("diagnostic_path") or {}).get("nodes", [])
        clean = (
            str(s.get("root_cause_node", "")).lower() in ("none", "")
            and str(s.get("fault_type", "")).lower() in ("normal", "no_fault")
            and str(s.get("fault_intensity", "")).lower() in ("none", "")
            and _is_fault_free_source(
                s.get("root_cause_system", ""),
                s.get("source_file", ""),
                fault_free_files,
            )
            and not any(n.get("role") == "root_cause" for n in path_nodes)
        )
        if not clean:
            bad_no_fault.append(s.get("scenario_id", "<unknown>"))
    r.check(
        not bad_no_fault,
        "No-fault scenarios use clean GT, baseline CSV, and no root-cause path",
        f"No-fault contamination remains: {bad_no_fault[:5]}",
    )


def check_sft_data(r: Reporter) -> None:
    r.header("5. SFT Dataset")
    path = DATA_DIR / "sft_train.jsonl"
    if not path.exists():
        r.fail(f"SFT data missing: {path}")
        return

    total = 0
    types = Counter()
    tool_counts = Counter()
    malformed = 0
    missing_diagnosis = 0
    bad_no_fault = []
    for total, entry in enumerate(_iter_jsonl(path), start=1):
        meta = entry.get("metadata", {})
        types[meta.get("scenario_type", "unknown")] += 1
        conversations = entry.get("conversations", [])
        if not entry.get("system") or not conversations:
            malformed += 1
        full_text = " ".join(c.get("value", "") for c in conversations)
        for match in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", full_text, re.DOTALL):
            try:
                tc = json.loads(match.group(1))
                tool_counts[tc.get("name", "unknown")] += 1
            except Exception:
                tool_counts["invalid"] += 1
        if "<diagnosis>" not in full_text:
            missing_diagnosis += 1
        gt = meta.get("ground_truth", {})
        if _is_no_fault_gt(gt, meta.get("scenario_type", "")) and not _clean_no_fault_gt(gt):
            bad_no_fault.append(entry.get("id", "<unknown>"))

    r.check(total == 3000, f"SFT examples = {total}", f"SFT examples = {total}, expected 3000")
    r.check(malformed == 0, "SFT structural format is valid", f"SFT malformed rows = {malformed}")
    r.check(
        missing_diagnosis == 0,
        "Every SFT trajectory includes a final <diagnosis>",
        f"SFT rows missing <diagnosis>: {missing_diagnosis}",
    )
    r.check(
        not bad_no_fault,
        "SFT no-fault ground truth is normalized",
        f"SFT no-fault GT contamination: {bad_no_fault[:5]}",
    )
    r.pass_(f"SFT scenario distribution = {dict(types)}")
    r.pass_(f"SFT tool distribution = {dict(tool_counts)}")
    r.check(
        tool_counts.get("get_node_status_summary", 0) > 0,
        "SFT includes get_node_status_summary trajectories",
        "SFT lacks get_node_status_summary trajectories; regenerate SFT data",
    )
    r.check(
        tool_counts.get("get_related_systems", 0) > 0,
        "SFT includes get_related_systems trajectories",
        "SFT lacks get_related_systems trajectories; regenerate SFT data",
    )
    r.check(
        tool_counts.get("get_node_sensors", 0) > 0,
        "SFT includes get_node_sensors trajectories",
        "SFT lacks get_node_sensors trajectories; regenerate SFT data",
    )

    audit_path = DATA_DIR / "sft_audit_report.json"
    if audit_path.exists():
        audit = _load_json(audit_path)
        issue_rate = audit.get("samples_with_issues", 0) / max(audit.get("total_samples", 1), 1)
        r.check(
            issue_rate <= 0.02,
            f"SFT audit issue rate = {issue_rate:.2%}",
            f"SFT audit issue rate = {issue_rate:.2%}; regenerate or repair SFT data",
            fatal=False,
        )
        if audit.get("category_counts"):
            r.warn(f"SFT audit categories: {audit['category_counts']}")
    else:
        r.warn("SFT audit report not found")

    try:
        import yaml

        config = yaml.safe_load(Path("configs/sft_config.yaml").read_text(encoding="utf-8"))
        r.check(
            bool(config.get("select_best_by_diagnostic", False)),
            "SFT will promote Oracle diagnostic best checkpoint to outputs/sft/best",
            "SFT config does not promote diagnostic best checkpoint",
        )
        r.check(
            config.get("diagnostic_best_metric") == "aggregate_score",
            "SFT diagnostic best metric = aggregate_score",
            f"SFT diagnostic best metric is {config.get('diagnostic_best_metric')}",
        )
        diag_eval_episodes = int(config.get("diag_eval_episodes", 0))
        diag_eval_sampling = config.get("diag_eval_sampling", "")
        r.check(
            diag_eval_sampling == "stratified",
            "SFT diagnostic eval uses stratified sampling",
            f"SFT diagnostic eval sampling is {diag_eval_sampling}",
        )
        r.check(
            diag_eval_episodes >= 40,
            f"SFT diagnostic eval episodes = {diag_eval_episodes} (stratified sample)",
            f"SFT diagnostic eval episodes too small: {diag_eval_episodes}",
        )
        r.check(
            int(config.get("diag_eval_max_steps", 0)) >= 15,
            f"SFT diagnostic eval max steps = {config.get('diag_eval_max_steps')}",
            f"SFT diagnostic eval max steps too small: {config.get('diag_eval_max_steps')}",
        )
    except Exception as exc:
        r.fail(f"Could not inspect SFT config: {type(exc).__name__}: {exc}")


def check_rl_data(r: Reporter) -> None:
    r.header("6. RL Dataset And Curriculum")
    fault_free_files = _fault_free_files()
    for filename, expected_count in EXPECTED_RL_SPLITS.items():
        path = DATA_DIR / filename
        if not path.exists():
            r.fail(f"RL split missing: {path}")
            continue

        total = 0
        malformed = 0
        bad_roles = 0
        bad_no_fault = []
        bad_no_fault_source = []
        types = Counter()
        for total, entry in enumerate(_iter_jsonl(path), start=1):
            messages = entry.get("messages", [])
            gt = entry.get("ground_truth", {})
            meta = entry.get("metadata", {})
            stype = meta.get("scenario_type", "unknown")
            types[stype] += 1
            if not entry.get("id") or not messages or not gt:
                malformed += 1
            if [m.get("role") for m in messages] != ["system", "user"]:
                bad_roles += 1
            if _is_no_fault_gt(gt, stype):
                if not _clean_no_fault_gt(gt):
                    bad_no_fault.append(entry.get("id", "<unknown>"))
                if meta.get("source_file") and not _is_fault_free_source(
                    gt.get("root_cause_system", ""),
                    meta.get("source_file", ""),
                    fault_free_files,
                ):
                    bad_no_fault_source.append(entry.get("id", "<unknown>"))

        r.check(
            total == expected_count,
            f"{filename}: prompts = {total}",
            f"{filename}: prompts = {total}, expected {expected_count}",
        )
        r.check(malformed == 0, f"{filename}: format valid", f"{filename}: malformed rows = {malformed}")
        r.check(bad_roles == 0, f"{filename}: roles are [system,user]", f"{filename}: bad roles = {bad_roles}")
        r.check(
            not bad_no_fault and not bad_no_fault_source,
            f"{filename}: no-fault GT and source CSV are clean",
            (
                f"{filename}: bad no-fault GT={bad_no_fault[:3]}, "
                f"bad source={bad_no_fault_source[:3]}"
            ),
        )
        r.pass_(f"{filename}: scenario distribution = {dict(types)}")

    try:
        import yaml

        config = yaml.safe_load(Path("configs/rl_config.yaml").read_text(encoding="utf-8"))
        p1 = set(config.get("curriculum", {}).get("phase1_types", []))
        p2 = set(config.get("curriculum", {}).get("phase2_types", []))
        r.check(
            "na_no_fault" not in p1 and "no_fault" not in p1,
            "Phase 1 excludes no-fault prompts",
            f"Phase 1 includes no-fault types: {p1}",
        )
        r.check(
            "na_no_fault" not in p2 and "no_fault" not in p2,
            "Phase 2 excludes no-fault prompts",
            f"Phase 2 includes no-fault types: {p2}",
        )
        r.check(
            config.get("no_fault_cap", 0) <= 100,
            f"RL no_fault_cap = {config.get('no_fault_cap')}",
            f"RL no_fault_cap too high: {config.get('no_fault_cap')}",
        )
        r.check(
            config.get("diag_eval_sampling") == "stratified",
            "RL diagnostic eval uses stratified sampling",
            f"RL diagnostic eval sampling is {config.get('diag_eval_sampling')}",
        )
        effective_phase3_size = 0
        effective_no_fault = 0
        train_rows = list(_iter_jsonl(DATA_DIR / "rl_train.jsonl"))
        no_fault_cap = config.get("no_fault_cap", 100)
        counts = Counter(
            row.get("metadata", {}).get("scenario_type", "unknown")
            for row in train_rows
        )
        for stype, count in counts.items():
            if stype in ("na_no_fault", "no_fault"):
                kept = min(count, no_fault_cap)
                effective_no_fault += kept
                effective_phase3_size += kept
            else:
                effective_phase3_size += count
        r.check(
            effective_no_fault <= 2 * no_fault_cap,
            (
                "Effective Phase 3 no-fault pool is capped: "
                f"{effective_no_fault}/{effective_phase3_size}"
            ),
            (
                "Effective Phase 3 no-fault pool is not capped: "
                f"{effective_no_fault}/{effective_phase3_size}"
            ),
        )
    except Exception as exc:
        r.fail(f"Could not inspect RL config: {type(exc).__name__}: {exc}")


def check_reward_and_metrics(r: Reporter) -> None:
    r.header("7. Reward And Metric Sanity")
    try:
        from src.evaluation.metrics import diagnostic_accuracy
        from src.training.reward_functions import compute_accuracy_reward, compute_total_reward

        false_no_fault = compute_accuracy_reward(
            {"status": "Normal", "root_cause_node": "none", "fault_type": "no_fault"},
            {"root_cause_system": "rtu", "root_cause_node": "rtu::RTU", "fault_type": "evapfouling_10"},
        )
        r.check(
            false_no_fault < 0,
            f"False no-fault is penalized ({false_no_fault})",
            f"False no-fault reward is not negative: {false_no_fault}",
        )

        no_diag = compute_total_reward(
            ["<think>checking</think><tool_call>{\"name\":\"diagnose_node\",\"arguments\":{\"node_id\":\"rtu::RTU\"}}</tool_call>"],
            None,
            {"root_cause_system": "rtu", "root_cause_node": "rtu::RTU", "fault_type": "evapfouling_10"},
            n_tool_calls=15,
            tool_results=[],
        )
        r.check(
            no_diag["total"] < 0,
            f"No-diagnosis trajectory is unattractive ({no_diag['total']})",
            f"No-diagnosis total reward should be negative: {no_diag}",
        )

        contradiction = compute_total_reward(
            [],
            {"status": "Normal", "root_cause_node": "none", "fault_type": "no_fault", "confidence": 0.9},
            {"root_cause_system": "rtu", "root_cause_node": "rtu::RTU", "fault_type": "evapfouling_10"},
            n_tool_calls=3,
            tool_results=[json.dumps({"status": "Fault", "node_id": "rtu::RTU", "fault_type": "evapfouling_10"})],
        )
        r.check(
            contradiction["consistency"] < 0 and contradiction["total"] < 0,
            "Tool-result contradiction is penalized",
            f"Contradiction reward too weak: {contradiction}",
        )

        da = diagnostic_accuracy([
            {
                "ground_truth": {"root_cause_node": "none", "fault_type": "Normal"},
                "final_diagnosis": {"status": "Normal", "root_cause_node": "none", "fault_type": "no_fault"},
                "tool_results": [],
            },
            {
                "ground_truth": {"root_cause_node": "none", "fault_type": "Normal"},
                "final_diagnosis": {"status": "Normal", "root_cause_node": "none", "fault_type": "no_fault"},
                "tool_results": [json.dumps({"status": "Fault", "node_id": "rtu::RTU"})],
            },
        ])
        r.check(
            abs(da - 0.5) < 1e-6,
            "No-fault metric rejects abnormal tool evidence",
            f"No-fault DA consistency mismatch: {da}",
        )
    except Exception as exc:
        r.fail(f"Reward/metric sanity failed: {type(exc).__name__}: {exc}")


def check_training_records(r: Reporter) -> None:
    r.header("8. Historical Training Records")
    sft_hist = Path("outputs/sft/diag_eval_history.json")
    rl_hist = Path("outputs/rl/rl_diag_eval_history.json")
    if sft_hist.exists():
        rows = _load_json(sft_hist)
        best = max(rows, key=lambda x: x.get("diagnostic_accuracy", 0))
        r.pass_(
            "Best SFT diagnostic record: "
            f"step={best.get('step')}, DA={best.get('diagnostic_accuracy'):.1%}, "
            f"Agg={best.get('aggregate_score'):.3f}"
        )
    else:
        r.warn("SFT diagnostic history not found locally")

    if rl_hist.exists():
        rows = _load_json(rl_hist)
        best = max(rows, key=lambda x: x.get("diagnostic_accuracy", 0))
        last = rows[-1]
        r.warn(
            "Current RL underperforms SFT: "
            f"best DA={best.get('diagnostic_accuracy'):.1%} at step {best.get('step')}, "
            f"last DA={last.get('diagnostic_accuracy'):.1%} at step {last.get('step')}"
        )
    else:
        r.warn("RL diagnostic history not found locally")

    ckpt = Path("outputs/sft/best")
    if ckpt.exists():
        has_adapter = (ckpt / "adapter_config.json").exists() and (
            ckpt / "adapter_model.safetensors"
        ).exists()
        r.check(has_adapter, "Local SFT best checkpoint exists", "Local SFT best checkpoint is incomplete")
    else:
        r.warn("Local outputs/sft/best checkpoint is absent; cloud must train or provide SFT checkpoint")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-oracle-runtime",
        action="store_true",
        help="Fail if rdflib/lightgbm runtime smoke cannot be performed locally.",
    )
    args = parser.parse_args()

    reporter = Reporter()
    check_dependencies(reporter, args.require_oracle_runtime)
    check_topology(reporter)
    check_oracle_models(reporter, args.require_oracle_runtime)
    check_scenarios(reporter)
    check_sft_data(reporter)
    check_rl_data(reporter)
    check_reward_and_metrics(reporter)
    check_training_records(reporter)

    reporter.header("Summary")
    if reporter.failures:
        print(f"  Status: NOT READY ({len(reporter.failures)} failure(s))")
        for item in reporter.failures:
            print(f"  - {item}")
        return 1
    if reporter.warnings:
        print(f"  Status: READY WITH WARNINGS ({len(reporter.warnings)} warning(s))")
        for item in reporter.warnings:
            print(f"  - {item}")
        return 0
    print("  Status: READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
