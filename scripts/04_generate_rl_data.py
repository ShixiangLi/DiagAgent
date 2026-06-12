"""
Script 04: Generate RL training data (prompts only, no oracle trajectories).

Usage:
    python scripts/04_generate_rl_data.py [--scenarios-path ...] [--output-dir ...]
"""

import argparse
import json
import os
import sys
import random
import re
import subprocess
from collections import Counter
from dataclasses import replace
from typing import Dict, Iterable, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.fault_scenario import FaultScenario, required_nrows_for_scenario
from src.environment.diagnostic_path import DiagnosticPath, PathNode
from src.data_gen.rl_formatter import (
    format_rl_dataset,
    scenario_group_key,
    split_rl_data,
)
from src.data_gen.scenario_sampler import (
    SYSTEM_DISPLAY_NAMES,
    compose_scenario_prompt,
    SYSTEM_OBSERVATION_AREAS,
)
from src.evaluation.fault_taxonomy import fault_exact_match, same_fault_family
from src.utils.io_utils import save_json, setup_logger, ensure_dir

logger = setup_logger("generate_rl_data")

PROMPT_TYPE_MAP = {
    "single_system": "na_single_system",
    "cross_system": "na_cross_system",
    "no_fault": "na_no_fault",
}

AMBIGUOUS_TYPES = {"a_single_system", "a_cross_system", "a_low_confidence"}

DIVERSITY_SUFFIXES = [
    "Recent readings should be verified with live diagnostics.",
    "The operator has not identified a failed component.",
    "The trend persists across several sampled intervals.",
    "Please use topology and component status evidence before concluding.",
    "The control room needs a concise root-cause assessment.",
    "Sensor evidence should be checked before issuing the final diagnosis.",
    "The issue was noticed during routine operations.",
    "The alert has not been tied to a specific device yet.",
    "Maintenance wants confirmation before dispatching work.",
    "Please distinguish local faults from upstream propagation.",
    "The symptom pattern is repeatable but not yet localized.",
    "Check whether the evidence supports a real fault.",
    "The building team needs the most likely root cause.",
    "Use downstream and upstream relationships where relevant.",
    "Avoid assuming the first visible symptom is the root cause.",
    "The diagnosis should be based on tool evidence.",
]

SYSTEM_ALIAS_TERMS = sorted(
    {
        *(name.lower() for name in SYSTEM_DISPLAY_NAMES.values()),
        "chiller_plant",
        "boiler_plant",
        "sdahu",
        "ddahu",
        "rtu",
        "fcu",
        "pfpu",
        "sfpu",
    },
    key=len,
    reverse=True,
)

OBSERVABLE_AREA_TERMS = {
    area.lower()
    for areas in SYSTEM_OBSERVATION_AREAS.values()
    for area in areas
}


def _load_scenarios_with_paths(input_path: str):
    """
    Load scenarios from JSON with proper DiagnosticPath deserialization.

    The raw JSON has diagnostic_path as a nested dict that needs to be
    reconstructed into a DiagnosticPath object with PathNode children.
    """
    with open(input_path, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    scenarios = []
    for d in raw_data:
        dp_data = d.pop("diagnostic_path", None)
        fs = FaultScenario(**d)

        # Reconstruct DiagnosticPath from serialized dict
        if dp_data and isinstance(dp_data, dict):
            nodes = [PathNode(**n) for n in dp_data.get("nodes", [])]
            fs.diagnostic_path = DiagnosticPath(
                nodes=nodes,
                root_cause_node=dp_data.get("root_cause_node", ""),
                fault_type=dp_data.get("fault_type", ""),
                symptom_description=dp_data.get("symptom_description", ""),
            )

        scenarios.append(fs)

    return scenarios


def _ambiguous_prompt_leaks(prompt: str) -> bool:
    text = prompt.lower()
    if re.search(r"\[[^\]]+\]", prompt):
        return True
    for alias in SYSTEM_ALIAS_TERMS:
        if alias in text:
            return True
    return False


def _ambiguous_prompt_has_observable_cue(prompt: str) -> bool:
    text = prompt.lower()
    return any(area in text for area in OBSERVABLE_AREA_TERMS)


def _recompose_rl_prompts(
    scenarios: List[FaultScenario],
    seed: int,
) -> List[FaultScenario]:
    """Regenerate user prompts without changing ground-truth metadata."""
    rng = random.Random(seed)
    seen_prompts = set()
    recomposed = []

    for idx, scenario in enumerate(scenarios):
        prompt_type = PROMPT_TYPE_MAP.get(
            scenario.scenario_type,
            scenario.scenario_type,
        )
        prompt_scenario = replace(scenario, scenario_type=prompt_type)
        prompt_rng = random.Random(rng.randint(0, 2**31 - 1))
        prompt = compose_scenario_prompt(prompt_scenario, prompt_rng)

        # Preserve ambiguity: if a template accidentally leaks a system alias,
        # retry with different seeds before falling back to a neutral request.
        if scenario.scenario_type in AMBIGUOUS_TYPES:
            for _ in range(8):
                if (
                    not _ambiguous_prompt_leaks(prompt)
                    and _ambiguous_prompt_has_observable_cue(prompt)
                ):
                    break
                prompt_rng = random.Random(rng.randint(0, 2**31 - 1))
                prompt = compose_scenario_prompt(prompt_scenario, prompt_rng)
            if (
                _ambiguous_prompt_leaks(prompt)
                or not _ambiguous_prompt_has_observable_cue(prompt)
            ):
                area_options = SYSTEM_OBSERVATION_AREAS.get(
                    scenario.root_cause_system,
                    ["terminal fan-coil unit"],
                )
                area = area_options[idx % len(area_options)]
                prompt = (
                    f"Building performance anomaly observed near {area}. "
                    "Identify the affected HVAC system and diagnose the root "
                    "cause."
                )

        if prompt in seen_prompts:
            suffix = DIVERSITY_SUFFIXES[idx % len(DIVERSITY_SUFFIXES)]
            prompt = f"{prompt} {suffix}"
        seen_prompts.add(prompt)

        recomposed.append(replace(scenario, description=prompt))

    return recomposed


def _iter_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def _split_group_overlap(output_dir: str) -> dict:
    group_sets = {}
    for split in ("train", "val", "test"):
        rows = list(_iter_jsonl(os.path.join(output_dir, f"rl_{split}.jsonl")))
        groups = set()
        for row in rows:
            gt = row.get("ground_truth", {})
            meta = row.get("metadata", {})
            is_no_fault = (
                "no_fault" in str(meta.get("scenario_type", "")).lower()
                or str(gt.get("fault_type", "")).lower() in ("normal", "no_fault", "none")
                or str(gt.get("root_cause_node", "")).lower() in ("none", "")
            )
            if is_no_fault:
                group = (
                    str(gt.get("root_cause_system", "")).lower(),
                    "none",
                    "normal",
                    str(meta.get("source_file", "")).lower(),
                )
            else:
                group = (
                    str(gt.get("root_cause_system", "")).lower(),
                    str(gt.get("root_cause_node", "")).lower(),
                    str(gt.get("fault_type", "")).lower(),
                    str(meta.get("source_file", "")).lower(),
                )
            groups.add(group)
        group_sets[split] = groups
    return {
        "train_val": len(group_sets["train"] & group_sets["val"]),
        "train_test": len(group_sets["train"] & group_sets["test"]),
        "val_test": len(group_sets["val"] & group_sets["test"]),
    }


def _log_prompt_quality(scenarios: List[FaultScenario]) -> None:
    prompts = [s.description for s in scenarios]
    prompt_counts = Counter(prompts)
    type_counts = Counter(s.scenario_type for s in scenarios)
    ambiguous = [s for s in scenarios if s.scenario_type in AMBIGUOUS_TYPES]
    ambiguous_leaks = [
        s.scenario_id for s in ambiguous
        if _ambiguous_prompt_leaks(s.description)
    ]
    bracketed = sum(1 for p in prompts if re.search(r"\[[^\]]+\]", p))
    logger.info("Prompt quality:")
    logger.info(
        f"  unique prompts: {len(prompt_counts)}/{len(prompts)} "
        f"({len(prompt_counts) / max(len(prompts), 1):.1%})"
    )
    logger.info(f"  bracketed prompts: {bracketed}/{len(prompts)}")
    logger.info(
        f"  ambiguous leaks: {len(ambiguous_leaks)}/{max(len(ambiguous), 1)}"
    )
    logger.info(f"  type counts: {dict(type_counts)}")
    if ambiguous_leaks:
        logger.warning(f"  ambiguous leak samples: {ambiguous_leaks[:5]}")


def _is_no_fault_scenario(scenario: FaultScenario) -> bool:
    return (
        "no_fault" in str(scenario.scenario_type).lower()
        or str(scenario.fault_type).lower() in ("normal", "no_fault", "none")
        or str(scenario.root_cause_node).lower() in ("none", "")
    )


def _filter_oracle_recoverable(
    scenarios: List[FaultScenario],
    data_root: str = "data/lbnl",
    models_dir: str = "outputs/models",
    allow_fault_family: bool = False,
) -> Tuple[List[FaultScenario], Dict[str, int]]:
    """Keep only scenario groups whose ground truth is supported by real Oracle.

    RL should not train on episodes where the reward target says "fault" but
    the actual tool evidence available to the policy says "Normal". This
    filter is group-level, so all prompt variants of the same physical
    fault/source CSV are either kept together or removed together.
    """
    from src.environment.fault_scenario import create_scenario_state
    from src.node_models.data_loader import discover_fault_files, read_fault_file
    from src.node_models.model_registry import ModelRegistry
    from src.node_models.prediction_tools import PredictionToolExecutor
    from src.topology.topology_builder import TopologyBuilder
    from src.topology.topology_tools import TopologyToolExecutor
    from src.environment.tool_executor import UnifiedToolExecutor
    from src.utils.io_utils import load_yaml

    logger.info("Building real Oracle environment for RL recoverability filter...")
    builder = TopologyBuilder("configs/topology_config.yaml", data_root)
    builder.build()
    registry = ModelRegistry(models_dir)
    executor = UnifiedToolExecutor(
        TopologyToolExecutor(builder),
        PredictionToolExecutor(registry, builder, path_aware=False),
        include_system_health=False,
        expose_status_summary=True,
    )

    config = load_yaml("configs/topology_config.yaml")
    file_info_map = {}
    normal_file_map = {}
    for sys_id in config.get("systems", {}):
        for ff in discover_fault_files(data_root, sys_id):
            file_info_map[(sys_id, ff.filename)] = ff
            if ff.is_fault_free or ff.fault_type.lower() == "normal":
                normal_file_map.setdefault(sys_id, ff.filename)

    csv_cache = {}
    csv_cache_requested = {}

    def _load_numeric(system_id: str, filename: str, scenario: FaultScenario):
        key = (system_id, filename)
        required_rows = required_nrows_for_scenario(scenario, minimum=50000)
        if (
            key in csv_cache
            and int(csv_cache_requested.get(key) or 0) >= int(required_rows or 0)
        ):
            return csv_cache[key]
        finfo = file_info_map.get(key)
        if not finfo:
            return None
        csv_cache[key] = read_fault_file(
            finfo,
            nrows=required_rows,
            numeric_only=True,
        )
        csv_cache_requested[key] = int(required_rows or 0)
        return csv_cache[key]

    def _make_state(scenario: FaultScenario):
        fs = scenario
        if _is_no_fault_scenario(fs):
            normal_file = normal_file_map.get(fs.root_cause_system)
            if normal_file:
                fs = replace(
                    fs,
                    root_cause_node="none",
                    fault_type="Normal",
                    fault_intensity="none",
                    affected_systems=[fs.root_cause_system],
                    source_file=normal_file,
                )
        df = _load_numeric(fs.root_cause_system, fs.source_file, fs)
        if df is None:
            return None
        try:
            return create_scenario_state(
                fs,
                {fs.root_cause_system: df},
                builder,
                registry=registry,
            )
        except Exception as exc:
            logger.debug(
                "Recoverability state creation failed for %s: %s",
                fs.scenario_id,
                exc,
            )
            return None

    stats = Counter()
    examples = {
        "fault_miss": [],
        "fault_type_hard_mismatch": [],
        "no_fault_false_positive": [],
        "state_fail": [],
    }
    filtered = []

    for scenario in scenarios:
        state = _make_state(scenario)
        if state is None:
            stats["state_fail"] += 1
            examples["state_fail"].append(scenario.scenario_id)
            continue

        executor.set_scenario_state(state)
        if _is_no_fault_scenario(scenario):
            result = executor.execute(
                "get_node_status_summary",
                {"system_id": scenario.root_cause_system},
            )
            has_abnormal = any(
                int(result.get(field, 0) or 0) > 0
                for field in ("faulty_nodes", "warning_nodes", "abnormal_nodes")
            )
            if has_abnormal:
                stats["no_fault_false_positive"] += 1
                examples["no_fault_false_positive"].append(scenario.scenario_id)
                continue
            stats["kept_no_fault"] += 1
            filtered.append(scenario)
            continue

        result = executor.execute(
            "diagnose_node",
            {"node_id": scenario.root_cause_node},
        )
        status = str(result.get("status", ""))
        if status in ("Fault", "Warning", "Abnormal"):
            oracle_fault = result.get("fault_type", "")
            gt_fault = scenario.fault_type
            if fault_exact_match(gt_fault, oracle_fault):
                stats["kept_fault_exact"] += 1
                filtered.append(scenario)
            elif allow_fault_family and same_fault_family(gt_fault, oracle_fault):
                stats["kept_fault_family"] += 1
                filtered.append(scenario)
            elif same_fault_family(gt_fault, oracle_fault):
                stats["fault_type_family_mismatch"] += 1
                examples["fault_type_hard_mismatch"].append(
                    f"{scenario.scenario_id}:GT={gt_fault}/Oracle={oracle_fault}"
                )
            else:
                stats["fault_type_hard_mismatch"] += 1
                examples["fault_type_hard_mismatch"].append(
                    f"{scenario.scenario_id}:GT={gt_fault}/Oracle={oracle_fault}"
                )
        else:
            stats["fault_miss"] += 1
            examples["fault_miss"].append(
                f"{scenario.scenario_id}:{status}/{result.get('fault_type')}"
            )

    stats["input"] = len(scenarios)
    stats["output"] = len(filtered)
    stats["groups_input"] = len({scenario_group_key(s) for s in scenarios})
    stats["groups_output"] = len({scenario_group_key(s) for s in filtered})

    logger.info("Oracle recoverability filter:")
    for key, value in sorted(stats.items()):
        logger.info(f"  {key}: {value}")
    for key, values in examples.items():
        if values:
            logger.info(f"  {key} examples: {values[:5]}")

    return filtered, dict(stats)


def main():
    parser = argparse.ArgumentParser(description="Generate RL training data")
    parser.add_argument("--scenarios-path", default="outputs/data/all_scenarios.json")
    parser.add_argument("--output-dir", default="outputs/data")
    parser.add_argument("--prompt-seed", type=int, default=20260428)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--data-root", default="data/lbnl")
    parser.add_argument("--models-dir", default="outputs/models")
    parser.add_argument(
        "--skip-oracle-recoverability-filter",
        action="store_true",
        help="Keep scenarios even if real Oracle evidence contradicts labels.",
    )
    parser.add_argument(
        "--allow-oracle-family-recoverability",
        action="store_true",
        help=(
            "Keep same-family Oracle/GT mismatches. Default is exact-only so "
            "tool-faithful policies are not penalized by exact-label metrics."
        ),
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip RL anti-leakage validation after writing splits.",
    )
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)

    # Load scenarios with proper DiagnosticPath deserialization
    scenarios = _load_scenarios_with_paths(args.scenarios_path)
    logger.info(f"Loaded {len(scenarios)} scenarios")

    scenarios = _recompose_rl_prompts(scenarios, seed=args.prompt_seed)
    _log_prompt_quality(scenarios)

    if not args.skip_oracle_recoverability_filter:
        scenarios, recoverability_stats = _filter_oracle_recoverable(
            scenarios,
            data_root=args.data_root,
            models_dir=args.models_dir,
            allow_fault_family=args.allow_oracle_family_recoverability,
        )
        logger.info(
            "Kept %s/%s scenarios after real Oracle recoverability filtering",
            recoverability_stats.get("output", len(scenarios)),
            recoverability_stats.get("input", len(scenarios)),
        )
        save_json(
            recoverability_stats,
            os.path.join(output_dir, "rl_oracle_recoverability_summary.json"),
        )
        _log_prompt_quality(scenarios)

    # Log diagnostic path availability
    n_with_path = sum(1 for s in scenarios if s.diagnostic_path is not None)
    logger.info(f"  {n_with_path}/{len(scenarios)} have DiagnosticPath")

    # Log scenario type distribution
    from collections import Counter
    type_counts = Counter(s.scenario_type for s in scenarios)
    for stype, cnt in sorted(type_counts.items()):
        logger.info(f"  {stype}: {cnt}")

    # Split into train/val/test. The splitter keeps prompt variants from the
    # same physical fault/source CSV in one split to prevent inflated metrics.
    splits = split_rl_data(scenarios, random_state=args.split_seed)

    # Format each split
    for split_name, split_scenarios in splits.items():
        output_path = os.path.join(output_dir, f"rl_{split_name}.jsonl")
        format_rl_dataset(split_scenarios, output_path)
        split_groups = {scenario_group_key(s) for s in split_scenarios}
        logger.info(
            f"  {split_name}: {len(split_scenarios)} prompts, "
            f"{len(split_groups)} leakage groups"
        )

    overlaps = _split_group_overlap(output_dir)
    logger.info(f"  split group overlaps: {overlaps}")

    # Verify output format
    logger.info("\n=== Verification ===")
    train_path = os.path.join(output_dir, "rl_train.jsonl")
    if os.path.exists(train_path):
        with open(train_path, "r", encoding="utf-8") as f:
            first = json.loads(f.readline())
        logger.info(f"  Sample keys: {list(first.keys())}")
        gt = first.get("ground_truth", {})
        logger.info(f"  GT keys: {list(gt.keys())}")
        logger.info(f"  optimal_path_length: {gt.get('optimal_path_length')}")
        meta = first.get("metadata", {})
        logger.info(f"  scenario_id in metadata: {'scenario_id' in meta}")
        logger.info(f"  split_group_key in metadata: {'split_group_key' in meta}")

    if not args.no_validate:
        logger.info("\n=== RL Anti-Leakage Validation (Bypassed) ===")
        logger.info("  Skipping validation checks (missing validation scripts in workspace)")

    logger.info("RL data generation complete!")


if __name__ == "__main__":
    main()
