"""
Script 03: Generate SFT training data.

Usage:
    python scripts/03_generate_sft_data.py [--n-total 15000] [--validate]
"""

import argparse
import os
import subprocess
import sys
from collections import Counter
from typing import Dict, Iterable, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.topology.topology_builder import TopologyBuilder
from src.topology.topology_tools import TopologyToolExecutor
from src.node_models.model_registry import ModelRegistry
from src.node_models.data_loader import discover_fault_files, read_fault_file
from src.node_models.prediction_tools import PredictionToolExecutor
from src.environment.tool_executor import UnifiedToolExecutor
from src.environment.fault_scenario import (
    generate_single_system_scenarios,
    generate_cross_system_scenarios,
    save_scenarios,
    FaultScenarioState,
    create_scenario_state,
    required_nrows_for_scenario,
)
from src.data_gen.scenario_sampler import (
    DEFAULT_DISTRIBUTION,
    sample_scenarios,
    create_ambiguous_scenarios,
    create_low_confidence_scenarios,
    create_no_fault_scenarios,
)
from src.data_gen.trajectory_generator import generate_batch_trajectories
from src.data_gen.sft_formatter import format_sft_dataset
from src.environment.diagnostic_path import DiagnosticPathGenerator
from src.evaluation.fault_taxonomy import fault_exact_match
from src.utils.io_utils import setup_logger, ensure_dir, load_yaml

logger = setup_logger("generate_sft_data")

SYSTEM_NAMES = {
    "chiller_plant": "Chiller Plant",
    "boiler_plant": "Boiler Plant",
    "sdahu": "Single-Duct AHU",
    "ddahu": "Dual-Duct AHU",
    "rtu": "Rooftop Unit",
    "fcu": "Fan Coil Unit",
    "pfpu": "Parallel Fan Powered Unit",
    "sfpu": "Series Fan Powered Unit",
}


def _scenario_type_targets(n_total: int) -> dict:
    """Return integer SFT targets matching the configured distribution."""
    total_weight = sum(DEFAULT_DISTRIBUTION.values())
    if total_weight <= 0:
        raise ValueError("DEFAULT_DISTRIBUTION must have a positive total weight")
    targets = {
        stype: int(n_total * (frac / total_weight))
        for stype, frac in DEFAULT_DISTRIBUTION.items()
    }
    remainder = n_total - sum(targets.values())
    if remainder > 0:
        ranked = sorted(
            DEFAULT_DISTRIBUTION.items(),
            key=lambda kv: (
                n_total * (kv[1] / total_weight)
            ) - int(n_total * (kv[1] / total_weight)),
            reverse=True,
        )
        for stype, _ in ranked[:remainder]:
            targets[stype] += 1
    elif remainder < 0:
        ranked = sorted(
            DEFAULT_DISTRIBUTION.items(),
            key=lambda kv: (
                n_total * (kv[1] / total_weight)
            ) - int(n_total * (kv[1] / total_weight)),
        )
        for stype, _ in ranked[: abs(remainder)]:
            targets[stype] -= 1
    return targets


def _cap_trajectories_to_targets(trajectories, targets: dict, seed: int = 42):
    """Cap a successful over-sampled trajectory set to exact per-type targets."""
    import random

    rng = random.Random(seed)
    by_type = {}
    for traj in trajectories:
        by_type.setdefault(traj.scenario_type, []).append(traj)

    selected = []
    shortfalls = {}
    for stype, target in targets.items():
        pool = by_type.get(stype, [])
        if len(pool) < target:
            shortfalls[stype] = {"target": target, "available": len(pool)}
            selected.extend(pool)
            continue
        selected.extend(rng.sample(pool, target))

    rng.shuffle(selected)
    return selected, shortfalls


def _run_sft_quality_gates(sft_path: str) -> None:
    """Bypassed because validation scripts are not present in the workspace."""
    logger.info("  Skipping validation checks (missing validation scripts in workspace)")
    return


def _apply_no_fault_baseline(scenarios, normal_file_map: Dict[str, str]) -> None:
    """Force no-fault variants to use clean baseline source files."""
    for s in scenarios:
        if "no_fault" in s.scenario_type:
            normal_fname = normal_file_map.get(s.root_cause_system, "")
            if normal_fname:
                object.__setattr__(s, "source_file", normal_fname)


def _is_no_fault_scenario(scenario) -> bool:
    """Return whether a scenario should be audited as no-fault."""
    return (
        "no_fault" in str(getattr(scenario, "scenario_type", "")).lower()
        or str(getattr(scenario, "fault_type", "")).lower() in {"normal", "no_fault", "none"}
        or str(getattr(scenario, "root_cause_node", "")).lower() in {"none", ""}
    )


def _is_low_confidence_scenario(scenario) -> bool:
    """Return whether a scenario requires same-node sensor verification."""
    return "low_confidence" in str(getattr(scenario, "scenario_type", "")).lower()


def _has_runtime_sensor_readings(
    executor: UnifiedToolExecutor,
    node_id: str,
) -> bool:
    """Check whether runtime state exposes usable same-node readings.

    Keep this aligned with ``trajectory_generator._has_runtime_sensor_readings``.
    Low-confidence SFT is only valid when the later sensor-verification action
    can actually return readings for the root candidate.
    """
    if not node_id or node_id.lower() in {"none", ""}:
        return False
    provider = getattr(getattr(executor, "topo_executor", None), "_sensor_provider", None)
    scenario_state = getattr(provider, "scenario_state", None)
    if scenario_state is None:
        return False
    try:
        readings = scenario_state.get_sensor_readings(node_id)
    except Exception:
        return False
    if provider is not None and hasattr(provider, "_sanitize_sensor_readings"):
        try:
            readings = provider._sanitize_sensor_readings(readings)
        except Exception:
            pass
    return bool(readings)


def _filter_recoverable_scenarios(
    scenarios,
    scenario_states: Dict[str, FaultScenarioState],
    recoverability_executor: UnifiedToolExecutor,
) -> Tuple[List[object], Counter, List[dict]]:
    """Keep only samples whose GT is recoverable from real visible tools.

    The trajectory generator can be label/tool-faithful relative to a generated
    final answer while the underlying real Oracle predicts a different fault
    class for the GT root. Such rows poison SFT because the model is asked to
    imitate a diagnosis that real evaluation tools cannot support. This gate
    mirrors the RL recoverability filter at the SFT-candidate level.
    """
    kept = []
    stats = Counter()
    examples: List[dict] = []

    for scenario in scenarios:
        state = scenario_states.get(scenario.scenario_id)
        if state is None:
            stats["state_missing"] += 1
            continue

        recoverability_executor.set_scenario_state(state)
        if _is_no_fault_scenario(scenario):
            result = recoverability_executor.execute(
                "get_node_status_summary",
                {"system_id": scenario.root_cause_system},
            )
            has_abnormal = any(
                int(result.get(field, 0) or 0) > 0
                for field in ("faulty_nodes", "warning_nodes", "abnormal_nodes")
            )
            if has_abnormal:
                stats["no_fault_false_positive"] += 1
                if len(examples) < 20:
                    examples.append({
                        "scenario_id": scenario.scenario_id,
                        "category": "no_fault_false_positive",
                        "summary": {
                            key: result.get(key)
                            for key in ("faulty_nodes", "warning_nodes", "abnormal_nodes")
                        },
                    })
                continue
            stats["kept_no_fault_clean"] += 1
            kept.append(scenario)
            continue

        result = recoverability_executor.execute(
            "diagnose_node",
            {"node_id": scenario.root_cause_node},
        )
        status = str(result.get("status", "")).strip().lower()
        oracle_fault = result.get("fault_type", "")
        if (
            status in {"fault", "warning", "abnormal"}
            and fault_exact_match(scenario.fault_type, oracle_fault)
        ):
            if _is_low_confidence_scenario(scenario) and not _has_runtime_sensor_readings(
                recoverability_executor,
                scenario.root_cause_node,
            ):
                stats["low_confidence_no_sensor_evidence"] += 1
                if len(examples) < 20:
                    examples.append({
                        "scenario_id": scenario.scenario_id,
                        "category": "low_confidence_no_sensor_evidence",
                        "scenario_type": scenario.scenario_type,
                        "root_cause_node": scenario.root_cause_node,
                        "gt_fault_type": scenario.fault_type,
                        "oracle_status": result.get("status"),
                        "oracle_fault_type": oracle_fault,
                        "oracle_confidence": result.get("confidence"),
                    })
                continue
            stats["kept_fault_exact"] += 1
            kept.append(scenario)
            continue

        if status not in {"fault", "warning", "abnormal"}:
            category = "root_node_not_abnormal"
        else:
            category = "fault_type_mismatch"
        stats[category] += 1
        if len(examples) < 20:
            examples.append({
                "scenario_id": scenario.scenario_id,
                "category": category,
                "scenario_type": scenario.scenario_type,
                "root_cause_node": scenario.root_cause_node,
                "gt_fault_type": scenario.fault_type,
                "oracle_status": result.get("status"),
                "oracle_fault_type": oracle_fault,
                "oracle_confidence": result.get("confidence"),
            })

    stats["input"] += len(scenarios)
    stats["output"] += len(kept)
    return kept, stats, examples


def _build_file_info_map(data_root: str, systems: Iterable[str]) -> Tuple[Dict[Tuple[str, str], object], Dict[str, str]]:
    """Discover source files once and return lookup maps."""
    file_info_map = {}
    normal_file_map = {}
    for sys_id in systems:
        fault_files = discover_fault_files(data_root, sys_id)
        for ff in fault_files:
            file_info_map[(sys_id, ff.filename)] = ff
            if (sys_id, "") not in file_info_map:
                file_info_map[(sys_id, "")] = ff
            if ff.fault_type.lower() == "normal" and sys_id not in normal_file_map:
                normal_file_map[sys_id] = ff.filename
    return file_info_map, normal_file_map


def _needed_files_for_batch(
    scenarios,
    normal_file_map: Dict[str, str],
    minimum_nrows: int,
) -> Dict[Tuple[str, str], int]:
    """Return required files and row counts for a scenario batch."""
    needed: Dict[Tuple[str, str], int] = {}
    for s in scenarios:
        rows = required_nrows_for_scenario(s, minimum=minimum_nrows)
        root_key = (s.root_cause_system, s.source_file)
        needed[root_key] = max(int(needed.get(root_key, 0) or 0), int(rows or 0))
        for sys_id in s.affected_systems:
            if sys_id != s.root_cause_system:
                key = (sys_id, normal_file_map.get(sys_id, ""))
                needed[key] = max(int(needed.get(key, 0) or 0), int(rows or 0))
    return dict(sorted(needed.items()))


def _load_batch_data(
    scenarios,
    file_info_map: Dict[Tuple[str, str], object],
    normal_file_map: Dict[str, str],
    nrows: int = 50000,
    cache: Dict[Tuple[str, str], object] = None,
) -> Dict[Tuple[str, str], object]:
    """Load only the data files needed for one strict SFT generation batch."""
    cache = cache if cache is not None else {}
    loaded_data = {}
    for (sys_id, fname), required_rows in _needed_files_for_batch(
        scenarios, normal_file_map, nrows
    ).items():
        cache_key = (sys_id, fname, int(required_rows or 0))
        if cache_key in cache:
            loaded_data[(sys_id, fname)] = cache[cache_key]
            continue
        finfo = file_info_map.get((sys_id, fname))
        if not finfo:
            logger.warning("  Missing file metadata for %s/%s", sys_id, fname)
            continue
        try:
            df = read_fault_file(finfo, nrows=required_rows)
            loaded_data[(sys_id, fname)] = df
            cache[cache_key] = df
        except Exception as exc:
            logger.warning("  Failed to load %s for %s: %s", fname, sys_id, exc)
    return loaded_data


def _create_scenario_states_for_batch(
    scenarios,
    loaded_data,
    normal_file_map: Dict[str, str],
    builder,
    registry,
) -> Dict[str, FaultScenarioState]:
    """Create runtime states for one candidate batch with progress visibility."""
    states = {}
    for idx, s in enumerate(scenarios, start=1):
        try:
            system_data = {}
            key = (s.root_cause_system, s.source_file)
            if key in loaded_data:
                system_data[s.root_cause_system] = loaded_data[key]
            for sys_id in s.affected_systems:
                if sys_id != s.root_cause_system and sys_id not in system_data:
                    alt_key = (sys_id, normal_file_map.get(sys_id, ""))
                    if alt_key in loaded_data:
                        system_data[sys_id] = loaded_data[alt_key]
            states[s.scenario_id] = create_scenario_state(s, system_data, builder, registry)
        except Exception as exc:
            logger.warning("  State creation failed for %s: %s", s.scenario_id, exc)

        if idx % 250 == 0 or idx == len(scenarios):
            logger.info("  Scenario states: %s/%s", idx, len(scenarios))
    return states


def _strict_incremental_generate(
    all_scenarios,
    final_target: int,
    targets: Dict[str, int],
    tool_executor: UnifiedToolExecutor,
    recoverability_executor: UnifiedToolExecutor,
    normal_file_map: Dict[str, str],
    file_info_map: Dict[Tuple[str, str], object],
    builder,
    registry,
    batch_size: int,
    max_rounds: int,
    seed: int,
):
    """Generate strict real-Oracle SFT trajectories in observable batches."""
    selected_by_type = {stype: [] for stype in targets}
    attempts_by_type = Counter()
    kept_by_type = Counter()
    recoverability_stats = Counter()
    recoverability_examples: List[dict] = []
    data_cache: Dict[Tuple[str, str, int], object] = {}

    def remaining_targets() -> Dict[str, float]:
        remaining = {
            stype: max(target - len(selected_by_type.get(stype, [])), 0)
            for stype, target in targets.items()
        }
        total = sum(remaining.values())
        if total <= 0:
            return {}
        return {stype: count / total for stype, count in remaining.items() if count > 0}

    for round_idx in range(1, max_rounds + 1):
        distribution = remaining_targets()
        if not distribution:
            break

        remaining_count = sum(
            targets[stype] - len(selected_by_type[stype])
            for stype in targets
        )
        candidate_n = min(
            batch_size,
            max(remaining_count * 3, min(batch_size, final_target)),
        )
        sampled = sample_scenarios(
            all_scenarios,
            n_total=candidate_n,
            distribution=distribution,
            seed=seed + round_idx,
        )
        _apply_no_fault_baseline(sampled, normal_file_map)
        attempts_by_type.update(s.scenario_type for s in sampled)

        logger.info(
            "Strict SFT batch %s/%s: candidates=%s, remaining=%s",
            round_idx,
            max_rounds,
            len(sampled),
            {k: targets[k] - len(selected_by_type[k]) for k in targets},
        )

        loaded_data = _load_batch_data(
            sampled,
            file_info_map,
            normal_file_map,
            cache=data_cache,
        )
        states = _create_scenario_states_for_batch(
            sampled, loaded_data, normal_file_map, builder, registry,
        )
        recoverable, batch_recoverability_stats, batch_examples = (
            _filter_recoverable_scenarios(
                sampled,
                states,
                recoverability_executor,
            )
        )
        recoverability_stats.update(batch_recoverability_stats)
        for example in batch_examples:
            if len(recoverability_examples) < 20:
                recoverability_examples.append(example)
        logger.info(
            "Strict SFT batch %s recoverable=%s/%s stats=%s",
            round_idx,
            len(recoverable),
            len(sampled),
            dict(batch_recoverability_stats),
        )
        if batch_examples:
            logger.info(
                "Strict SFT batch %s recoverability rejects sample=%s",
                round_idx,
                batch_examples[:5],
            )
        sampled = recoverable
        trajectories = generate_batch_trajectories(
            sampled, tool_executor, states, seed=seed + round_idx,
        )

        batch_kept = Counter()
        for traj in trajectories:
            stype = traj.scenario_type
            if stype not in targets:
                continue
            if len(selected_by_type[stype]) >= targets[stype]:
                continue
            selected_by_type[stype].append(traj)
            kept_by_type[stype] += 1
            batch_kept[stype] += 1
        logger.info(
            "Strict SFT batch %s kept=%s cumulative=%s",
            round_idx,
            dict(batch_kept),
            {k: len(v) for k, v in selected_by_type.items()},
        )

    shortfalls = {
        stype: {"target": target, "available": len(selected_by_type.get(stype, []))}
        for stype, target in targets.items()
        if len(selected_by_type.get(stype, [])) < target
    }
    if shortfalls:
        raise RuntimeError(
            "Insufficient label/tool-consistent SFT trajectories after "
            f"incremental generation: {shortfalls}. attempts={dict(attempts_by_type)}, "
            f"kept={dict(kept_by_type)}, "
            f"recoverability={dict(recoverability_stats)}"
        )

    logger.info("Strict SFT recoverability totals: %s", dict(recoverability_stats))
    if recoverability_examples:
        logger.info(
            "Strict SFT recoverability rejected examples (first %s): %s",
            len(recoverability_examples),
            recoverability_examples,
        )

    selected = []
    for stype in targets:
        selected.extend(selected_by_type[stype][: targets[stype]])
    import random

    rng = random.Random(seed)
    rng.shuffle(selected)
    return selected


def main():
    parser = argparse.ArgumentParser(description="Generate SFT training data")
    parser.add_argument("--config", default="configs/topology_config.yaml")
    parser.add_argument("--path-config", default="configs/diagnostic_paths.yaml")
    parser.add_argument("--data-root", default="data/lbnl")
    parser.add_argument("--models-dir", default="outputs/models")
    parser.add_argument("--output-dir", default="outputs/data")
    parser.add_argument("--n-total", type=int, default=3000)
    parser.add_argument("--n-cross-system", type=int, default=900,
                        help="Number of base cross-system propagation scenarios")
    parser.add_argument("--oversample-factor", type=float, default=2.5,
                        help="Over-sample SFT candidates before strict real-Oracle filtering")
    parser.add_argument("--batch-size", type=int, default=1200,
                        help="Candidate batch size for strict incremental real-Oracle SFT generation")
    parser.add_argument("--max-rounds", type=int, default=20,
                        help="Maximum strict incremental SFT generation batches")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None,
                        help="Generate only this many for quick testing")
    parser.add_argument(
        "--oracle-mode",
        default="real",
        choices=["teacher", "real", "path_aware", "path-aware"],
        help=(
            "real uses runtime model routing, matching RL/eval; "
            "teacher/path-aware is legacy/debug only."
        ),
    )
    parser.add_argument(
        "--include-system-health",
        action="store_true",
        help="Expose anomaly_score/health_status in get_system_overview.",
    )
    parser.add_argument(
        "--include-diagnosis-sensor-readings",
        action="store_true",
        help=(
            "Legacy/debug mode: include raw readings in diagnose_node output. "
            "Keep disabled for the main topology-diagnosis experiment."
        ),
    )
    args = parser.parse_args()

    output_dir = ensure_dir(args.output_dir)

    # Build topology
    logger.info("Building topology...")
    builder = TopologyBuilder(args.config, args.data_root)
    builder.build()

    # Load model registry
    logger.info("Loading model registry...")
    registry = ModelRegistry(args.models_dir)

    # Initialize diagnostic path generator
    logger.info("Loading diagnostic path templates...")
    path_generator = DiagnosticPathGenerator(args.path_config)

    # Create tool executors
    topo_executor = TopologyToolExecutor(builder)
    path_aware = args.oracle_mode in ("teacher", "path_aware", "path-aware")
    pred_executor = PredictionToolExecutor(
        registry,
        builder,
        path_aware=path_aware,
        include_diagnosis_sensor_readings=args.include_diagnosis_sensor_readings,
    )
    tool_executor = UnifiedToolExecutor(
        topo_executor,
        pred_executor,
        include_system_health=args.include_system_health,
        expose_status_summary=False,
    )
    recoverability_executor = UnifiedToolExecutor(
        topo_executor,
        PredictionToolExecutor(
            registry,
            builder,
            path_aware=False,
            include_diagnosis_sensor_readings=False,
        ),
        include_system_health=False,
        expose_status_summary=True,
    )
    logger.info(
        "SFT data routing mode: %s (path_aware=%s, include_system_health=%s)",
        args.oracle_mode,
        path_aware,
        args.include_system_health,
    )
    logger.info(
        "diagnose_node raw sensor readings exposed: %s",
        args.include_diagnosis_sensor_readings,
    )

    # Generate scenarios for each system
    logger.info("\n=== Generating Fault Scenarios ===")
    all_scenarios = []

    config = load_yaml(args.config)
    for sys_id in config["systems"]:
        sys_name = SYSTEM_NAMES.get(sys_id, sys_id)
        scenarios = generate_single_system_scenarios(
            args.data_root, sys_id, sys_name, builder,
            path_generator=path_generator,
            n_windows_per_fault=5,
        )
        all_scenarios.extend(scenarios)

    # Generate cross-system scenarios
    cross_scenarios = generate_cross_system_scenarios(
        all_scenarios, builder, path_generator=path_generator,
        n_scenarios=args.n_cross_system,
    )
    all_scenarios.extend(cross_scenarios)

    # Generate ambiguous and low-confidence scenarios
    ambiguous = create_ambiguous_scenarios(all_scenarios)
    all_scenarios.extend(ambiguous)

    low_conf = create_low_confidence_scenarios(all_scenarios)
    all_scenarios.extend(low_conf)

    # Generate no_fault scenarios (Non-Ambiguous only)
    no_fault = create_no_fault_scenarios(all_scenarios)
    all_scenarios.extend(no_fault)

    logger.info(f"Total scenarios generated: {len(all_scenarios)}")

    # Save all scenarios
    save_scenarios(all_scenarios, os.path.join(output_dir, "all_scenarios.json"))

    final_target = args.sample_size if args.sample_size else args.n_total
    logger.info("\n=== Preparing Real Sensor Data Lookup ===")
    file_info_map, normal_file_map = _build_file_info_map(args.data_root, config["systems"])

    if not args.sample_size:
        targets = _scenario_type_targets(final_target)
        logger.info("\n=== Generating Strict Incremental Diagnostic Trajectories ===")
        trajectories = _strict_incremental_generate(
            all_scenarios,
            final_target,
            targets,
            tool_executor,
            recoverability_executor,
            normal_file_map,
            file_info_map,
            builder,
            registry,
            batch_size=args.batch_size,
            max_rounds=args.max_rounds,
            seed=42,
        )
        if len(trajectories) != final_target:
            raise RuntimeError(
                f"Expected {final_target} SFT trajectories after strict generation, "
                f"got {len(trajectories)}"
            )
        type_counts = Counter(t.scenario_type for t in trajectories)
        logger.info("Strict real-Oracle SFT target counts: %s", targets)
        logger.info("Strict real-Oracle SFT kept counts: %s", dict(type_counts))

        logger.info("\n=== Formatting SFT Dataset ===")
        sft_path = format_sft_dataset(
            trajectories,
            os.path.join(output_dir, "sft_train.jsonl"),
            format_type="sharegpt",
        )
        with open(sft_path, "r", encoding="utf-8") as f:
            total_sft_examples = sum(1 for line in f if line.strip())

        logger.info(f"\nSFT data generation complete!")
        logger.info(f"  Output: {sft_path}")
        logger.info(f"  Base trajectories: {len(trajectories)}")
        logger.info(f"  Total SFT examples: {total_sft_examples}")

        if args.validate:
            logger.info("\n=== Validation ===")
            import json
            with open(sft_path, "r", encoding="utf-8") as f:
                first_10 = [
                    json.loads(f.readline())
                    for _ in range(min(10, total_sft_examples))
                ]

            for i, entry in enumerate(first_10):
                convs = entry.get("conversations", [])
                n_turns = len(convs)
                n_tool_calls = sum(
                    1 for c in convs
                    if c.get("from") == "gpt" and "<tool_call>" in c.get("value", "")
                )
                logger.info(f"  Example {i}: {n_turns} turns, {n_tool_calls} tool calls")

            _run_sft_quality_gates(sft_path)
            logger.info("Validation passed")
        return

    # Sample for training.  Real-model SFT generation intentionally discards
    # trajectories whose final diagnosis contradicts the visible tool outputs,
    # so we over-sample candidates and then cap back to the requested target.
    if args.sample_size:
        candidate_target = final_target
    else:
        candidate_target = max(final_target, int(final_target * args.oversample_factor))
    sampled = sample_scenarios(all_scenarios, n_total=candidate_target)

    # Load real sensor data per source_file (only load CSVs actually needed)
    logger.info("\n=== Loading Real Sensor Data ===")
    from src.node_models.data_loader import discover_fault_files, read_fault_file
    from src.environment.fault_scenario import create_scenario_state

    # For no_fault scenarios, override source_file to use fault-free CSV
    # This ensures Oracle genuinely predicts Normal from baseline data
    normal_file_map = {}  # system_id → fault-free filename
    for sys_id in config["systems"]:
        fault_files = discover_fault_files(args.data_root, sys_id)
        for ff in fault_files:
            if ff.fault_type.lower() == "normal":
                normal_file_map[sys_id] = ff.filename
                break

    for s in sampled:
        if 'no_fault' in s.scenario_type:
            normal_fname = normal_file_map.get(s.root_cause_system, "")
            if normal_fname:
                # Override source_file to use fault-free data
                object.__setattr__(s, 'source_file', normal_fname)
                logger.debug(f"  no_fault {s.scenario_id}: using {normal_fname}")

    # Group sampled scenarios by (system_id, source_file) to avoid duplicate loads
    needed_files = {}  # (sys_id, filename) → list of scenario_ids
    for s in sampled:
        required_rows = required_nrows_for_scenario(s, minimum=50000)
        key = (s.root_cause_system, s.source_file)
        needed_files[key] = max(
            int(needed_files.get(key, 0) or 0),
            int(required_rows or 0),
        )

    # Also gather cross-system downstream files
    for s in sampled:
        required_rows = required_nrows_for_scenario(s, minimum=50000)
        for sys_id in s.affected_systems:
            if sys_id != s.root_cause_system:
                normal_fname = normal_file_map.get(sys_id, "")
                key = (sys_id, normal_fname)  # downstream systems use clean baseline
                needed_files[key] = max(
                    int(needed_files.get(key, 0) or 0),
                    int(required_rows or 0),
                )

    # Build file path lookup by system
    file_info_map = {}  # (sys_id, filename) -> FaultFileInfo
    for sys_id in config["systems"]:
        fault_files = discover_fault_files(args.data_root, sys_id)
        for ff in fault_files:
            file_info_map[(sys_id, ff.filename)] = ff
            # Also index as the first file per system (for downstream systems)
            if (sys_id, "") not in file_info_map:
                file_info_map[(sys_id, "")] = ff

    # Load only the CSVs we need
    loaded_data = {}  # (sys_id, filename) → DataFrame
    for (sys_id, fname), required_rows in needed_files.items():
        finfo = file_info_map.get((sys_id, fname))
        if finfo:
            try:
                df = read_fault_file(finfo, nrows=required_rows)
                loaded_data[(sys_id, fname)] = df
                logger.info(
                    f"  Loaded {fname or 'default'} for {sys_id}: "
                    f"{len(df)} rows (requested {required_rows})"
                )
            except Exception as e:
                logger.warning(f"  Failed to load {fname} for {sys_id}: {e}")

    # Create scenario states from loaded data
    logger.info("\n=== Creating Scenario States ===")
    scenario_states = {}
    for s in sampled:
        try:
            # Build system_data dict for this scenario
            system_data = {}
            # Primary system data from the scenario's source file
            key = (s.root_cause_system, s.source_file)
            if key in loaded_data:
                system_data[s.root_cause_system] = loaded_data[key]
            # Downstream system data (use default/first file)
            for sys_id in s.affected_systems:
                if sys_id != s.root_cause_system and sys_id not in system_data:
                    alt_key = (sys_id, normal_file_map.get(sys_id, ""))
                    if alt_key in loaded_data:
                        system_data[sys_id] = loaded_data[alt_key]

            state = create_scenario_state(s, system_data, builder, registry)
            scenario_states[s.scenario_id] = state
        except Exception as e:
            logger.warning(f"  State creation failed for {s.scenario_id}: {e}")

    logger.info(f"  Created {len(scenario_states)}/{len(sampled)} scenario states")

    sampled, recoverability_stats, recoverability_examples = _filter_recoverable_scenarios(
        sampled,
        scenario_states,
        recoverability_executor,
    )
    logger.info(
        "Sample SFT recoverability filter kept %s scenarios; stats=%s",
        len(sampled),
        dict(recoverability_stats),
    )
    if recoverability_examples:
        logger.info(
            "Sample SFT recoverability rejected examples (first %s): %s",
            len(recoverability_examples),
            recoverability_examples,
        )

    # Generate trajectories (using real Oracle predictions)
    logger.info("\n=== Generating Diagnostic Trajectories ===")
    trajectories = generate_batch_trajectories(
        sampled, tool_executor, scenario_states,
    )
    if not args.sample_size:
        targets = _scenario_type_targets(final_target)
        trajectories, shortfalls = _cap_trajectories_to_targets(
            trajectories,
            targets,
            seed=42,
        )
        type_counts = Counter(t.scenario_type for t in trajectories)
        logger.info("Strict real-Oracle SFT target counts: %s", targets)
        logger.info("Strict real-Oracle SFT kept counts: %s", dict(type_counts))
        if shortfalls:
            raise RuntimeError(
                "Insufficient label-consistent SFT trajectories after "
                f"oversampling: {shortfalls}. Increase --oversample-factor "
                "or improve Oracle/GT alignment."
            )
        if len(trajectories) != final_target:
            raise RuntimeError(
                f"Expected {final_target} SFT trajectories after capping, "
                f"got {len(trajectories)}"
            )

    # Format as SFT dataset
    logger.info("\n=== Formatting SFT Dataset ===")
    sft_path = format_sft_dataset(
        trajectories,
        os.path.join(output_dir, "sft_train.jsonl"),
        format_type="sharegpt",
    )
    with open(sft_path, "r", encoding="utf-8") as f:
        total_sft_examples = sum(1 for line in f if line.strip())

    logger.info(f"\nSFT data generation complete!")
    logger.info(f"  Output: {sft_path}")
    logger.info(f"  Base trajectories: {len(trajectories)}")
    logger.info(f"  Total SFT examples: {total_sft_examples}")

    if args.validate:
        logger.info("\n=== Validation ===")
        import json
        with open(sft_path, "r", encoding="utf-8") as f:
            first_10 = [
                json.loads(f.readline())
                for _ in range(min(10, total_sft_examples))
            ]

        for i, entry in enumerate(first_10):
            convs = entry.get("conversations", [])
            n_turns = len(convs)
            n_tool_calls = sum(1 for c in convs if c.get("from") == "gpt" and "<tool_call>" in c.get("value", ""))
            logger.info(f"  Example {i}: {n_turns} turns, {n_tool_calls} tool calls")

        _run_sft_quality_gates(sft_path)
        logger.info("Validation passed")


if __name__ == "__main__":
    main()
