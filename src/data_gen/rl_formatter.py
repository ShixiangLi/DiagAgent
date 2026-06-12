"""
RL Formatter — Format scenarios for reinforcement learning training.

Produces RL prompts (user query + tool definitions only, no oracle trajectory)
and ground truth files for reward computation.
"""

import json
import os
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.data_gen.sft_formatter import (
    SFT_DESIGN_VERSION,
    format_system_prompt,
    system_prompt_sha256,
)
from src.environment.fault_scenario import FaultScenario
from src.utils.io_utils import save_json, setup_logger

logger = setup_logger(__name__)


def _norm_group_value(value: Any) -> str:
    return str(value or "").strip().lower()


def _is_no_fault_scenario(scenario: FaultScenario) -> bool:
    return (
        "no_fault" in _norm_group_value(scenario.scenario_type)
        or _norm_group_value(scenario.fault_type) in ("normal", "no_fault", "none")
        or _norm_group_value(scenario.root_cause_node) in ("none", "")
    )


def scenario_group_key(scenario: FaultScenario) -> Tuple[str, str, str, str]:
    """Return a leakage-safe grouping key for train/val/test splitting.

    Scenarios derived from the same physical fault file and same root-cause
    label must stay in one split. Otherwise validation/test metrics can be
    inflated by prompt variants of the same underlying fault already appearing
    in training.
    """
    if _is_no_fault_scenario(scenario):
        return (
            _norm_group_value(scenario.root_cause_system),
            "none",
            "normal",
            _norm_group_value(scenario.source_file),
        )
    return (
        _norm_group_value(scenario.root_cause_system),
        _norm_group_value(scenario.root_cause_node),
        _norm_group_value(scenario.fault_type),
        _norm_group_value(scenario.source_file),
    )


def format_rl_prompt(scenario: FaultScenario) -> Dict[str, Any]:
    """
    Format a scenario as an RL prompt (input only, no oracle output).

    Args:
        scenario: The fault scenario.

    Returns:
        Dict with prompt messages and ground truth for reward computation.
    """
    messages = [
        {"role": "system", "content": format_system_prompt()},
        {"role": "user", "content": scenario.description},
    ]

    # Compute optimal path length from diagnostic path if available
    if scenario.diagnostic_path is not None:
        optimal_path_length = scenario.diagnostic_path.path_length
        diagnostic_path_nodes = scenario.diagnostic_path.node_ids
        symptom_node = scenario.diagnostic_path.symptom_node
    elif scenario.optimal_path:
        optimal_path_length = len([p for p in scenario.optimal_path if p])
        diagnostic_path_nodes = []
        symptom_node = ""
    else:
        optimal_path_length = 3  # Fallback
        diagnostic_path_nodes = []
        symptom_node = ""

    is_no_fault = (
        "no_fault" in scenario.scenario_type
        or str(scenario.fault_type).lower() in ("normal", "no_fault")
        or str(scenario.root_cause_node).lower() in ("none", "")
    )

    root_cause_node = "none" if is_no_fault else scenario.root_cause_node
    fault_type = "Normal" if is_no_fault else scenario.fault_type
    fault_intensity = "none" if is_no_fault else scenario.fault_intensity
    split_group_key = "|".join(scenario_group_key(scenario))

    # Reference propagation edges for the TR metric (eq:exp_tr): consecutive
    # node pairs along the ordered symptom->root diagnostic path.
    reference_propagation_edges = [
        [diagnostic_path_nodes[i], diagnostic_path_nodes[i + 1]]
        for i in range(len(diagnostic_path_nodes) - 1)
    ]

    return {
        "id": scenario.scenario_id,
        "messages": messages,
        "ground_truth": {
            "root_cause_system": scenario.root_cause_system,
            "root_cause_node": root_cause_node,
            "fault_type": fault_type,
            "fault_intensity": fault_intensity,
            "affected_systems": scenario.affected_systems,
            "optimal_path_length": optimal_path_length,
            "diagnostic_path_nodes": diagnostic_path_nodes,
            "symptom_node": symptom_node,
            "reference_review_nodes": diagnostic_path_nodes if is_no_fault else [],
            "reference_propagation_edges": reference_propagation_edges,
        },
        "metadata": {
            "design_version": SFT_DESIGN_VERSION,
            "system_prompt_sha256": system_prompt_sha256(),
            "scenario_id": scenario.scenario_id,
            "scenario_type": scenario.scenario_type,
            "difficulty": scenario.difficulty,
            "source_file": scenario.source_file,
            "split_group_key": split_group_key,
            "time_window_start": scenario.time_window_start,
            "time_window_end": scenario.time_window_end,
            "diagnostic_path_nodes": diagnostic_path_nodes,
        },
    }


def format_rl_dataset(
    scenarios: List[FaultScenario],
    output_path: str,
) -> str:
    """
    Format all scenarios into an RL training dataset.

    Args:
        scenarios: List of fault scenarios.
        output_path: Path to save the JSONL file.

    Returns:
        Path to the saved file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for scenario in scenarios:
            entry = format_rl_prompt(scenario)
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(scenarios)} RL prompts to {output_path}")

    # Summary
    from collections import Counter
    type_counts = Counter(s.scenario_type for s in scenarios)
    user_prompts = [s.description for s in scenarios]
    summary = {
        "design_version": SFT_DESIGN_VERSION,
        "system_prompt_sha256": system_prompt_sha256(),
        "total_prompts": len(scenarios),
        "scenario_type_counts": dict(type_counts),
        "unique_user_prompts": len(set(user_prompts)),
        "unique_user_prompt_rate": (
            len(set(user_prompts)) / max(len(user_prompts), 1)
        ),
        "leakage_group_count": len({scenario_group_key(s) for s in scenarios}),
    }
    summary_path = output_path.replace(".jsonl", "_summary.json")
    save_json(summary, summary_path)

    return output_path


def split_rl_data(
    scenarios: List[FaultScenario],
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    random_state: int = 42,
) -> Dict[str, List[FaultScenario]]:
    """
    Split scenarios into train/val/test sets with leakage-safe grouping.

    The split is stratified by scenario type as far as possible, but the hard
    constraint is that all prompt variants sharing the same physical fault
    group (system, root node, fault label, source CSV) stay in the same split.

    Returns:
        Dict with 'train', 'val', 'test' keys.
    """
    import random
    rng = random.Random(random_state)

    if not scenarios:
        return {"train": [], "val": [], "test": []}

    total_frac = train_frac + val_frac + test_frac
    if total_frac <= 0:
        raise ValueError("train/val/test fractions must sum to a positive value")
    fractions = {
        "train": train_frac / total_frac,
        "val": val_frac / total_frac,
        "test": test_frac / total_frac,
    }

    total_type_counts = Counter(s.scenario_type for s in scenarios)
    type_targets = {
        split: {
            stype: total_type_counts[stype] * frac
            for stype in total_type_counts
        }
        for split, frac in fractions.items()
    }
    size_targets = {
        split: len(scenarios) * frac for split, frac in fractions.items()
    }

    grouped: Dict[Tuple[str, str, str, str], List[FaultScenario]] = defaultdict(list)
    for s in scenarios:
        grouped[scenario_group_key(s)].append(s)

    splits = {"train": [], "val": [], "test": []}
    split_type_counts = {split: Counter() for split in splits}
    split_sizes = Counter()

    groups = list(grouped.values())
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    def _remaining_ratio_score(projected: int, target: float) -> float:
        target = max(target, 1.0)
        if projected <= target:
            return -(target - projected) / target
        return 5.0 * (projected - target) / target

    def _score(split: str, group: List[FaultScenario]) -> float:
        group_counts = Counter(s.scenario_type for s in group)
        type_score = 0.0
        for stype, count in group_counts.items():
            target = max(type_targets[split].get(stype, 0.0), 1.0)
            projected = split_type_counts[split][stype] + count
            type_score += _remaining_ratio_score(projected, target)
        type_score /= max(len(group_counts), 1)

        projected_size = split_sizes[split] + len(group)
        size_score = _remaining_ratio_score(projected_size, size_targets[split])
        return 0.75 * size_score + 0.25 * type_score

    for group in groups:
        best_split = min(("train", "val", "test"), key=lambda split: _score(split, group))
        splits[best_split].extend(group)
        split_sizes[best_split] += len(group)
        split_type_counts[best_split].update(s.scenario_type for s in group)

    def _remove_group(split: str, group: List[FaultScenario]) -> None:
        key = scenario_group_key(group[0])
        splits[split] = [
            s for s in splits[split]
            if scenario_group_key(s) != key
        ]
        split_sizes[split] -= len(group)
        for s in group:
            split_type_counts[split][s.scenario_type] -= 1

    def _add_group(split: str, group: List[FaultScenario]) -> None:
        splits[split].extend(group)
        split_sizes[split] += len(group)
        split_type_counts[split].update(s.scenario_type for s in group)

    def _group_cross_count(
        group: List[FaultScenario],
        root_system: Optional[str] = None,
    ) -> int:
        return sum(
            1 for s in group
            if "cross_system" in _norm_group_value(s.scenario_type)
            and (
                root_system is None
                or _norm_group_value(s.root_cause_system) == root_system
            )
        )

    def _split_group_keys(split: str) -> set:
        return {scenario_group_key(s) for s in splits[split]}

    def _cross_root_counts(rows: List[FaultScenario]) -> Counter:
        counts = Counter()
        for s in rows:
            if "cross_system" in _norm_group_value(s.scenario_type):
                counts[_norm_group_value(s.root_cause_system)] += 1
        return counts

    # Validation/test selection must see both central-plant cross-system
    # roots.  Without this, a best-checkpoint gate can optimize one plant and
    # still collapse on the other while reporting acceptable validation metrics.
    grouped_by_key = {
        scenario_group_key(group[0]): group
        for group in groups
    }
    needed_cross_roots = {
        _norm_group_value(s.root_cause_system)
        for s in scenarios
        if "cross_system" in _norm_group_value(s.scenario_type)
    }
    for target_split in ("val", "test"):
        for _ in range(12):
            counts = _cross_root_counts(splits[target_split])
            cross_total = sum(counts.values())
            if cross_total <= 0:
                break

            min_per_root = max(20, int(round(0.20 * cross_total)))
            under_roots = [
                root for root in sorted(needed_cross_roots)
                if counts.get(root, 0) < min_per_root
            ]
            if not under_roots:
                break

            under_root = min(under_roots, key=lambda root: counts.get(root, 0))
            over_root = max(counts, key=lambda root: counts[root])
            if counts.get(over_root, 0) <= min_per_root:
                break

            train_keys = _split_group_keys("train")
            target_keys = _split_group_keys(target_split)
            needed = min_per_root - counts.get(under_root, 0)

            source_candidates = [
                group for key, group in grouped_by_key.items()
                if key in train_keys and _group_cross_count(group, under_root) > 0
            ]
            if not source_candidates:
                break
            source_candidates.sort(
                key=lambda group: (
                    _group_cross_count(group, under_root) < needed,
                    abs(_group_cross_count(group, under_root) - needed),
                    abs(len(group) - size_targets[target_split] * 0.08),
                )
            )
            source_group = source_candidates[0]
            source_cross = _group_cross_count(source_group, under_root)

            donor_candidates = []
            for key, group in grouped_by_key.items():
                if key not in target_keys:
                    continue
                donor_cross = _group_cross_count(group, over_root)
                if donor_cross <= 0:
                    continue
                if counts[over_root] - donor_cross < min_per_root:
                    continue
                donor_candidates.append(group)

            donor_group = None
            if donor_candidates:
                donor_candidates.sort(
                    key=lambda group: (
                        abs(_group_cross_count(group, over_root) - source_cross),
                        abs(len(group) - len(source_group)),
                    )
                )
                donor_group = donor_candidates[0]

            _remove_group("train", source_group)
            _add_group(target_split, source_group)
            if donor_group is not None:
                _remove_group(target_split, donor_group)
                _add_group("train", donor_group)

    for key in splits:
        rng.shuffle(splits[key])

    logger.info(
        f"RL data split: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )
    group_sets = {
        name: {scenario_group_key(s) for s in rows}
        for name, rows in splits.items()
    }
    logger.info(
        "RL split leakage groups: "
        f"train={len(group_sets['train'])}, val={len(group_sets['val'])}, "
        f"test={len(group_sets['test'])}, "
        f"train-val overlap={len(group_sets['train'] & group_sets['val'])}, "
        f"train-test overlap={len(group_sets['train'] & group_sets['test'])}, "
        f"val-test overlap={len(group_sets['val'] & group_sets['test'])}"
    )
    return splits
