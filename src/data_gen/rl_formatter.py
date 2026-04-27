"""
RL Formatter — Format scenarios for reinforcement learning training.

Produces RL prompts (user query + tool definitions only, no oracle trajectory)
and ground truth files for reward computation.
"""

import json
import os
from typing import Any, Dict, List, Optional

from src.data_gen.sft_formatter import format_system_prompt
from src.environment.fault_scenario import FaultScenario
from src.utils.io_utils import save_json, setup_logger

logger = setup_logger(__name__)


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
    elif scenario.optimal_path:
        optimal_path_length = len([p for p in scenario.optimal_path if p])
    else:
        optimal_path_length = 3  # Fallback

    is_no_fault = (
        "no_fault" in scenario.scenario_type
        or str(scenario.fault_type).lower() in ("normal", "no_fault")
        or str(scenario.root_cause_node).lower() in ("none", "")
    )

    root_cause_node = "none" if is_no_fault else scenario.root_cause_node
    fault_type = "Normal" if is_no_fault else scenario.fault_type
    fault_intensity = "none" if is_no_fault else scenario.fault_intensity

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
        },
        "metadata": {
            "scenario_id": scenario.scenario_id,
            "scenario_type": scenario.scenario_type,
            "difficulty": scenario.difficulty,
            "source_file": scenario.source_file,
            "time_window_start": scenario.time_window_start,
            "time_window_end": scenario.time_window_end,
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
    summary = {
        "total_prompts": len(scenarios),
        "scenario_type_counts": dict(type_counts),
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
    Split scenarios into train/val/test sets with stratification.

    Returns:
        Dict with 'train', 'val', 'test' keys.
    """
    import random
    rng = random.Random(random_state)

    # Group by scenario type for stratified split
    from collections import defaultdict
    by_type = defaultdict(list)
    for s in scenarios:
        by_type[s.scenario_type].append(s)

    splits = {"train": [], "val": [], "test": []}

    for stype, group in by_type.items():
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        splits["train"].extend(group[:n_train])
        splits["val"].extend(group[n_train:n_train + n_val])
        splits["test"].extend(group[n_train + n_val:])

    for key in splits:
        rng.shuffle(splits[key])

    logger.info(
        f"RL data split: train={len(splits['train'])}, "
        f"val={len(splits['val'])}, test={len(splits['test'])}"
    )
    return splits
