"""
Evaluator — Run evaluation pipeline with real Oracle environment.

Executes diagnostic episodes using different models (base, SFT, RL)
against a standardized test set, with real tool execution through the
PredictionToolExecutor and Oracle model inference.
"""

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from src.evaluation.metrics import compute_all_metrics
from src.environment.scenario_matching import match_scenario
from src.utils.io_utils import save_json, ensure_dir, setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Scenario Sampling
# ============================================================================

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Load a JSONL file into memory."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _scenario_type(row: Dict[str, Any]) -> str:
    """Return the scenario type used for stratified evaluation."""
    return row.get("metadata", {}).get("scenario_type", "unknown")


def stratified_sample_scenarios(
    scenarios: List[Dict[str, Any]],
    max_episodes: int,
    seed: int = 42,
    strategy: str = "stratified",
) -> List[Dict[str, Any]]:
    """
    Sample evaluation scenarios while preserving scenario-type proportions.

    This is intended for in-training eval steps: it keeps the episode count
    small, but avoids a biased first-N slice of the eval file.  When
    ``max_episodes`` covers the whole set, the original set is returned.
    """
    if max_episodes <= 0 or len(scenarios) <= max_episodes:
        return list(scenarios)
    if strategy != "stratified":
        import random
        rng = random.Random(seed)
        sampled = list(scenarios)
        rng.shuffle(sampled)
        return sampled[:max_episodes]

    import random
    from collections import defaultdict

    rng = random.Random(seed)
    by_type: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for scenario in scenarios:
        by_type[_scenario_type(scenario)].append(scenario)

    for entries in by_type.values():
        rng.shuffle(entries)

    total = len(scenarios)
    types = sorted(by_type)
    target = {
        stype: max_episodes * len(by_type[stype]) / total
        for stype in types
    }
    allocation = {stype: 0 for stype in types}

    # If possible, include every type at least once.
    if max_episodes >= len(types):
        for stype in types:
            allocation[stype] = 1

    while sum(allocation.values()) < max_episodes:
        candidates = [
            stype for stype in types
            if allocation[stype] < len(by_type[stype])
        ]
        if not candidates:
            break
        stype = max(
            candidates,
            key=lambda t: (target[t] - allocation[t], len(by_type[t]), t),
        )
        allocation[stype] += 1

    sampled: List[Dict[str, Any]] = []
    for stype in types:
        sampled.extend(by_type[stype][:allocation[stype]])
    rng.shuffle(sampled)
    return sampled


def _log_eval_sample_distribution(
    scenarios: List[Dict[str, Any]],
    total_available: int,
    strategy: str,
) -> None:
    """Log the scenario distribution selected for an evaluation step."""
    from collections import Counter

    counts = Counter(_scenario_type(s) for s in scenarios)
    logger.info(
        f"  Evaluation sample: {len(scenarios)}/{total_available} episodes "
        f"({strategy}) - "
        + ", ".join(f"{t}={n}" for t, n in sorted(counts.items()))
    )


# ============================================================================
# Test Data Preparation
# ============================================================================

def prepare_test_scenarios(
    sft_data_path: str,
    output_path: str,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> str:
    """
    Extract test scenarios from SFT data in evaluator-compatible format.

    Converts ShareGPT format to the evaluation format:
    {
        "id": str,
        "messages": [{"role": ..., "content": ...}],  # only initial user msg
        "ground_truth": {...},
        "metadata": {...},
    }

    Args:
        sft_data_path: Path to sft_train.jsonl.
        output_path: Path to write test scenarios JSONL.
        test_ratio: Fraction of data to sample for testing.
        seed: Random seed.

    Returns:
        Path to the generated test file.
    """
    import random
    rng = random.Random(seed)

    all_data = []
    with open(sft_data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_data.append(json.loads(line))

    # Stratified sampling by scenario_type
    by_type = {}
    for d in all_data:
        st = d["metadata"]["scenario_type"]
        by_type.setdefault(st, []).append(d)

    test_scenarios = []
    for stype, entries in by_type.items():
        rng.shuffle(entries)
        n_test = max(1, int(len(entries) * test_ratio))
        for entry in entries[:n_test]:
            # Extract only the initial user message (the diagnostic query)
            user_msg = None
            system_prompt = entry.get("system", "")
            for c in entry["conversations"]:
                if c["from"] == "human":
                    user_msg = c["value"]
                    break

            if user_msg is None:
                continue

            gt = entry["metadata"]["ground_truth"]
            meta = entry["metadata"]

            test_scenarios.append({
                "id": entry["id"],
                "system_prompt": system_prompt,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_msg},
                ],
                "ground_truth": {
                    "root_cause_system": gt.get("root_cause_system", ""),
                    "root_cause_node": gt.get("root_cause_node", ""),
                    "fault_type": gt.get("fault_type", ""),
                    "fault_intensity": gt.get("fault_intensity", ""),
                    "optimal_path_length": meta.get("path_length", 3),
                },
                "metadata": {
                    "scenario_type": meta.get("scenario_type", ""),
                    "scenario_id": meta.get("scenario_id", ""),
                    "difficulty": meta.get("difficulty", "medium"),
                    "n_expected_tool_calls": meta.get("n_tool_calls", 5),
                },
            })

    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as f:
        for s in test_scenarios:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(
        f"Prepared {len(test_scenarios)} test scenarios "
        f"from {len(all_data)} total samples"
    )
    for stype, entries in by_type.items():
        n = sum(1 for s in test_scenarios if s["metadata"]["scenario_type"] == stype)
        logger.info(f"  {stype}: {n} test scenarios")

    return output_path


# ============================================================================
# Real Oracle Environment
# ============================================================================

def _create_tool_environment():
    """
    Create the real Oracle tool execution environment.

    Loads topology, Oracle models, and both topology and prediction tools
    via UnifiedToolExecutor so that all agent tool calls are handled.

    Returns:
        Tuple of (topology_builder, tool_executor, registry) or (None, None, None) on failure.
    """
    try:
        from src.topology.topology_builder import TopologyBuilder
        from src.topology.topology_tools import TopologyToolExecutor
        from src.node_models.prediction_tools import PredictionToolExecutor
        from src.node_models.model_registry import ModelRegistry
        from src.environment.tool_executor import UnifiedToolExecutor

        config_path = "configs/topology_config.yaml"
        data_root = "data/lbnl"

        builder = TopologyBuilder(config_path, data_root)
        builder.build()

        registry = ModelRegistry("outputs/models")

        topo_executor = TopologyToolExecutor(builder)
        pred_executor = PredictionToolExecutor(registry, builder)
        executor = UnifiedToolExecutor(topo_executor, pred_executor)

        logger.info(
            f"Oracle environment loaded: "
            f"{len(executor.get_available_tools())} tools available "
            f"({executor.get_available_tools()})"
        )
        return builder, executor, registry

    except Exception as e:
        logger.warning(f"Failed to create Oracle environment: {e}")
        return None, None, None


def _execute_tool_call(executor, tool_name: str, tool_args: dict) -> str:
    """Execute a tool call through the real Oracle environment."""
    try:
        result = executor.execute(tool_name, tool_args)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": str(e),
        })


# ============================================================================
# Model Evaluation
# ============================================================================

def run_model_evaluation(
    model_name: str,
    model_path: str,
    test_scenarios_path: str,
    output_dir: str,
    max_episodes: int = 200,
    max_steps: int = 15,
    allow_mock: bool = False,
    sampling_strategy: str = "stratified",
    sampling_seed: int = 42,
) -> Dict[str, Any]:
    """
    Evaluate a single model on the test set with real Oracle environment.

    Args:
        model_name: Human-readable model name (e.g., "base", "sft", "rl").
        model_path: Path to model checkpoint.
        test_scenarios_path: Path to test scenarios JSONL.
        output_dir: Directory for evaluation outputs.
        max_episodes: Maximum number of episodes to run.
        max_steps: Maximum tool calls per episode.

    Returns:
        Dict with metrics and episode-level results.
    """
    ensure_dir(output_dir)
    logger.info(f"=== Evaluating model: {model_name} ===")
    logger.info(f"  Model path: {model_path}")

    # Ensure test data exists
    if not os.path.exists(test_scenarios_path):
        logger.info("Test data not found. Generating from SFT data...")
        sft_path = "outputs/data/sft_train.jsonl"
        if os.path.exists(sft_path):
            prepare_test_scenarios(sft_path, test_scenarios_path)
        else:
            raise FileNotFoundError(
                f"Neither test data ({test_scenarios_path}) "
                f"nor SFT data ({sft_path}) found"
            )

    # Load and sample test scenarios.
    all_scenarios = _load_jsonl(test_scenarios_path)
    scenarios = stratified_sample_scenarios(
        all_scenarios,
        max_episodes=max_episodes,
        seed=sampling_seed,
        strategy=sampling_strategy,
    )
    _log_eval_sample_distribution(
        scenarios,
        total_available=len(all_scenarios),
        strategy=sampling_strategy,
    )

    # Create Oracle environment
    builder, tool_executor, model_registry = _create_tool_environment()
    if tool_executor is None:
        raise RuntimeError("Oracle environment is unavailable; refusing non-Oracle evaluation")

    # Run episodes
    episodes = []
    try:
        episodes = _run_episodes_with_model(
            model_path, scenarios, max_steps, tool_executor,
            builder=builder, model_registry=model_registry,
        )
    except ImportError:
        if not allow_mock:
            raise
        logger.warning("GPU/transformers not available. Running with mock evaluation.")
        episodes = _create_mock_episodes(scenarios, model_name, tool_executor)
    except Exception as e:
        if not allow_mock:
            raise RuntimeError(
                f"Model evaluation failed for {model_name}: {type(e).__name__}: {e}"
            ) from e
        logger.warning(
            f"Model loading failed ({type(e).__name__}: {e}). Running with mock evaluation."
        )
        episodes = _create_mock_episodes(scenarios, model_name, tool_executor)

    # Compute metrics
    metrics = compute_all_metrics(episodes)

    # Per-scenario-type breakdown
    from collections import defaultdict
    by_type = defaultdict(list)
    for ep in episodes:
        by_type[ep.get("scenario_type", "unknown")].append(ep)

    type_metrics = {}
    for stype, type_eps in by_type.items():
        type_metrics[stype] = compute_all_metrics(type_eps)

    # By difficulty breakdown
    by_diff = defaultdict(list)
    for ep in episodes:
        by_diff[ep.get("difficulty", "unknown")].append(ep)

    diff_metrics = {}
    for diff, diff_eps in by_diff.items():
        diff_metrics[diff] = compute_all_metrics(diff_eps)

    result = {
        "model_name": model_name,
        "model_path": model_path,
        "n_episodes": len(episodes),
        "sampling_strategy": sampling_strategy,
        "sampling_seed": sampling_seed,
        "overall_metrics": metrics,
        "metrics_by_scenario_type": type_metrics,
        "metrics_by_difficulty": diff_metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Save results
    result_path = os.path.join(output_dir, f"eval_{model_name}.json")
    save_json(result, result_path)
    logger.info(f"  Results saved to {result_path}")

    # Save episode details for analysis
    episodes_path = os.path.join(output_dir, f"episodes_{model_name}.jsonl")
    with open(episodes_path, "w", encoding="utf-8") as f:
        for ep in episodes:
            f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    # Log summary
    logger.info(f"  Overall Metrics for {model_name}:")
    for k, v in metrics.items():
        logger.info(f"    {k}: {v:.4f}")

    return result


def _run_episodes_with_model(
    model_path: str,
    scenarios: List[Dict],
    max_steps: int,
    tool_executor=None,
    builder=None,
    model_registry=None,
) -> List[Dict]:
    """
    Run diagnostic episodes using a real model with real Oracle environment.

    Each episode:
    1. Presents the user query to the model
    2. Model generates a response (possibly with tool call)
    3. If tool call detected, execute through PredictionToolExecutor
    4. Feed real tool result back to model
    5. Repeat until diagnosis or max_steps reached
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading model from {model_path}...")

    # Detect if this is a PEFT/LoRA checkpoint or a full model
    adapter_config = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config):
        # LoRA checkpoint: load base model + adapter
        from peft import PeftModel
        import json as _json
        with open(adapter_config, "r") as _f:
            _adapter_cfg = _json.load(_f)
        base_model_name = _adapter_cfg.get(
            "base_model_name_or_path", "Qwen/Qwen2.5-7B-Instruct"
        )
        logger.info(f"  Detected LoRA adapter, base model: {base_model_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model_name, trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        # Full model (e.g., base Qwen model from HuggingFace)
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
    model.eval()

    # Build scenario state loader if topology available
    _scenario_lookup = {}
    _file_path_map = {}
    _normal_file_map = {}
    _csv_cache = {}
    if builder is not None:
        import pandas as pd
        from dataclasses import replace
        from src.environment.fault_scenario import FaultScenario, create_scenario_state
        from src.environment.diagnostic_path import DiagnosticPath, PathNode
        from src.environment.diagnostic_path import generate_no_fault_path
        from src.node_models.data_loader import discover_fault_files

        all_scenarios_path = "outputs/data/all_scenarios.json"
        if os.path.exists(all_scenarios_path):
            raw = json.load(open(all_scenarios_path, "r", encoding="utf-8"))
            for d in raw:
                dp = d.pop("diagnostic_path", None)
                fs = FaultScenario(**d)
                if dp and isinstance(dp, dict):
                    nodes = [PathNode(**n) for n in dp.get("nodes", [])]
                    fs.diagnostic_path = DiagnosticPath(
                        nodes=nodes,
                        root_cause_node=dp.get("root_cause_node", ""),
                        fault_type=dp.get("fault_type", ""),
                        symptom_description=dp.get("symptom_description", ""),
                    )
                _scenario_lookup[fs.scenario_id] = fs

        import yaml
        topo_config = yaml.safe_load(open("configs/topology_config.yaml", "r"))
        for sys_id in topo_config.get("systems", {}):
            try:
                for ff in discover_fault_files("data/lbnl", sys_id):
                    _file_path_map[(sys_id, ff.filename)] = ff.filepath
                    if ff.is_fault_free or ff.fault_type.lower() == "normal":
                        _normal_file_map.setdefault(sys_id, ff.filename)
            except Exception:
                pass

    episodes = []
    for i, scenario in enumerate(scenarios):
        messages = scenario.get("messages", [])
        ground_truth = scenario.get("ground_truth", {})

        # Load scenario state for Oracle predictions
        sid = scenario.get("id", scenario.get("metadata", {}).get("scenario_id", ""))
        if _scenario_lookup and tool_executor is not None:
            fs, _, _ = match_scenario(
                sid,
                _scenario_lookup,
                ground_truth=ground_truth,
                scenario_type=scenario.get("metadata", {}).get("scenario_type", ""),
            )
            if fs is not None:
                gt_fault = str(ground_truth.get("fault_type", "")).lower()
                gt_node = str(ground_truth.get("root_cause_node", "")).lower()
                is_no_fault = (
                    "no_fault" in getattr(fs, "scenario_type", "")
                    or gt_fault in ("normal", "no_fault", "none")
                    or gt_node in ("none", "")
                )
                if is_no_fault:
                    normal_file = _normal_file_map.get(fs.root_cause_system)
                    if normal_file:
                        window_size = max(1, fs.time_window_end - fs.time_window_start)
                        if window_size <= 1:
                            window_size = 15
                        fs = replace(
                            fs,
                            root_cause_node="none",
                            fault_type="Normal",
                            fault_intensity="none",
                            affected_systems=[fs.root_cause_system],
                            source_file=normal_file,
                            time_window_end=fs.time_window_start + window_size,
                            diagnostic_path=generate_no_fault_path(
                                fs.root_cause_system, builder,
                            ),
                        )
                key = (fs.root_cause_system, fs.source_file)
                if key not in _csv_cache:
                    fpath = _file_path_map.get(key)
                    if fpath and os.path.exists(fpath):
                        _csv_cache[key] = pd.read_csv(
                            fpath, nrows=50000, low_memory=False
                        ).select_dtypes(include=["number"])
                if key in _csv_cache:
                    try:
                        system_data = {fs.root_cause_system: _csv_cache[key]}
                        for sys_id in fs.affected_systems:
                            if sys_id == fs.root_cause_system or sys_id in system_data:
                                continue
                            normal_file = _normal_file_map.get(sys_id)
                            if not normal_file:
                                continue
                            alt_key = (sys_id, normal_file)
                            if alt_key not in _csv_cache:
                                fpath = _file_path_map.get(alt_key)
                                if fpath and os.path.exists(fpath):
                                    _csv_cache[alt_key] = pd.read_csv(
                                        fpath, nrows=50000, low_memory=False
                                    ).select_dtypes(include=["number"])
                            if alt_key in _csv_cache:
                                system_data[sys_id] = _csv_cache[alt_key]
                        state = create_scenario_state(
                            fs, system_data,
                            builder, registry=model_registry,
                        )
                        tool_executor.set_scenario_state(state)
                    except Exception:
                        tool_executor.set_scenario_state(None)
                else:
                    tool_executor.set_scenario_state(None)
            else:
                tool_executor.set_scenario_state(None)

        agent_outputs = []
        tool_results = []
        n_tool_calls = 0
        total_outputs = 0
        max_total_outputs = max_steps * 2  # Guard against text-only loops

        # Multi-turn generation loop
        conversation = list(messages)
        for step in range(max_steps):
            prompt = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(
                output[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            # Truncate at </tool_call> to prevent hallucinated tool responses
            tc_end_pos = response.find("</tool_call>")
            if tc_end_pos != -1:
                response = response[:tc_end_pos + len("</tool_call>")]

            agent_outputs.append(response)
            conversation.append({"role": "assistant", "content": response})
            total_outputs += 1

            if total_outputs >= max_total_outputs:
                break

            # Check for tool calls
            tc_match = re.search(
                r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL
            )

            if tc_match:
                n_tool_calls += 1
                try:
                    tc = json.loads(tc_match.group(1))
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("arguments", {})

                    # Execute through real Oracle environment
                    if tool_executor is not None:
                        tool_result = _execute_tool_call(
                            tool_executor, tool_name, tool_args
                        )
                    else:
                        tool_result = json.dumps({
                            "status": "error",
                            "message": "Oracle environment not available",
                        })

                except json.JSONDecodeError:
                    tool_result = json.dumps({
                        "status": "error",
                        "message": "Invalid tool call JSON",
                    })

                tool_results.append(tool_result)
                conversation.append({"role": "tool", "content": tool_result})
            else:
                # Check for final diagnosis
                if "<diagnosis>" in response or "final diagnosis" in response.lower():
                    break

        # Parse final diagnosis
        final_diag = _parse_final_diagnosis(agent_outputs)

        episodes.append({
            "scenario_id": scenario.get("id", f"scenario_{i}"),
            "scenario_type": scenario.get("metadata", {}).get(
                "scenario_type", "unknown"
            ),
            "difficulty": scenario.get("metadata", {}).get("difficulty", "medium"),
            "ground_truth": ground_truth,
            "final_diagnosis": final_diag,
            "agent_outputs": agent_outputs,
            "tool_results": tool_results,
            "n_tool_calls": n_tool_calls,
            "optimal_path_length": ground_truth.get("optimal_path_length", 3),
        })

        if (i + 1) % 20 == 0:
            logger.info(f"  Completed {i + 1}/{len(scenarios)} episodes")

    return episodes


def run_diagnostic_eval_with_model(
    model,
    tokenizer,
    test_scenarios_path: str,
    max_episodes: int = 50,
    max_steps: int = 15,
    output_dir: Optional[str] = None,
    step: int = 0,
    sampling_strategy: str = "stratified",
    sampling_seed: int = 42,
) -> Dict[str, Any]:
    """
    Run diagnostic evaluation using an already-loaded model and tokenizer.

    Designed for in-training callbacks where the model is already on GPU.
    Uses real Oracle environment for tool execution.

    Args:
        model: The loaded model (already on GPU).
        tokenizer: The loaded tokenizer.
        test_scenarios_path: Path to test scenarios JSONL.
        max_episodes: Max episodes to evaluate (keep small for speed).
        max_steps: Max tool calls per episode.
        output_dir: Directory to save episode details for analysis.
        step: Current training step (for filename).

    Returns:
        Dict with overall metrics and per-type breakdown.
    """
    import torch

    # Load and sample test scenarios. Add the current step to the seed so
    # repeated eval steps rotate through the eval pool while preserving type
    # proportions.
    all_scenarios = _load_jsonl(test_scenarios_path)
    scenarios = stratified_sample_scenarios(
        all_scenarios,
        max_episodes=max_episodes,
        seed=sampling_seed + int(step or 0),
        strategy=sampling_strategy,
    )
    _log_eval_sample_distribution(
        scenarios,
        total_available=len(all_scenarios),
        strategy=sampling_strategy,
    )

    # Create Oracle environment
    builder, tool_executor, model_registry = _create_tool_environment()
    if tool_executor is None:
        raise RuntimeError("Oracle environment is unavailable; refusing non-Oracle diagnostic eval")

    # Load original scenarios + sensor data for Oracle predictions
    scenario_lookup = {}  # scenario_id → FaultScenario
    csv_cache = {}        # (system_id, source_file) → DataFrame
    normal_file_map = {}  # system_id -> baseline filename
    all_scenarios_path = "outputs/data/all_scenarios.json"
    try:
        if os.path.exists(all_scenarios_path):
            import pandas as pd
            from src.environment.fault_scenario import (
                FaultScenario, FaultScenarioState, create_scenario_state,
            )
            from src.node_models.data_loader import discover_fault_files

            # Load all scenarios as FaultScenario objects
            raw_data = json.load(open(all_scenarios_path, "r", encoding="utf-8"))
            for d in raw_data:
                try:
                    # Handle diagnostic_path deserialization
                    dp = d.pop("diagnostic_path", None)
                    fs = FaultScenario(**d)
                    if dp and isinstance(dp, dict):
                        from src.environment.diagnostic_path import (
                            DiagnosticPath, PathNode,
                        )
                        nodes = [PathNode(**n) for n in dp.get("nodes", [])]
                        fs.diagnostic_path = DiagnosticPath(
                            nodes=nodes,
                            root_cause_node=dp.get("root_cause_node", ""),
                            fault_type=dp.get("fault_type", ""),
                            symptom_description=dp.get("symptom_description", ""),
                        )
                    scenario_lookup[fs.scenario_id] = fs
                except Exception:
                    pass

            # Build CSV filepath lookup
            topo_config = builder.config if hasattr(builder, 'config') else {}
            systems = list(topo_config.get("systems", {}).keys()) if isinstance(topo_config, dict) else []
            if not systems:
                # Fallback: infer from scenarios
                systems = list(set(s.root_cause_system for s in scenario_lookup.values()))

            file_path_map = {}  # (sys_id, filename) → filepath
            for sys_id in systems:
                try:
                    fault_files = discover_fault_files("data/lbnl", sys_id)
                    for ff in fault_files:
                        file_path_map[(sys_id, ff.filename)] = ff.filepath
                        if ff.is_fault_free or ff.fault_type.lower() == "normal":
                            normal_file_map.setdefault(sys_id, ff.filename)
                except Exception:
                    pass

            logger.info(
                f"Loaded {len(scenario_lookup)} scenarios, "
                f"{len(file_path_map)} CSV paths indexed"
            )
    except Exception as e:
        logger.warning(f"Could not load scenario data: {e}")
        import traceback
        traceback.print_exc()

    def _load_scenario_state(sid, ground_truth=None, scenario_type: str = ""):
        """Load scenario state for a given scenario_id.

        Uses multiple matching strategies since SFT IDs differ from
        original all_scenarios.json IDs:
          SFT:      a_ddahu_VLVStuck_Cooling_20__default_2_a_single_system_510
          Original: ddahu_VLVStuck_Cooling_20__default_2

        Strategies (in order):
          1. Direct ID match
          2. Strip SFT prefixes (a_, na_lc_, a_lc_, nf_) and suffixes (_stype_N)
          3. Match by ground_truth (root_cause_system + fault_type)
        """
        fs, _, _ = match_scenario(
            sid,
            scenario_lookup,
            ground_truth=ground_truth,
            scenario_type=scenario_type,
        )

        if fs is None or builder is None:
            return None
        try:
            import pandas as pd
            from dataclasses import replace
            from src.environment.fault_scenario import create_scenario_state
            from src.environment.diagnostic_path import generate_no_fault_path

            gt_fault = str((ground_truth or {}).get("fault_type", "")).lower()
            gt_node = str((ground_truth or {}).get("root_cause_node", "")).lower()
            is_no_fault = (
                "no_fault" in getattr(fs, "scenario_type", "")
                or gt_fault in ("normal", "no_fault", "none")
                or gt_node in ("none", "")
            )
            if is_no_fault:
                normal_file = normal_file_map.get(fs.root_cause_system)
                if normal_file:
                    window_size = max(1, fs.time_window_end - fs.time_window_start)
                    if window_size <= 1:
                        window_size = 15
                    fs = replace(
                        fs,
                        root_cause_node="none",
                        fault_type="Normal",
                        fault_intensity="none",
                        affected_systems=[fs.root_cause_system],
                        source_file=normal_file,
                        time_window_end=fs.time_window_start + window_size,
                        diagnostic_path=generate_no_fault_path(
                            fs.root_cause_system, builder,
                        ),
                    )

            # Build system_data dict for this scenario
            system_data = {}
            key = (fs.root_cause_system, fs.source_file)
            if key not in csv_cache:
                fpath = file_path_map.get(key)
                if fpath and os.path.exists(fpath):
                    df = pd.read_csv(fpath, nrows=50000, low_memory=False)
                    # Select only numeric columns to avoid string conversion errors
                    numeric_df = df.select_dtypes(include=['number'])
                    csv_cache[key] = numeric_df
            if key in csv_cache:
                system_data[fs.root_cause_system] = csv_cache[key]

            for sys_id in fs.affected_systems:
                if sys_id == fs.root_cause_system or sys_id in system_data:
                    continue
                normal_file = normal_file_map.get(sys_id)
                if not normal_file:
                    continue
                alt_key = (sys_id, normal_file)
                if alt_key not in csv_cache:
                    fpath = file_path_map.get(alt_key)
                    if fpath and os.path.exists(fpath):
                        df = pd.read_csv(fpath, nrows=50000, low_memory=False)
                        csv_cache[alt_key] = df.select_dtypes(include=['number'])
                if alt_key in csv_cache:
                    system_data[sys_id] = csv_cache[alt_key]

            if not system_data:
                return None

            return create_scenario_state(fs, system_data, builder, registry=model_registry)
        except Exception as e:
            logger.debug(f"Scenario state creation failed for {sid}: {e}")
            return None

    # Run episodes with the provided model
    model.eval()
    episodes = []
    for i, scenario in enumerate(scenarios):
        messages = scenario.get("messages", [])
        ground_truth = scenario.get("ground_truth", {})

        # Set scenario state for Oracle predictions
        sid = scenario.get("id", scenario.get("metadata", {}).get("scenario_id", ""))
        stype = scenario.get("metadata", {}).get("scenario_type", "")
        state = _load_scenario_state(sid, ground_truth=ground_truth, scenario_type=stype)
        if tool_executor is not None:
            tool_executor.set_scenario_state(state)

        agent_outputs = []
        tool_results = []
        n_tool_calls = 0
        total_outputs = 0
        max_total_outputs = max_steps * 2

        conversation = list(messages)
        for turn in range(max_steps):
            prompt = tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

            response = tokenizer.decode(
                output[0][inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )

            # Truncate at </tool_call> to prevent hallucinated tool responses
            tc_end_pos = response.find("</tool_call>")
            if tc_end_pos != -1:
                response = response[:tc_end_pos + len("</tool_call>")]

            agent_outputs.append(response)
            conversation.append({"role": "assistant", "content": response})
            total_outputs += 1
            if total_outputs >= max_total_outputs:
                break

            tc_match = re.search(
                r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL
            )
            if tc_match:
                n_tool_calls += 1
                try:
                    tc = json.loads(tc_match.group(1))
                    tool_name = tc.get("name", "")
                    tool_args = tc.get("arguments", {})
                    if tool_executor is not None:
                        tool_result = _execute_tool_call(
                            tool_executor, tool_name, tool_args
                        )
                    else:
                        tool_result = json.dumps({
                            "status": "error",
                            "message": "Oracle environment not available",
                        })
                except json.JSONDecodeError:
                    tool_result = json.dumps({
                        "status": "error",
                        "message": "Invalid tool call JSON",
                    })
                tool_results.append(tool_result)
                conversation.append({"role": "tool", "content": tool_result})
            else:
                if "<diagnosis>" in response or "final diagnosis" in response.lower():
                    break

        final_diag = _parse_final_diagnosis(agent_outputs)

        # Determine correctness for real-time logging
        gt_node = str(ground_truth.get("root_cause_node", "")).lower()
        gt_fault = str(ground_truth.get("fault_type", "")).lower()
        if final_diag:
            diag_node = str(final_diag.get("root_cause_node", "")).lower()
            diag_fault = str(final_diag.get("fault_type", "")).lower()
            if gt_fault in ("normal", "no_fault", "none") or gt_node in ("none", ""):
                diag_all = (str(final_diag.get("status", "")) + " " + diag_fault).lower()
                correct = any(kw in diag_all for kw in ("normal", "no fault", "no_fault", "none"))
            else:
                node_ok = gt_node in diag_node or diag_node in gt_node
                fault_ok = gt_fault in diag_fault or diag_fault in gt_fault
                correct = node_ok and fault_ok
            status_str = "✓" if correct else "✗"
            diag_str = f"{final_diag.get('root_cause_node', '?')} / {final_diag.get('fault_type', '?')}"
        else:
            status_str = "✗ (no diagnosis)"
            diag_str = "—"

        stype = scenario.get("metadata", {}).get("scenario_type", "?")
        logger.info(
            f"  [{i+1}/{len(scenarios)}] {status_str}  "
            f"type={stype}  tools={n_tool_calls}  "
            f"GT={gt_node}/{gt_fault}  "
            f"Pred={diag_str}"
        )

        episodes.append({
            "scenario_id": scenario.get("id", f"scenario_{i}"),
            "scenario_type": stype,
            "difficulty": scenario.get("metadata", {}).get("difficulty", "medium"),
            "ground_truth": ground_truth,
            "final_diagnosis": final_diag,
            "agent_outputs": agent_outputs,
            "tool_results": tool_results,
            "n_tool_calls": n_tool_calls,
            "optimal_path_length": ground_truth.get("optimal_path_length", 3),
        })

    # Compute metrics
    metrics = compute_all_metrics(episodes)

    from collections import defaultdict
    by_type = defaultdict(list)
    for ep in episodes:
        by_type[ep.get("scenario_type", "unknown")].append(ep)
    type_metrics = {}
    for stype, type_eps in by_type.items():
        type_metrics[stype] = compute_all_metrics(type_eps)

    model.train()  # Restore training mode

    # Save full episode details for post-hoc analysis
    if output_dir:
        ensure_dir(output_dir)
        episodes_path = os.path.join(
            output_dir, f"diag_episodes_step{step}.jsonl"
        )
        with open(episodes_path, "w", encoding="utf-8") as f:
            for ep in episodes:
                # Save compact version: agent outputs, tool results, diagnosis
                record = {
                    "scenario_id": ep["scenario_id"],
                    "scenario_type": ep["scenario_type"],
                    "difficulty": ep["difficulty"],
                    "ground_truth": ep["ground_truth"],
                    "final_diagnosis": ep["final_diagnosis"],
                    "n_tool_calls": ep["n_tool_calls"],
                    "agent_outputs": ep["agent_outputs"],
                    "tool_results": ep["tool_results"],
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info(f"  Episode details saved to {episodes_path}")

    return {
        "overall": metrics,
        "by_type": type_metrics,
        "n_episodes": len(episodes),
        "sampling_strategy": sampling_strategy,
        "sampling_seed": sampling_seed + int(step or 0),
    }


def _parse_final_diagnosis(agent_outputs: List[str]) -> Optional[Dict]:
    """Extract the final diagnosis from agent outputs."""
    for output in reversed(agent_outputs):
        diag_match = re.search(
            r"<diagnosis>\s*(\{.*?\})\s*</diagnosis>", output, re.DOTALL
        )
        if diag_match:
            try:
                return json.loads(diag_match.group(1))
            except json.JSONDecodeError:
                pass
    return None


def _create_mock_episodes(
    scenarios: List[Dict],
    model_name: str,
    tool_executor=None,
) -> List[Dict]:
    """
    Create mock episodes with real Oracle tool results for testing without GPU.

    Unlike the old version which returned dummy tool results, this uses the
    actual PredictionToolExecutor when available.
    """
    import random
    rng = random.Random(42)

    episodes = []
    for i, scenario in enumerate(scenarios):
        gt = scenario.get("ground_truth", {})
        meta = scenario.get("metadata", {})
        root_system = gt.get("root_cause_system", "")
        root_node = gt.get("root_cause_node", "")

        # Mock performance varies by model type
        if model_name == "base":
            correct_prob = 0.05
        elif model_name == "sft":
            correct_prob = 0.65
        else:  # rl
            correct_prob = 0.80

        is_correct = rng.random() < correct_prob

        # Simulate a diagnostic trajectory with real tool calls
        mock_outputs = []
        mock_tool_results = []
        n_tools = rng.randint(3, 8)

        # Step 1: get_system_overview
        mock_outputs.append(
            '<think>Starting diagnosis. Let me get a system overview.</think>\n'
            '<tool_call>{"name": "get_system_overview", "arguments": {}}</tool_call>'
        )
        if tool_executor:
            result = _execute_tool_call(tool_executor, "get_system_overview", {})
            mock_tool_results.append(result)
        else:
            mock_tool_results.append('{"systems": ["chiller_plant", "sdahu"]}')

        # Step 2: get_node_children
        if root_system:
            mock_outputs.append(
                f'<think>Investigating {root_system}.</think>\n'
                f'<tool_call>{{"name": "get_node_children", '
                f'"arguments": {{"node_id": "{root_system}"}}}}</tool_call>'
            )
            if tool_executor:
                result = _execute_tool_call(
                    tool_executor, "get_node_children",
                    {"node_id": root_system}
                )
                mock_tool_results.append(result)
            else:
                mock_tool_results.append('{"children": []}')

        # Simulate additional diagnostic steps
        for _ in range(min(n_tools - 2, 4)):
            mock_outputs.append(
                '<think>Checking next component.</think>\n'
                '<tool_call>{"name": "diagnose_node", '
                '"arguments": {"node_id": "' + root_node + '"}}</tool_call>'
            )
            if tool_executor and root_node:
                result = _execute_tool_call(
                    tool_executor, "diagnose_node",
                    {"node_id": root_node}
                )
                mock_tool_results.append(result)
            else:
                mock_tool_results.append(
                    '{"status": "Normal", "confidence": 0.9}'
                )

        # Final diagnosis
        if is_correct:
            final_diag = {
                "root_cause_node": gt.get("root_cause_node", "unknown"),
                "fault_type": gt.get("fault_type", "unknown"),
                "confidence": round(rng.uniform(0.7, 0.95), 2),
                "status": "Normal" if gt.get("fault_type") == "Normal" else "Fault",
            }
        else:
            final_diag = {
                "root_cause_node": "wrong_node",
                "fault_type": "wrong_fault",
                "confidence": round(rng.uniform(0.3, 0.6), 2),
            }

        mock_outputs.append(
            f'<think>Based on analysis...</think>\n'
            f'<diagnosis>{json.dumps(final_diag)}</diagnosis>'
        )

        episodes.append({
            "scenario_id": scenario.get("id", f"scenario_{i}"),
            "scenario_type": meta.get("scenario_type", "unknown"),
            "difficulty": meta.get("difficulty", "medium"),
            "ground_truth": gt,
            "final_diagnosis": final_diag,
            "agent_outputs": mock_outputs,
            "tool_results": mock_tool_results,
            "n_tool_calls": len(mock_tool_results),
            "optimal_path_length": gt.get("optimal_path_length", 3),
        })

    return episodes


# ============================================================================
# Model Comparison
# ============================================================================

def compare_models(
    eval_results: List[Dict[str, Any]],
    output_path: str,
) -> Dict[str, Any]:
    """
    Generate a comparative analysis of multiple model evaluations.

    Args:
        eval_results: List of evaluation result dicts from run_model_evaluation.
        output_path: Path to save the comparison report.

    Returns:
        Comparison dict with per-metric improvements.
    """
    comparison = {
        "models": [],
        "metric_comparison": {},
        "improvements": {},
    }

    metric_names = [
        "diagnostic_accuracy", "tool_format_validity",
        "diagnostic_completeness", "search_efficiency",
        "reasoning_authenticity", "tool_invocation_rationality",
        "aggregate_score",
    ]

    for result in eval_results:
        model_info = {
            "name": result["model_name"],
            "n_episodes": result["n_episodes"],
            "metrics": result["overall_metrics"],
        }
        comparison["models"].append(model_info)

    # Compute improvements between consecutive models
    for i in range(1, len(eval_results)):
        prev = eval_results[i - 1]["overall_metrics"]
        curr = eval_results[i]["overall_metrics"]
        prev_name = eval_results[i - 1]["model_name"]
        curr_name = eval_results[i]["model_name"]

        improvements = {}
        for metric in metric_names:
            prev_val = prev.get(metric, 0)
            curr_val = curr.get(metric, 0)
            if prev_val > 0:
                pct_improvement = ((curr_val - prev_val) / prev_val) * 100
            else:
                pct_improvement = float("inf") if curr_val > 0 else 0
            improvements[metric] = {
                "previous": round(prev_val, 4),
                "current": round(curr_val, 4),
                "absolute_change": round(curr_val - prev_val, 4),
                "pct_change": round(pct_improvement, 1),
            }

        comparison["improvements"][f"{prev_name}_to_{curr_name}"] = improvements

    save_json(comparison, output_path)
    logger.info(f"Model comparison saved to {output_path}")

    return comparison
