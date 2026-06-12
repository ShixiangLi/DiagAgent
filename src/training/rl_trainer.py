"""
RL Trainer — GRPO-based Reinforcement Learning for diagnostic agent.

Implements Group Relative Policy Optimization (GRPO) with:
  - Multi-turn Oracle rollout: agent generates, detects tool calls,
    executes via Oracle, feeds back results, repeats until diagnosis
  - Reference model KL divergence penalty to prevent policy drift
  - Step-based Oracle evaluation with 6 diagnostic metrics
  - Episode detail logging for post-hoc analysis
"""

import json
import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional
import pandas as pd

from src.environment.scenario_matching import match_scenario
from src.environment.fault_scenario import required_nrows_for_scenario
from src.data_gen.sft_formatter import format_system_prompt, system_prompt_sha256
from src.training.hf_loading import cached_from_pretrained, resolve_device_map
from src.training.reward_functions import compute_total_reward
from src.training.epo import compute_epo_step_rewards
from src.utils.io_utils import load_yaml, save_json, ensure_dir, setup_logger

logger = setup_logger(__name__)


def _refresh_system_prompt_entries(
    entries: List[Dict[str, Any]],
    *,
    enabled: bool = True,
) -> int:
    """Refresh persisted RL prompts in memory after prompt-level repairs."""
    if not enabled:
        return 0
    prompt = format_system_prompt()
    sha = system_prompt_sha256()
    refreshed = 0
    for entry in entries:
        messages = entry.get("messages")
        if not isinstance(messages, list) or not messages:
            continue
        if messages[0].get("role") != "system":
            continue
        if messages[0].get("content") == prompt:
            continue
        messages[0]["content"] = prompt
        metadata = entry.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["system_prompt_sha256"] = sha
            metadata["system_prompt_refreshed_for_training"] = True
        refreshed += 1
    return refreshed


# ============================================================================
# Configuration Defaults
# ============================================================================

DEFAULT_RL_CONFIG = {
    # Model
    "model_name_or_path": "outputs/sft/best",   # SFT best checkpoint
    "ref_model_name": "outputs/sft/best",        # Reference for KL penalty
    "output_dir": "outputs/rl",

    # GRPO parameters
    "group_size": 4,                # G: number of rollouts per prompt
    "kl_coeff": 0.05,              # KL divergence penalty coefficient
    "clip_range": 0.2,             # PPO-style clipping
    "max_rollout_steps": 15,       # Max tool calls per episode
    "temperature": 0.8,            # Sampling temperature for rollouts
    "rollout_do_sample": True,
    "rollout_top_p": 0.9,
    "model_device_map": "single",

    # Training
    "num_train_steps": 500,
    "per_device_batch_size": 2,    # Prompts per batch (×group_size = rollouts)
    "learning_rate": 5e-6,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "gradient_accumulation_steps": 4,
    "bf16": True,
    "max_new_tokens": 1024,        # Max tokens per generation step
    "attn_implementation": "flash_attention_2",  # falls back if unavailable
    "stop_on_tool_tags": True,
    "rollout_generate_timeout_seconds": 90,
    "rollout_max_input_tokens": None,
    "log_rollout_stages": False,
    "save_train_rollouts": True,
    "rollout_warmup_new_tokens": 8,
    "clear_cuda_cache_each_step": False,
    "gradient_checkpointing_for_loss": True,
    "max_loss_context_tokens": 4096,
    "max_kl_context_tokens": 1024,
    "log_prob_mode": "per_turn",
    "backward_per_trajectory": True,

    # EPO step-level optimization (eq:epo_reward + eq:epo_pg_loss).
    # When True, the trainer uses the paper-faithful per-step evidence-potential
    # reward with step-level (group+leave-one-out) advantages instead of a single
    # trajectory-scalar reward. The composite reward in reward_functions still
    # provides auxiliary diagnostics, but the policy-gradient signal is the
    # step-level EPO return.
    "use_step_level_epo": True,
    "epo_discount": 0.95,

    # Evaluation
    "eval_steps": 50,
    "run_diagnostic_eval": False,
    "diag_eval_episodes": 64,
    "diag_eval_csv_nrows": 10000,
    "diag_eval_episode_timeout_seconds": 180,
    "diag_eval_sampling": "stratified",
    "diag_eval_samples_per_type": None,
    "diag_eval_seed": 42,
    "diag_eval_rotate_samples": False,
    "save_steps": 50,
    "best_model_metric": "selection_score",
    "early_stop_metric": "balanced_diagnostic_accuracy",
    "oracle_mode": "real",
    "include_system_health": False,
    "expose_status_summary": False,

    # Reward weights
    "reward_weights": {
        "accuracy": 0.65,
        "efficiency": 0.08,
        "format": 0.04,
        "reasoning": 0.04,
        "completeness": 0.05,
        "topology": 0.06,
        "consistency": 0.05,
        "evidence": 0.05,
        "sensor_verification": 0.08,
        "cross_trace": 0.06,
        "process": 0.04,
        "epo": 0.20,
        "epo_config": {},
    },

    # Data
    "train_data_path": "outputs/data/rl_train.jsonl",
    "eval_data_path": "outputs/data/rl_val.jsonl",
    "resume_from_checkpoint": None,
    "no_fault_replay": {
        "types": ["na_no_fault", "no_fault"],
        "phase1_ratio": 0.05,
        "phase2_ratio": 0.15,
        "phase3_ratio": 0.10,
    },
    "evidence_replay": {
        "types": ["a_low_confidence", "na_low_confidence"],
        "phase1_ratio": 0.20,
        "phase2_ratio": 0.15,
        "phase3_ratio": 0.10,
    },
}


def load_rl_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load RL config, merging with defaults."""
    config = DEFAULT_RL_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        user_config = load_yaml(config_path)
        config.update(user_config)
    return config


# ============================================================================
# Generation helpers
# ============================================================================

def _build_stop_criteria(tokenizer, enabled: bool):
    """Build lightweight token-sequence stopping for tool/diagnosis tags."""
    if not enabled:
        return None
    try:
        from transformers import StoppingCriteria, StoppingCriteriaList
    except Exception:
        return None

    stop_sequences = []
    for text in ("</tool_call>", "</diagnosis>"):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if ids:
            stop_sequences.append(ids)
    if not stop_sequences:
        return None

    class StopOnTokenSequences(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            row = input_ids[0].tolist()
            for seq in stop_sequences:
                if len(row) >= len(seq) and row[-len(seq):] == seq:
                    return True
            return False

    return StoppingCriteriaList([StopOnTokenSequences()])


def _truncate_to_first_action(response: str) -> str:
    """Keep only the first completed executable block in a rollout turn."""
    text = str(response or "")
    endings = []
    for tag in ("</diagnosis>", "</tool_call>"):
        pos = text.find(tag)
        if pos != -1:
            endings.append((pos + len(tag), pos))
    if not endings:
        return text
    end, _ = min(endings, key=lambda item: item[1])
    return text[:end]


def _json_or_empty(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _result_status(result: Dict[str, Any]) -> str:
    return str(result.get("status") or "").strip().lower()


def _result_node(result: Dict[str, Any]) -> str:
    return str(
        result.get("node_id")
        or result.get("component_node")
        or result.get("target_node")
        or ""
    ).strip()


def _result_confidence(result: Dict[str, Any]) -> float:
    try:
        return float(result.get("confidence") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _result_has_sensor_readings(result: Dict[str, Any]) -> bool:
    if result.get("readings_available") is True:
        return True
    readings = result.get("sensor_readings")
    if isinstance(readings, dict) and readings:
        return True
    sensors = result.get("sensors")
    return isinstance(sensors, list) and any(
        isinstance(sensor, dict) and "current_value" in sensor
        for sensor in sensors
    )


def _closed_rollout_final_prompt(tool_results: List[Any]) -> Optional[str]:
    """Allow a terminal diagnosis turn after the tool-call cap is reached."""
    parsed = [_json_or_empty(raw) for raw in tool_results]
    parsed = [item for item in parsed if item]
    if not parsed:
        return None

    for result in reversed(parsed):
        if _result_status(result) != "fault":
            continue
        node = _result_node(result)
        fault = str(result.get("fault_type") or "").strip()
        confidence = _result_confidence(result)
        if not node:
            continue
        return (
            "The tool-call budget is exhausted, but visible evidence is closed: "
            f"diagnose_node returned same-node Fault for node_id={node}, "
            f"fault_type={fault}, confidence={confidence:.3f}. Output exactly "
            "one final <diagnosis>...</diagnosis> block using that node and "
            "fault type. Do not call another tool."
        )

    latest_warning = None
    for result in reversed(parsed):
        status = _result_status(result)
        fault_type = str(result.get("fault_type") or "").strip().lower()
        if status in {"warning", "indeterminate"} or (
            fault_type == "uncertain"
            and status not in {"normal", "unknown", "error", "data_unavailable"}
        ):
            latest_warning = result
            break
    if latest_warning:
        node = _result_node(latest_warning)
        verified_nodes = {
            _result_node(result)
            for result in parsed
            if _result_has_sensor_readings(result) and _result_node(result)
        }
        if node and node in verified_nodes:
            fault = str(latest_warning.get("fault_type") or "uncertain").strip()
            confidence = _result_confidence(latest_warning)
            return (
                "The tool-call budget is exhausted and the latest low-confidence "
                "candidate has same-node sensor evidence. Output exactly one "
                f"calibrated <diagnosis>...</diagnosis> block for node_id={node}, "
                f"fault_type={fault}, confidence around {confidence:.3f}. "
                "Do not call another tool."
            )

    normal_count = sum(
        1
        for result in parsed
        if _result_status(result) == "normal"
        and _result_node(result)
        and _result_confidence(result) >= 0.85
    )
    has_abnormal = any(
        _result_status(result) in {"fault", "warning", "abnormal"}
        for result in parsed
    )
    if normal_count >= 2 and not has_abnormal:
        return (
            "The tool-call budget is exhausted and no-fault evidence is closed: "
            "at least two relevant diagnose_node observations returned "
            "high-confidence Normal and no contradictory Fault/Warning/Abnormal "
            "evidence is visible. Output exactly one final no-fault "
            '<diagnosis> block with root_cause_node="none", '
            'fault_type="no_fault", status="Normal". Do not call another tool.'
        )

    return None


def _model_input_device(model):
    """Return the device that should receive input ids for sharded models."""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        pass
    try:
        return model.device
    except Exception:
        return next(model.parameters()).device


def _set_model_use_cache(model, enabled: bool) -> None:
    """Best-effort toggle for generation KV cache on PEFT/base models."""
    for obj in (
        model,
        getattr(model, "base_model", None),
        getattr(getattr(model, "base_model", None), "model", None),
    ):
        cfg = getattr(obj, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = bool(enabled)


def _save_jsonl_record(path: str, record: Dict[str, Any]) -> None:
    """Append one JSONL record and flush it for live training inspection."""
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()


def _save_policy_checkpoint(
    model,
    tokenizer,
    ckpt_dir: str,
    *,
    is_peft_model: bool,
    policy_adapter_name: Optional[str],
) -> None:
    """Save only the trainable policy adapter in evaluator-loadable layout."""
    ensure_dir(ckpt_dir)
    if is_peft_model and policy_adapter_name:
        model.set_adapter(policy_adapter_name)
        try:
            model.save_pretrained(
                ckpt_dir,
                selected_adapters=[policy_adapter_name],
            )
        except TypeError:
            model.save_pretrained(ckpt_dir)

        # Non-default PEFT adapter names are saved under a subdirectory.  The
        # evaluator expects adapter_config.json at the checkpoint root, so keep
        # only the active policy adapter and flatten it after saving.
        nested_dir = os.path.join(ckpt_dir, policy_adapter_name)
        nested_cfg = os.path.join(nested_dir, "adapter_config.json")
        if os.path.exists(nested_cfg):
            for name in os.listdir(nested_dir):
                src = os.path.join(nested_dir, name)
                dst = os.path.join(ckpt_dir, name)
                if os.path.isdir(src):
                    if os.path.exists(dst):
                        shutil.rmtree(dst)
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)
            shutil.rmtree(nested_dir)

        # A frozen reference adapter is useful in memory but should not be part
        # of deployable RL checkpoints.
        ref_dir = os.path.join(ckpt_dir, "reference")
        if os.path.exists(ref_dir):
            shutil.rmtree(ref_dir)
    else:
        model.save_pretrained(ckpt_dir)
    tokenizer.save_pretrained(ckpt_dir)


def _warmup_generation(
    model,
    tokenizer,
    messages: List[Dict[str, Any]],
    max_new_tokens: int = 8,
) -> None:
    """Run a tiny generation once so the first real rollout is not a warmup."""
    if max_new_tokens <= 0:
        return
    import torch

    try:
        model.gradient_checkpointing_disable()
    except Exception:
        pass
    _set_model_use_cache(model, True)
    model.eval()
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(_model_input_device(model))
    start = time.monotonic()
    with torch.no_grad():
        model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        )
    logger.info(
        "RL generation warmup complete: prompt_tokens=%s, new_tokens=%s, "
        "elapsed=%.1fs",
        inputs.input_ids.shape[1],
        max_new_tokens,
        time.monotonic() - start,
    )


# ============================================================================
# Multi-turn Oracle Rollout
# ============================================================================

def _run_multi_turn_rollout(
    model,
    tokenizer,
    messages: List[Dict],
    tool_executor,
    max_steps: int = 15,
    temperature: float = 0.8,
    max_new_tokens: int = 1024,
    stop_on_tool_tags: bool = True,
    do_sample: bool = True,
    top_p: Optional[float] = 0.9,
    generate_timeout_seconds: Optional[float] = None,
    max_input_tokens: Optional[int] = None,
    log_stages: bool = False,
    rollout_label: str = "",
) -> Dict[str, Any]:
    """
    Execute a multi-turn rollout with real Oracle environment.

    The agent generates responses, detects tool calls, executes them
    via Oracle, feeds back results, and repeats until diagnosis or max_steps.

    Args:
        model: The policy model.
        tokenizer: The tokenizer.
        messages: Initial conversation messages [system, user].
        tool_executor: Oracle environment executor.
        max_steps: Maximum tool calls.
        temperature: Sampling temperature.
        max_new_tokens: Max tokens per generation step.

    Returns:
        Dict with agent_outputs, tool_results, n_tool_calls,
        full_token_ids (for log-prob computation), and final_diagnosis.
    """
    import torch

    agent_outputs = []
    tool_results = []
    n_tool_calls = 0
    all_completion_ids = []  # Track generated token IDs for log-prob
    invalid_no_action = False
    timeout_hit = False
    prompt_token_lengths: List[int] = []
    generation_durations: List[float] = []
    no_action_retries = 0
    max_no_action_retries = 2  # recover truncated/no-action turns before giving up

    conversation = list(messages)
    stopping_criteria = _build_stop_criteria(tokenizer, stop_on_tool_tags)
    input_device = _model_input_device(model)

    turn_index = 0
    final_prompt_after_cap_sent = False
    max_turns = max_steps + 2
    while turn_index < max_turns:
        if n_tool_calls >= max_steps:
            final_prompt = _closed_rollout_final_prompt(tool_results)
            if not final_prompt or final_prompt_after_cap_sent:
                break
            conversation.append({"role": "user", "content": final_prompt})
            final_prompt_after_cap_sent = True
        turn_index += 1
        turn_start = time.monotonic()
        prompt = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=max_input_tokens is not None,
            max_length=max_input_tokens,
        ).to(input_device)
        prompt_len = inputs.input_ids.shape[1]
        prompt_token_lengths.append(int(prompt_len))
        if log_stages:
            logger.info(
                "  Rollout%s turn %s/%s: prompt_tokens=%s, "
                "max_new_tokens=%s, timeout=%s",
                f" {rollout_label}" if rollout_label else "",
                turn_index,
                max_turns,
                prompt_len,
                max_new_tokens,
                generate_timeout_seconds,
            )

        with torch.no_grad():
            generate_kwargs = dict(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            if do_sample:
                generate_kwargs["temperature"] = temperature
                if top_p is not None:
                    generate_kwargs["top_p"] = top_p
            if generate_timeout_seconds and generate_timeout_seconds > 0:
                generate_kwargs["max_time"] = float(generate_timeout_seconds)
            if stopping_criteria is not None:
                generate_kwargs["stopping_criteria"] = stopping_criteria
            output = model.generate(**generate_kwargs)
        generation_elapsed = time.monotonic() - turn_start
        generation_durations.append(float(generation_elapsed))

        completion_ids = output[0][prompt_len:]
        response = tokenizer.decode(completion_ids, skip_special_tokens=True)
        if log_stages:
            logger.info(
                "  Rollout%s turn %s/%s: generated_tokens=%s, elapsed=%.1fs, "
                "complete_tool=%s, complete_diag=%s",
                f" {rollout_label}" if rollout_label else "",
                turn_index,
                max_turns,
                int(len(completion_ids)),
                generation_elapsed,
                int("</tool_call>" in response),
                int("</diagnosis>" in response),
            )
        if (
            generate_timeout_seconds
            and generate_timeout_seconds > 0
            and generation_elapsed >= float(generate_timeout_seconds) * 0.95
            and "</tool_call>" not in response
            and "</diagnosis>" not in response
        ):
            timeout_hit = True

        truncated_response = _truncate_to_first_action(response)
        if truncated_response != response:
            response = truncated_response
            completion_ids = tokenizer.encode(
                response,
                add_special_tokens=False,
                return_tensors="pt",
            ).to(completion_ids.device)[0]

        agent_outputs.append(response)
        all_completion_ids.append(completion_ids)
        conversation.append({"role": "assistant", "content": response})

        diag_match = re.search(
            r"<diagnosis>\s*(\{.*?\})\s*</diagnosis>", response, re.DOTALL
        )
        if diag_match:
            break

        # Check for tool calls
        tc_match = re.search(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>", response, re.DOTALL
        )

        if tc_match:
            if n_tool_calls >= max_steps:
                invalid_no_action = True
                conversation.append({
                    "role": "user",
                    "content": (
                        "The tool-call budget is exhausted. Do not call "
                        "another tool; output exactly one final "
                        "<diagnosis>...</diagnosis> block from the visible "
                        "evidence."
                    ),
                })
                continue
            n_tool_calls += 1
            try:
                tc = json.loads(tc_match.group(1))
                tool_name = tc.get("name", "")
                tool_args = tc.get("arguments", {})
                if tool_executor is not None:
                    result = tool_executor.execute(tool_name, tool_args)
                    tool_result = json.dumps(result, ensure_ascii=False)
                else:
                    tool_result = json.dumps({
                        "status": "error",
                        "message": "Oracle environment not available",
                    })
            except (json.JSONDecodeError, Exception) as e:
                tool_result = json.dumps({
                    "status": "error",
                    "message": str(e),
                })

            tool_results.append(tool_result)
            conversation.append({"role": "tool", "content": tool_result})
        else:
            # Check for final diagnosis
            if "<diagnosis>" in response or "final diagnosis" in response.lower():
                break
            # No executable action this turn (e.g. the generation was truncated
            # before reaching <tool_call>/<diagnosis>, or rambled). Instead of
            # immediately ending the episode (which collapses the trajectory to
            # no_diag and kills the reward signal), inject a corrective prompt
            # and retry a bounded number of times.
            if no_action_retries < max_no_action_retries:
                no_action_retries += 1
                conversation.append({
                    "role": "user",
                    "content": (
                        "Your previous response did not contain a complete "
                        "executable block. Respond now with EXACTLY ONE of: a "
                        "<tool_call>{...}</tool_call> to gather more evidence, or "
                        "a final <diagnosis>{...}</diagnosis> if the root cause "
                        "is already supported by the visible evidence. Output the "
                        "complete block and nothing after it."
                    ),
                })
                continue
            invalid_no_action = True
            break

    # Parse final diagnosis
    final_diag = _parse_diagnosis_from_text(
        "\n".join(agent_outputs)
    )

    return {
        "agent_outputs": agent_outputs,
        "tool_results": tool_results,
        "n_tool_calls": n_tool_calls,
        "final_diagnosis": final_diag,
        "conversation": conversation,
        "all_completion_ids": all_completion_ids,
        "hit_max_steps": n_tool_calls >= max_steps and final_diag is None,
        "invalid_no_action": invalid_no_action,
        "generation_timeout_hit": timeout_hit,
        "prompt_token_lengths": prompt_token_lengths,
        "generation_durations": generation_durations,
    }


def _compute_trajectory_log_prob(
    model,
    tokenizer,
    conversation: List[Dict],
    all_completion_ids: List,
    max_context_tokens: int = 4096,
) -> "torch.Tensor":
    """
    Compute the log probability of the full trajectory under the current policy.

    Re-encodes the full conversation and computes log-probs only on
    assistant-generated tokens. This is called WITH gradient enabled.

    Returns:
        Scalar tensor of mean log probability (with gradient attached).
    """
    import torch
    import torch.nn.functional as F

    # Encode the full conversation
    full_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer.encode(
        full_text, return_tensors="pt",
    ).to(_model_input_device(model))

    # Truncate to avoid OOM on very long multi-turn conversations. Keeping the
    # tail preserves the final diagnostic decision and recent tool evidence.
    max_len = max(512, int(max_context_tokens or 4096))
    if full_ids.shape[1] > max_len:
        full_ids = full_ids[:, -max_len:]

    # Forward pass WITH gradient
    logits = model(full_ids).logits

    # Compute log-probs for all completion tokens
    log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)
    target_ids = full_ids[0, 1:]
    token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)

    # Only count assistant tokens (approximate: use completion_ids lengths)
    # For efficiency, compute mean over the last portion of tokens
    total_completion_len = sum(len(ids) for ids in all_completion_ids)
    if total_completion_len > 0 and total_completion_len <= len(token_log_probs):
        assistant_log_probs = token_log_probs[-total_completion_len:]
    else:
        assistant_log_probs = token_log_probs

    return assistant_log_probs.sum()


def _compute_per_turn_log_prob(
    model,
    tokenizer,
    conversation: List[Dict],
    max_context_tokens: int = 4096,
) -> "torch.Tensor":
    """Compute mean log-prob over assistant turns with local context windows.

    The previous full-trajectory loss kept only the tail when transcripts were
    long, which can drop early topology choices. This version scores each
    assistant message using the conversation prefix available before that turn,
    so early tool decisions still receive gradient while memory stays bounded.
    """
    import torch
    import torch.nn.functional as F

    max_len = max(512, int(max_context_tokens or 4096))
    turn_log_probs = []
    prefix: List[Dict] = []

    for message in conversation:
        role = message.get("role")
        if role != "assistant":
            prefix.append(message)
            continue

        content = message.get("content", "")
        if not content:
            prefix.append(message)
            continue

        prompt_text = tokenizer.apply_chat_template(
            prefix, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        )
        completion_ids = tokenizer.encode(
            content, add_special_tokens=False, return_tensors="pt"
        )
        if completion_ids.shape[1] == 0:
            prefix.append(message)
            continue

        full_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(
            _model_input_device(model)
        )
        completion_len = completion_ids.shape[1]
        needed_len = completion_len + 1

        # Keep every target completion token. If a single assistant message is
        # too long, keep the tail of that message; otherwise trim only prompt
        # context from the left.
        if full_ids.shape[1] > max_len:
            keep_len = max(needed_len, max_len)
            full_ids = full_ids[:, -keep_len:]
        if full_ids.shape[1] < 2:
            prefix.append(message)
            continue

        logits = model(full_ids).logits
        log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)
        target_ids = full_ids[0, 1:]
        token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        usable_completion_len = min(completion_len, len(token_log_probs))
        if usable_completion_len > 0:
            turn_log_probs.append(token_log_probs[-usable_completion_len:].sum())

        prefix.append(message)

    if turn_log_probs:
        return torch.stack(turn_log_probs).sum()

    # Fallback should be rare, but keeps the trainer robust to malformed
    # conversations.
    return _compute_trajectory_log_prob(
        model, tokenizer, conversation, [], max_context_tokens=max_context_tokens
    )


def _compute_per_turn_log_prob_list(
    model,
    tokenizer,
    conversation: List[Dict],
    max_context_tokens: int = 4096,
) -> "List[torch.Tensor]":
    """Per-assistant-turn summed log-probs as a LIST (for step-level EPO).

    Mirrors :func:`_compute_per_turn_log_prob` but returns one scalar tensor per
    assistant turn instead of their sum, so the EPO trainer can weight each turn
    by its own step advantage A_t. Turn order matches the order of assistant
    messages in ``conversation`` (which equals the tool-call/diagnosis step
    order in a rollout).
    """
    import torch
    import torch.nn.functional as F

    max_len = max(512, int(max_context_tokens or 4096))
    turn_log_probs: List["torch.Tensor"] = []
    prefix: List[Dict] = []

    for message in conversation:
        role = message.get("role")
        if role != "assistant":
            prefix.append(message)
            continue
        content = message.get("content", "")
        if not content:
            prefix.append(message)
            continue

        prompt_text = tokenizer.apply_chat_template(
            prefix, tokenize=False, add_generation_prompt=True
        )
        prompt_ids = tokenizer.encode(
            prompt_text, add_special_tokens=False, return_tensors="pt"
        )
        completion_ids = tokenizer.encode(
            content, add_special_tokens=False, return_tensors="pt"
        )
        if completion_ids.shape[1] == 0:
            prefix.append(message)
            continue

        full_ids = torch.cat([prompt_ids, completion_ids], dim=1).to(
            _model_input_device(model)
        )
        completion_len = completion_ids.shape[1]
        needed_len = completion_len + 1
        if full_ids.shape[1] > max_len:
            keep_len = max(needed_len, max_len)
            full_ids = full_ids[:, -keep_len:]
        if full_ids.shape[1] < 2:
            prefix.append(message)
            continue

        logits = model(full_ids).logits
        log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)
        target_ids = full_ids[0, 1:]
        token_log_probs = log_probs.gather(1, target_ids.unsqueeze(1)).squeeze(1)
        usable = min(completion_len, len(token_log_probs))
        if usable > 0:
            turn_log_probs.append(token_log_probs[-usable:].sum())
        prefix.append(message)

    return turn_log_probs


def _epo_step_returns(step_rewards: List[float], terminal_reward: float, discount: float) -> List[float]:
    """Discounted return-to-go G_t for an EPO step-reward sequence.

    The terminal reward is attached to the final step, then returns are computed
    backwards: G_t = r_t + ξ·G_{t+1}. This is the per-step target the policy
    gradient maximizes (eq:epo_reward / eq:epo_pg_loss).
    """
    rewards = list(step_rewards)
    if rewards:
        rewards[-1] += float(terminal_reward)
    else:
        rewards = [float(terminal_reward)]
    returns: List[float] = [0.0] * len(rewards)
    running = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        running = rewards[i] + discount * running
        returns[i] = running
    return returns


def _forward_logits(model, input_ids, logits_to_keep: Optional[int] = None):
    """Forward helper that uses logits slicing when supported by transformers."""
    if logits_to_keep:
        try:
            return model(input_ids, logits_to_keep=logits_to_keep).logits
        except TypeError:
            pass
    return model(input_ids).logits


def _compute_kl_divergence(
    model,
    ref_model,
    tokenizer,
    conversation: List[Dict],
    is_peft: bool = False,
    policy_adapter_name: Optional[str] = None,
    ref_adapter_name: Optional[str] = None,
    max_context_tokens: int = 1024,
) -> "torch.Tensor":
    """
    Compute KL(π_θ || π_ref) for the conversation.

    For PEFT policy models, prefer a frozen reference adapter loaded into the
    same base model. Falling back to disabled adapters is allowed only when a
    real SFT reference adapter is unavailable.

    Returns scalar KL divergence tensor.
    """
    import torch
    import torch.nn.functional as F

    full_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer.encode(full_text, return_tensors="pt").to(
        _model_input_device(model)
    )

    # KL only needs a recent slice of the trajectory to constrain style and
    # tool-use drift; using the whole tool transcript is memory-heavy.
    max_len = max(256, int(max_context_tokens or 1024))
    if full_ids.shape[1] > max_len:
        full_ids = full_ids[:, -max_len:]

    n_tokens = min(256, full_ids.shape[1])

    if ref_model is None and is_peft and ref_adapter_name:
        with torch.no_grad():
            model.set_adapter(ref_adapter_name)
            ref_logits = _forward_logits(model, full_ids, logits_to_keep=n_tokens)
            if policy_adapter_name:
                model.set_adapter(policy_adapter_name)
    elif ref_model is None and is_peft:
        # Fallback only: compare policy adapter to the base model.
        with torch.no_grad():
            model.disable_adapter_layers()
            ref_logits = _forward_logits(model, full_ids, logits_to_keep=n_tokens)
            model.enable_adapter_layers()
            if policy_adapter_name:
                model.set_adapter(policy_adapter_name)
    elif ref_model is not None:
        with torch.no_grad():
            ref_logits = _forward_logits(ref_model, full_ids, logits_to_keep=n_tokens)
    else:
        # No ref model available, return zero KL
        return torch.tensor(0.0, device=model.device)

    if policy_adapter_name and is_peft:
        model.set_adapter(policy_adapter_name)
    policy_logits = _forward_logits(model, full_ids, logits_to_keep=n_tokens)

    # KL divergence on last 256 tokens (where assistant content is)
    n_tokens = min(256, policy_logits.shape[1], ref_logits.shape[1])
    p_log = F.log_softmax(policy_logits[0, -n_tokens:, :], dim=-1)
    q_log = F.log_softmax(ref_logits[0, -n_tokens:, :], dim=-1)

    kl = F.kl_div(q_log, p_log, log_target=True, reduction="batchmean")
    return kl


# ============================================================================
# GRPO Training Loop
# ============================================================================

def _load_causal_lm_with_attn(
    cls,
    model_name_or_path,
    *,
    logger,
    description,
    attn_implementation,
    **kwargs,
):
    """Load a CausalLM, preferring an explicit attn implementation with fallback.

    Flash-Attention-2 cuts memory and speeds up the long multi-turn rollout
    contexts, but it is not always installed/compatible. When requested, try it
    first and fall back to the default attention on any error so RL training
    never hard-fails on an environment without flash-attn.
    """
    if attn_implementation:
        try:
            logger.info(
                "Loading %s with attn_implementation=%s",
                description, attn_implementation,
            )
            return cached_from_pretrained(
                cls,
                model_name_or_path,
                logger=logger,
                description=description,
                attn_implementation=attn_implementation,
                **kwargs,
            )
        except Exception as exc:
            logger.warning(
                "Failed to load %s with attn_implementation=%s (%s); "
                "falling back to default attention.",
                description, attn_implementation, exc,
            )
    return cached_from_pretrained(
        cls,
        model_name_or_path,
        logger=logger,
        description=description,
        **kwargs,
    )


def run_rl_training(config: Dict[str, Any]) -> str:
    """
    Run GRPO-based RL training with multi-turn Oracle rollouts.

    Training loop:
    1. Sample a batch of prompts from the training set
    2. For each prompt, generate G multi-turn rollout trajectories
    3. Execute each trajectory in the Oracle environment
    4. Compute composite rewards for each trajectory
    5. Compute group-relative advantages (reward - group_mean) / std
    6. Re-compute log-probs WITH gradient, add KL penalty
    7. Update policy using advantage-weighted loss

    Args:
        config: RL training configuration.

    Returns:
        Path to the best checkpoint.
    """
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    output_dir = ensure_dir(config["output_dir"])

    # ---- Load policy model and SFT reference ----
    policy_path = config["model_name_or_path"]
    ref_path = config.get("ref_model_name") or policy_path
    logger.info(f"Loading policy model from: {policy_path}")
    logger.info(f"KL reference checkpoint: {ref_path}")
    tokenizer = cached_from_pretrained(
        AutoTokenizer,
        policy_path,
        logger=logger,
        description="RL tokenizer",
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_peft_model = False
    policy_adapter_name = None
    ref_adapter_name = None
    ref_model = None
    resolved_device_map = resolve_device_map(
        config.get("model_device_map", "single"),
        logger=logger,
        description="RL model",
    )
    attn_impl = config.get("attn_implementation")

    adapter_config_path = os.path.join(policy_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r", encoding="utf-8") as f:
            adapter_cfg = json.load(f)
        base_model_name = adapter_cfg.get(
            "base_model_name_or_path", "Qwen/Qwen2.5-7B-Instruct"
        )
        base_model = _load_causal_lm_with_attn(
            AutoModelForCausalLM,
            base_model_name,
            logger=logger,
            description="RL base model",
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=resolved_device_map,
        )
        policy_adapter_name = "policy"
        model = PeftModel.from_pretrained(
            base_model, policy_path,
            adapter_name=policy_adapter_name,
            is_trainable=True,
        )
        is_peft_model = True
        model.set_adapter(policy_adapter_name)
        logger.info(
            f"Loaded trainable PEFT policy adapter '{policy_adapter_name}' "
            f"on base {base_model_name}"
        )

        ref_adapter_config = os.path.join(ref_path, "adapter_config.json")
        if os.path.exists(ref_adapter_config):
            ref_adapter_name = "reference"
            model.load_adapter(
                ref_path,
                adapter_name=ref_adapter_name,
                is_trainable=False,
            )
            model.set_adapter(policy_adapter_name)
            logger.info(
                f"Loaded frozen PEFT KL reference adapter '{ref_adapter_name}' "
                f"from {ref_path}"
            )
        else:
            logger.warning(
                f"Reference adapter not found at {ref_path}; KL will fall back "
                "to the base model and may allow policy drift."
            )
    else:
        model = _load_causal_lm_with_attn(
            AutoModelForCausalLM,
            policy_path,
            logger=logger,
            description="RL policy model",
            attn_implementation=attn_impl,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map=resolved_device_map,
        )
        logger.info("Loaded policy as full model")
        if ref_path and os.path.exists(ref_path):
            ref_model = _load_causal_lm_with_attn(
                AutoModelForCausalLM,
                ref_path,
                logger=logger,
                description="RL reference model",
                attn_implementation=attn_impl,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map=resolved_device_map,
            )
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
            logger.info("Loaded separate frozen full-model KL reference")
        else:
            logger.warning("No KL reference model available for full-model policy")

    # ---- Load Oracle environment ----
    from src.evaluation.evaluator import _create_tool_environment
    oracle_mode = config.get("oracle_mode", "real")
    include_system_health = bool(config.get("include_system_health", False))
    expose_status_summary = bool(config.get("expose_status_summary", False))
    builder, tool_executor, model_registry = _create_tool_environment(
        oracle_mode=oracle_mode,
        include_system_health=include_system_health,
        expose_status_summary=expose_status_summary,
    )
    if tool_executor is None:
        raise RuntimeError("Failed to create Oracle environment for RL training")
    logger.info(
        f"RL Oracle mode: {oracle_mode}, "
        f"include_system_health={include_system_health}, "
        f"expose_status_summary={expose_status_summary}"
    )

    # ---- Build scenario state loader ----
    from dataclasses import replace
    from src.environment.fault_scenario import FaultScenario, create_scenario_state
    from src.environment.diagnostic_path import DiagnosticPath, PathNode
    from src.environment.diagnostic_path import generate_no_fault_path
    from src.node_models.data_loader import discover_fault_files, read_fault_file

    scenario_lookup = {}
    all_scenarios_path = "outputs/data/all_scenarios.json"
    if os.path.exists(all_scenarios_path):
        raw_scenarios = json.load(open(all_scenarios_path, "r", encoding="utf-8"))
        for d in raw_scenarios:
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
            scenario_lookup[fs.scenario_id] = fs

    # Build CSV file path lookup
    import yaml
    topo_config = yaml.safe_load(open("configs/topology_config.yaml", "r"))
    file_info_map = {}  # (sys_id, filename) -> FaultFileInfo
    normal_file_map = {}  # sys_id -> baseline filename
    for sys_id in topo_config.get("systems", {}):
        try:
            for ff in discover_fault_files("data/lbnl", sys_id):
                file_info_map[(sys_id, ff.filename)] = ff
                if ff.is_fault_free or ff.fault_type.lower() == "normal":
                    normal_file_map.setdefault(sys_id, ff.filename)
        except Exception:
            pass

    csv_cache = {}  # (sys_id, filename) -> DataFrame
    csv_cache_requested = {}

    def _metadata_int(metadata: Dict[str, Any], key: str) -> Optional[int]:
        try:
            value = metadata.get(key)
            if value in (None, ""):
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _bind_scenario_data_reference(fs, metadata: Dict[str, Any]):
        if fs is None or not isinstance(metadata, dict):
            return fs
        updates = {}
        source_file = str(metadata.get("source_file") or "").strip()
        if source_file:
            updates["source_file"] = source_file
        for key in ("time_window_start", "time_window_end"):
            value = _metadata_int(metadata, key)
            if value is not None and value >= 0:
                updates[key] = value
        if not updates:
            return fs
        try:
            return replace(fs, **updates)
        except Exception:
            return fs

    def _scenario_match_ids(scenario_id: str, metadata: Dict[str, Any]) -> List[str]:
        ids = []
        for key in ("source_scenario_id", "scenario_id"):
            value = str((metadata or {}).get(key) or "").strip()
            if value and value not in ids:
                ids.append(value)
        sid = str(scenario_id or "").strip()
        if sid and sid not in ids:
            ids.append(sid)
        return ids

    def _load_cached_csv(system_id: str, filename: str, fs):
        key = (system_id, filename)
        required_rows = required_nrows_for_scenario(fs, minimum=50000)
        if (
            key in csv_cache
            and int(csv_cache_requested.get(key) or 0) >= int(required_rows or 0)
        ):
            return csv_cache[key]
        finfo = file_info_map.get(key)
        if not finfo:
            return None
        df = read_fault_file(
            finfo,
            nrows=required_rows,
            numeric_only=True,
        )
        csv_cache[key] = df
        csv_cache_requested[key] = int(required_rows or 0)
        return df

    def _load_rl_scenario_state(
        scenario_id: str,
        ground_truth=None,
        scenario_type: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Load scenario state for an RL prompt, including downstream data.

        Uses the same 3-strategy matching as the evaluator to prevent
        ID mismatch issues:
          1. Direct ID match
          2. Strip SFT prefixes/suffixes
          3. Match by ground_truth (root_cause_system + fault_type)
        """
        metadata = metadata or {}
        fs = None
        for candidate_sid in _scenario_match_ids(scenario_id, metadata):
            fs, _, _ = match_scenario(
                candidate_sid,
                scenario_lookup,
                ground_truth=ground_truth,
                scenario_type=scenario_type,
                allow_ground_truth_fallback=False,
            )
            if fs is not None:
                break
        if fs is None:
            fs, _, _ = match_scenario(
                scenario_id,
                scenario_lookup,
                ground_truth=ground_truth,
                scenario_type=scenario_type,
            )

        if fs is None:
            return None

        fs = _bind_scenario_data_reference(fs, metadata)
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

        # Build system_data with primary and downstream system data
        system_data = {}
        # Primary system
        key = (fs.root_cause_system, fs.source_file)
        root_df = _load_cached_csv(fs.root_cause_system, fs.source_file, fs)
        if root_df is not None:
            system_data[fs.root_cause_system] = root_df
        else:
            return None
        # Load normal files for ALL other systems in the building
        for sys_id, normal_file in normal_file_map.items():
            if sys_id not in system_data:
                alt_df = _load_cached_csv(sys_id, normal_file, fs)
                if alt_df is not None:
                    if len(alt_df) < fs.time_window_end:
                        repeats = (fs.time_window_end // len(alt_df)) + 1
                        alt_df = pd.concat([alt_df] * repeats, ignore_index=True)
                    system_data[sys_id] = alt_df
        try:
            return create_scenario_state(
                fs, system_data,
                builder, registry=model_registry,
            )
        except Exception:
            return None

    # ---- Load training prompts ----
    train_prompts = []
    with open(config["train_data_path"], "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                train_prompts.append(json.loads(line))
    logger.info(f"Loaded {len(train_prompts)} RL training prompts")
    refreshed_train_prompts = _refresh_system_prompt_entries(
        train_prompts,
        enabled=bool(config.get("refresh_system_prompt_for_training", True)),
    )
    if refreshed_train_prompts:
        logger.info(
            "Refreshed %s persisted RL system prompts in memory "
            "(current_sha=%s)",
            refreshed_train_prompts,
            system_prompt_sha256(),
        )

    # ---- Build per-type index for curriculum learning ----
    prompts_by_type = {}
    for entry in train_prompts:
        stype = entry.get("metadata", {}).get("scenario_type", "single_system")
        prompts_by_type.setdefault(stype, []).append(entry)

    # v6: Cap no_fault scenarios to prevent reward hacking shortcut
    # Without this, 34% of Phase 1 pool is no_fault → model learns "just say no_fault"
    no_fault_cap = config.get("no_fault_cap", 100)
    no_fault_types = ["na_no_fault", "no_fault"]
    for nf_type in no_fault_types:
        if nf_type in prompts_by_type and len(prompts_by_type[nf_type]) > no_fault_cap:
            original_count = len(prompts_by_type[nf_type])
            import random as _rnd
            _rnd.shuffle(prompts_by_type[nf_type])
            prompts_by_type[nf_type] = prompts_by_type[nf_type][:no_fault_cap]
            logger.info(f"  Capped {nf_type}: {original_count} → {no_fault_cap}")

    phase3_prompts = [
        prompt
        for entries in prompts_by_type.values()
        for prompt in entries
    ]
    no_fault_replay_cfg = config.get("no_fault_replay", {}) or {}
    no_fault_replay_types = no_fault_replay_cfg.get(
        "types", ["na_no_fault", "no_fault"]
    )
    no_fault_replay_pool = [
        prompt
        for t in no_fault_replay_types
        for prompt in prompts_by_type.get(t, [])
    ]
    evidence_replay_cfg = config.get("evidence_replay", {}) or {}
    evidence_replay_types = evidence_replay_cfg.get(
        "types", ["a_low_confidence", "na_low_confidence"]
    )
    evidence_replay_pool = [
        prompt
        for t in evidence_replay_types
        for prompt in prompts_by_type.get(t, [])
    ]
    cross_trace_replay_cfg = config.get("cross_trace_replay", {}) or {}
    cross_trace_replay_types = cross_trace_replay_cfg.get(
        "types", ["cross_system", "a_cross_system"]
    )
    cross_trace_replay_pool = [
        prompt
        for t in cross_trace_replay_types
        for prompt in prompts_by_type.get(t, [])
    ]

    logger.info("Scenario type distribution: " + ", ".join(
        f"{t}={len(v)}" for t, v in sorted(prompts_by_type.items())
    ))
    logger.info(
        "No-fault replay pool: "
        f"{len(no_fault_replay_pool)} prompts from {no_fault_replay_types}"
    )
    logger.info(
        "Evidence replay pool: "
        f"{len(evidence_replay_pool)} prompts from {evidence_replay_types}"
    )
    logger.info(
        "Cross-trace replay pool: "
        f"{len(cross_trace_replay_pool)} prompts from {cross_trace_replay_types}"
    )

    # Curriculum learning configuration
    curriculum = config.get("curriculum", {})
    phase1_end = curriculum.get("phase1_end", 100)
    phase2_end = curriculum.get("phase2_end", 200)
    # Default types use na_/a_ prefixed names matching actual data
    phase1_types = set(curriculum.get("phase1_types",
        ["na_single_system", "a_single_system", "na_no_fault"]))
    phase2_types = set(curriculum.get("phase2_types",
        ["na_single_system", "a_single_system", "na_no_fault",
         "a_low_confidence", "na_low_confidence"]))

    def _get_curriculum_pool(step):
        """Return the training prompt pool for the current curriculum phase."""
        if step <= phase1_end:
            pool = []
            for t in phase1_types:
                pool.extend(prompts_by_type.get(t, []))
            phase_name = "Phase 1 (easy)"
        elif step <= phase2_end:
            pool = []
            for t in phase2_types:
                pool.extend(prompts_by_type.get(t, []))
            phase_name = "Phase 2 (medium)"
        else:
            pool = phase3_prompts
            phase_name = "Phase 3 (capped all)"
        # Fallback: if pool is empty (type mismatch), use all prompts
        if not pool:
            pool = train_prompts
            phase_name += " (fallback: all)"
        return pool, phase_name

    def _phase_replay_ratio(phase_name: str) -> float:
        if "Phase 1" in phase_name:
            return float(no_fault_replay_cfg.get("phase1_ratio", 0.0))
        if "Phase 2" in phase_name:
            return float(no_fault_replay_cfg.get("phase2_ratio", 0.0))
        return float(no_fault_replay_cfg.get("phase3_ratio", 0.0))

    def _phase_evidence_ratio(phase_name: str) -> float:
        if "Phase 1" in phase_name:
            return float(evidence_replay_cfg.get("phase1_ratio", 0.0))
        if "Phase 2" in phase_name:
            return float(evidence_replay_cfg.get("phase2_ratio", 0.0))
        return float(evidence_replay_cfg.get("phase3_ratio", 0.0))

    def _phase_cross_trace_ratio(phase_name: str) -> float:
        if "Phase 1" in phase_name:
            return float(cross_trace_replay_cfg.get("phase1_ratio", 0.0))
        if "Phase 2" in phase_name:
            return float(cross_trace_replay_cfg.get("phase2_ratio", 0.0))
        return float(cross_trace_replay_cfg.get("phase3_ratio", 0.0))

    def _sample_curriculum_batch(pool, phase_name: str):
        """Sample a batch with protected no-fault and evidence replay streams."""
        actual_size = min(batch_size, len(pool))
        if actual_size <= 0:
            return []
        replay_ratio = _phase_replay_ratio(phase_name)
        evidence_ratio = _phase_evidence_ratio(phase_name)
        cross_trace_ratio = _phase_cross_trace_ratio(phase_name)
        batch_rows = []
        for _ in range(actual_size):
            if (
                no_fault_replay_pool
                and replay_ratio > 0.0
                and rng.random() < replay_ratio
            ):
                batch_rows.append(rng.choice(no_fault_replay_pool))
            elif (
                cross_trace_replay_pool
                and cross_trace_ratio > 0.0
                and rng.random() < cross_trace_ratio
            ):
                batch_rows.append(rng.choice(cross_trace_replay_pool))
            elif (
                evidence_replay_pool
                and evidence_ratio > 0.0
                and rng.random() < evidence_ratio
            ):
                batch_rows.append(rng.choice(evidence_replay_pool))
            else:
                batch_rows.append(rng.choice(pool))
        return batch_rows

    def _train_rollout_cap_for_phase(phase_name: str) -> int:
        """Return the phase-specific training rollout cap."""
        if "Phase 1" in phase_name:
            cap = train_rollout_steps_cfg.get("phase1", max_rollout_steps)
        elif "Phase 2" in phase_name:
            cap = train_rollout_steps_cfg.get("phase2", max_rollout_steps)
        else:
            cap = train_rollout_steps_cfg.get("phase3", max_rollout_steps)
        return max(1, min(int(cap), int(max_rollout_steps)))

    # Early stopping configuration
    early_stop_da_thresh = config.get("early_stop_da_threshold", 0.5)
    early_stop_patience = config.get("early_stop_patience", 2)
    da_decline_count = 0

    # ---- Prepare eval test data ----
    eval_test_path = os.path.join(output_dir, "rl_eval_test.jsonl")
    if os.path.exists(config.get("eval_data_path", "")):
        eval_test_path = config["eval_data_path"]
    else:
        # Use a subset of training prompts for evaluation
        import random
        rng = random.Random(42)
        eval_subset = rng.sample(
            train_prompts, min(config.get("diag_eval_episodes", 30), len(train_prompts))
        )
        with open(eval_test_path, "w", encoding="utf-8") as f:
            for entry in eval_subset:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ---- Setup optimizer ----
    from torch.optim import AdamW

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=config["learning_rate"], weight_decay=0.01)

    total_steps = config["num_train_steps"]
    warmup_steps = int(total_steps * config.get("warmup_ratio", 0.05))

    # Simple linear warmup + cosine decay
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        import math
        return 0.5 * (1 + math.cos(math.pi * progress))

    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)

    # ---- Training state ----
    best_reward = -float("inf")
    best_step = 0
    global_step = 0
    start_step = 1
    reward_history = []
    diag_eval_history = []

    import random
    rng = random.Random(42)

    group_size = config["group_size"]
    batch_size = config["per_device_batch_size"]
    accum_steps = config["gradient_accumulation_steps"]
    kl_coeff = config.get("kl_coeff", 0.05)
    max_rollout_steps = config.get("max_rollout_steps", 15)
    train_rollout_steps_cfg = config.get("train_rollout_steps", {}) or {}
    eval_rollout_steps = int(config.get("diag_eval_max_steps", max_rollout_steps))
    temperature = config.get("temperature", 0.8)
    rollout_do_sample = bool(config.get("rollout_do_sample", True))
    rollout_top_p = config.get("rollout_top_p", 0.9)
    max_new_tokens = config.get("max_new_tokens", 1024)
    stop_on_tool_tags = bool(config.get("stop_on_tool_tags", True))
    rollout_generate_timeout_seconds = config.get(
        "rollout_generate_timeout_seconds"
    )
    rollout_max_input_tokens = config.get("rollout_max_input_tokens")
    log_rollout_stages = bool(config.get("log_rollout_stages", False))
    save_train_rollouts = bool(config.get("save_train_rollouts", True))
    rollout_warmup_new_tokens = int(config.get("rollout_warmup_new_tokens", 8) or 0)
    clear_cuda_cache_each_step = bool(
        config.get("clear_cuda_cache_each_step", False)
    )
    gradient_checkpointing_for_loss = bool(
        config.get("gradient_checkpointing_for_loss", True)
    )
    max_loss_context_tokens = int(config.get("max_loss_context_tokens", 4096))
    max_kl_context_tokens = int(config.get("max_kl_context_tokens", 1024))
    log_prob_mode = str(config.get("log_prob_mode", "per_turn")).lower()
    backward_per_trajectory = bool(config.get("backward_per_trajectory", True))
    use_step_level_epo = bool(config.get("use_step_level_epo", True))
    epo_discount = float(config.get("epo_discount", 0.95))
    # EPO step config is shared with the composite reward's epo_config block.
    _rw = config.get("reward_weights") or {}
    epo_step_config = dict(_rw.get("epo_config") or {})
    epo_step_config.setdefault("discount", epo_discount)
    reward_weights = config.get("reward_weights")
    best_model_metric = config.get("best_model_metric", "selection_score")
    early_stop_metric = config.get(
        "early_stop_metric", "balanced_diagnostic_accuracy"
    )

    episodes_dir = ensure_dir(os.path.join(output_dir, "rl_episodes"))
    train_rollouts_path = os.path.join(output_dir, "train_rollouts.jsonl")
    if save_train_rollouts:
        ensure_dir(output_dir)
        open(train_rollouts_path, "w", encoding="utf-8").close()

    def _move_optimizer_state_to_device():
        for state in optimizer.state.values():
            for key, value in list(state.items()):
                if hasattr(value, "to"):
                    state[key] = value.to(model.device)

    def _save_training_state(ckpt_dir: str, step: int):
        torch.save(
            {
                "global_step": step,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng_state": rng.getstate(),
                "best_reward": best_reward,
                "best_step": best_step,
                "reward_history": reward_history,
                "diag_eval_history": diag_eval_history,
                "config": {k: v for k, v in config.items() if not callable(v)},
            },
            os.path.join(ckpt_dir, "training_state.pt"),
        )

    resume_ckpt = config.get("resume_from_checkpoint")
    if resume_ckpt:
        state_path = os.path.join(resume_ckpt, "training_state.pt")
        if os.path.exists(state_path):
            try:
                state = torch.load(
                    state_path, map_location="cpu", weights_only=False
                )
            except TypeError:
                state = torch.load(state_path, map_location="cpu")
            optimizer.load_state_dict(state.get("optimizer", {}))
            scheduler.load_state_dict(state.get("scheduler", {}))
            _move_optimizer_state_to_device()
            if state.get("rng_state") is not None:
                rng.setstate(state["rng_state"])
            best_reward = state.get("best_reward", best_reward)
            best_step = state.get("best_step", best_step)
            reward_history = state.get("reward_history", reward_history)
            diag_eval_history = state.get("diag_eval_history", diag_eval_history)
            global_step = int(state.get("global_step", 0))
            start_step = global_step + 1
            logger.info(
                f"Resumed optimizer/scheduler state from {state_path}; "
                f"continuing at step {start_step}"
            )
        else:
            logger.warning(
                f"resume_from_checkpoint={resume_ckpt} has no training_state.pt; "
                "using model weights as a warm start."
            )

    logger.info("Starting GRPO training with curriculum learning...")
    logger.info(f"  batch_size={batch_size}, group_size={group_size}, "
                f"accum_steps={accum_steps}")
    logger.info(f"  total_steps={total_steps}, kl_coeff={kl_coeff}")
    logger.info(
        f"  best_model_metric={best_model_metric}, "
        f"early_stop_metric={early_stop_metric}"
    )
    logger.info(
        f"  stop_on_tool_tags={stop_on_tool_tags}, "
        f"rollout_do_sample={rollout_do_sample}, "
        f"rollout_top_p={rollout_top_p}, "
        f"model_device_map={config.get('model_device_map', 'single')}, "
        f"resolved_device_map={resolved_device_map}, "
        f"rollout_generate_timeout_seconds={rollout_generate_timeout_seconds}, "
        f"rollout_max_input_tokens={rollout_max_input_tokens}, "
        f"log_rollout_stages={log_rollout_stages}, "
        f"save_train_rollouts={save_train_rollouts}, "
        f"rollout_warmup_new_tokens={rollout_warmup_new_tokens}, "
        f"gradient_checkpointing_for_loss={gradient_checkpointing_for_loss}, "
        f"clear_cuda_cache_each_step={clear_cuda_cache_each_step}, "
        f"backward_per_trajectory={backward_per_trajectory}, "
        f"log_prob_mode={log_prob_mode}, "
        f"max_loss_context_tokens={max_loss_context_tokens}, "
        f"max_kl_context_tokens={max_kl_context_tokens}"
    )
    logger.info(
        f"  rollout caps: train={train_rollout_steps_cfg or max_rollout_steps}, "
        f"hard_cap={max_rollout_steps}, eval={eval_rollout_steps}"
    )
    logger.info(f"  curriculum: phase1(easy)≤{phase1_end}, "
                f"phase2(medium)≤{phase2_end}, phase3(all)>{phase2_end}")
    start_time = time.time()

    model.train()
    optimizer.zero_grad()
    accumulated_loss = 0.0

    if train_prompts and rollout_warmup_new_tokens > 0:
        _warmup_generation(
            model,
            tokenizer,
            train_prompts[0]["messages"],
            max_new_tokens=rollout_warmup_new_tokens,
        )
        model.train()

    # NOTE: gradient_checkpointing is toggled per-phase:
    #   OFF during rollout (generate needs use_cache=True for speed)
    #   ON during loss computation (backward pass needs memory saving)

    for step in range(start_step, total_steps + 1):
        # ---- Sample from curriculum-appropriate pool ----
        pool, phase_name = _get_curriculum_pool(step)
        train_rollout_cap = _train_rollout_cap_for_phase(phase_name)
        if step in (1, phase1_end + 1, phase2_end + 1):
            logger.info(
                f"  Curriculum: entering {phase_name} "
                f"({len(pool)} prompts available, rollout_cap={train_rollout_cap})"
            )
        batch = _sample_curriculum_batch(pool, phase_name)

        step_rewards = []
        step_loss_value = 0.0
        step_backward_count = 0
        step_loss = None
        step_tool_counts = []
        step_no_diag = 0
        step_hit_cap = 0
        step_invalid_no_action = 0
        step_sensor_calls = 0
        step_sensor_uncertain_nodes = 0
        step_sensor_verified_nodes = 0
        step_sensor_post_fault_calls = 0

        for prompt_entry in batch:
            messages = prompt_entry["messages"]
            ground_truth = prompt_entry.get("ground_truth", {})

            # ---- Generate G multi-turn rollouts ----
            group_rollouts = []
            group_rewards_raw = []
            group_epo_steps = []  # per-rollout step-level EPO reward bundles

            # Load scenario state for Oracle predictions
            sid = prompt_entry.get("id", prompt_entry.get("metadata", {}).get("scenario_id", ""))
            state = _load_rl_scenario_state(
                sid,
                ground_truth=ground_truth,
                scenario_type=prompt_entry.get("metadata", {}).get("scenario_type", ""),
                metadata=prompt_entry.get("metadata", {}) or {},
            )
            if tool_executor is not None:
                tool_executor.set_scenario_state(state)

            for g in range(group_size):
                # Disable gradient checkpointing for fast generation
                model.gradient_checkpointing_disable()
                _set_model_use_cache(model, True)
                model.eval()  # eval mode for generate (enables use_cache)

                rollout = _run_multi_turn_rollout(
                    model=model,
                    tokenizer=tokenizer,
                    messages=messages,
                    tool_executor=tool_executor,
                    max_steps=train_rollout_cap,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    stop_on_tool_tags=stop_on_tool_tags,
                    do_sample=rollout_do_sample,
                    top_p=rollout_top_p,
                    generate_timeout_seconds=rollout_generate_timeout_seconds,
                    max_input_tokens=rollout_max_input_tokens,
                    log_stages=log_rollout_stages,
                    rollout_label=f"step={step} g={g+1}/{group_size}",
                )

                model.train()  # back to train mode

                # Compute reward (pass tool_results for topology checking)
                reward_dict = compute_total_reward(
                    agent_outputs=rollout["agent_outputs"],
                    final_diagnosis=rollout["final_diagnosis"],
                    ground_truth=ground_truth,
                    n_tool_calls=rollout["n_tool_calls"],
                    optimal_path_length=ground_truth.get("optimal_path_length", 3),
                    max_steps=train_rollout_cap,
                    weights=reward_weights,
                    tool_results=rollout.get("tool_results"),
                )

                group_rollouts.append(rollout)
                group_rewards_raw.append(reward_dict["total"])
                if use_step_level_epo:
                    epo_steps = compute_epo_step_rewards(
                        agent_outputs=rollout["agent_outputs"],
                        final_diagnosis=rollout["final_diagnosis"],
                        ground_truth=ground_truth,
                        n_tool_calls=rollout["n_tool_calls"],
                        optimal_path_length=ground_truth.get("optimal_path_length", 3),
                        max_steps=train_rollout_cap,
                        tool_results=rollout.get("tool_results"),
                        config=epo_step_config,
                    )
                    group_epo_steps.append(epo_steps)
                step_tool_counts.append(rollout["n_tool_calls"])
                if rollout["final_diagnosis"] is None:
                    step_no_diag += 1
                if rollout.get("hit_max_steps"):
                    step_hit_cap += 1
                if rollout.get("invalid_no_action"):
                    step_invalid_no_action += 1
                step_sensor_calls += int(reward_dict.get("sensor_calls", 0))
                step_sensor_uncertain_nodes += int(
                    reward_dict.get("sensor_uncertain_nodes", 0)
                )
                step_sensor_verified_nodes += int(
                    reward_dict.get("sensor_verified_uncertain_nodes", 0)
                )
                step_sensor_post_fault_calls += int(
                    reward_dict.get("sensor_post_fault_calls", 0)
                )

                # Per-rollout progress log
                diag = rollout["final_diagnosis"]
                diag_str = f"{diag.get('root_cause_node','?')}/{diag.get('fault_type','?')}" if diag else "no_diag"
                logger.info(
                    f"  Step {step} rollout {g+1}/{group_size}: "
                    f"tools={rollout['n_tool_calls']}, "
                    f"hit_cap={int(bool(rollout.get('hit_max_steps')))}, "
                    f"gen_timeout={int(bool(rollout.get('generation_timeout_hit')))}, "
                    f"max_prompt_tokens="
                    f"{max(rollout.get('prompt_token_lengths') or [0])}, "
                    f"gen_time="
                    f"{sum(rollout.get('generation_durations') or [0.0]):.1f}s, "
                    f"reward={reward_dict['total']:.3f}, "
                    f"acc={reward_dict.get('accuracy', 0.0):.3f}, "
                    f"proc={reward_dict.get('process', 0.0):.3f}, "
                    f"sv={reward_dict.get('sensor_verification', 0.0):.3f}, "
                    f"sensors={reward_dict.get('sensor_calls', 0)}, "
                    f"diag={diag_str}"
                )

                if save_train_rollouts:
                    _save_jsonl_record(
                        train_rollouts_path,
                        {
                            "step": step,
                            "group_index": g,
                            "phase": phase_name,
                            "prompt_id": sid,
                            "scenario_type": prompt_entry.get(
                                "metadata", {}
                            ).get("scenario_type", ""),
                            "ground_truth": ground_truth,
                            "final_diagnosis": rollout["final_diagnosis"],
                            "n_tool_calls": rollout["n_tool_calls"],
                            "hit_max_steps": rollout.get("hit_max_steps"),
                            "invalid_no_action": rollout.get(
                                "invalid_no_action"
                            ),
                            "generation_timeout_hit": rollout.get(
                                "generation_timeout_hit"
                            ),
                            "prompt_token_lengths": rollout.get(
                                "prompt_token_lengths"
                            ),
                            "generation_durations": rollout.get(
                                "generation_durations"
                            ),
                            "reward": reward_dict,
                            "agent_outputs": rollout["agent_outputs"],
                            "tool_results": rollout.get("tool_results", []),
                        },
                    )

            # ---- Compute advantages ----
            # Step-level EPO (default): per-step discounted returns with a
            # group + leave-one-out baseline at each step index, giving genuine
            # step-level credit assignment (eq:epo_pg_loss). Legacy path: a
            # single trajectory-scalar leave-one-out advantage.
            n_rollouts = len(group_rewards_raw)
            max_adv = 2.0

            step_returns_per_rollout: List[List[float]] = []
            if use_step_level_epo and group_epo_steps:
                for bundle in group_epo_steps:
                    step_returns_per_rollout.append(
                        _epo_step_returns(
                            bundle["step_rewards"],
                            bundle["terminal_reward"],
                            bundle.get("discount", epo_discount),
                        )
                    )
                # Per-step-index group baseline (leave-one-out over rollouts that
                # have a return at that index).
                max_len = max((len(r) for r in step_returns_per_rollout), default=0)
                step_baselines: List[float] = []
                for t in range(max_len):
                    vals = [r[t] for r in step_returns_per_rollout if t < len(r)]
                    step_baselines.append(sum(vals) / max(len(vals), 1))
                # Trajectory-mean return drives logging/checkpoint comparisons.
                traj_scalar = [
                    (sum(r) / max(len(r), 1)) if r else 0.0
                    for r in step_returns_per_rollout
                ]
            else:
                step_baselines = []
                traj_scalar = list(group_rewards_raw)

            # Legacy scalar advantage (also used as fallback when a rollout has
            # no per-turn alignment).
            scalar_advantages = []
            for i in range(n_rollouts):
                others = [r for j, r in enumerate(traj_scalar) if j != i]
                baseline = sum(others) / max(len(others), 1)
                scalar_advantages.append(
                    max(-max_adv, min(max_adv, traj_scalar[i] - baseline))
                )

            # ---- Compute policy gradient loss (WITH gradient) ----
            if gradient_checkpointing_for_loss:
                _set_model_use_cache(model, False)
                model.gradient_checkpointing_enable()
            else:
                model.gradient_checkpointing_disable()
                _set_model_use_cache(model, True)
            if clear_cuda_cache_each_step and torch.cuda.is_available():
                torch.cuda.empty_cache()

            for ridx, rollout in enumerate(group_rollouts):
                if not rollout["all_completion_ids"]:
                    continue

                kl = _compute_kl_divergence(
                    model, ref_model, tokenizer,
                    rollout["conversation"],
                    is_peft=is_peft_model,
                    policy_adapter_name=policy_adapter_name,
                    ref_adapter_name=ref_adapter_name,
                    max_context_tokens=max_kl_context_tokens,
                )

                pg_term = None
                if use_step_level_epo and ridx < len(step_returns_per_rollout):
                    # Per-turn log-probs aligned to step returns.
                    turn_lps = _compute_per_turn_log_prob_list(
                        model, tokenizer,
                        rollout["conversation"],
                        max_context_tokens=max_loss_context_tokens,
                    )
                    returns = step_returns_per_rollout[ridx]
                    n_align = min(len(turn_lps), len(returns))
                    if n_align > 0:
                        adv_terms = []
                        for t in range(n_align):
                            baseline = step_baselines[t] if t < len(step_baselines) else 0.0
                            a_t = returns[t] - baseline
                            a_t = max(-max_adv, min(max_adv, a_t))
                            adv_terms.append(-a_t * turn_lps[t])
                        pg_term = torch.stack(adv_terms).sum() / max(n_align, 1)

                if pg_term is None:
                    # Fallback: scalar advantage × summed log-prob.
                    adv = scalar_advantages[ridx]
                    if log_prob_mode == "per_turn":
                        log_prob = _compute_per_turn_log_prob(
                            model, tokenizer,
                            rollout["conversation"],
                            max_context_tokens=max_loss_context_tokens,
                        )
                    else:
                        log_prob = _compute_trajectory_log_prob(
                            model, tokenizer,
                            rollout["conversation"],
                            rollout["all_completion_ids"],
                            max_context_tokens=max_loss_context_tokens,
                        )
                    pg_term = -adv * log_prob

                loss = pg_term + kl_coeff * kl
                scaled_loss = loss / (batch_size * group_size)
                step_loss_value += float(scaled_loss.detach().item())
                if backward_per_trajectory and scaled_loss.requires_grad:
                    scaled_loss.backward()
                    step_backward_count += 1
                    del scaled_loss, loss, pg_term, kl
                    if clear_cuda_cache_each_step and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    if step_loss is None:
                        step_loss = torch.tensor(
                            0.0, device=model.device, requires_grad=False
                        )
                    step_loss = step_loss + scaled_loss

            step_rewards.extend(traj_scalar)

        # ---- Backward pass ----
        if not backward_per_trajectory:
            if step_loss is None:
                step_loss = torch.tensor(
                    0.0, device=model.device, requires_grad=False
                )
            effective_loss = step_loss
            if effective_loss.requires_grad:
                effective_loss.backward()
            step_loss_value = float(effective_loss.detach().item())
        model.gradient_checkpointing_disable()  # disable for next rollout phase

        if step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ---- Logging ----
        global_step = step
        avg_reward = sum(step_rewards) / max(len(step_rewards), 1)
        avg_tools = sum(step_tool_counts) / max(len(step_tool_counts), 1)
        rollout_count = max(len(step_tool_counts), 1)
        no_diag_rate = step_no_diag / rollout_count
        hit_cap_rate = step_hit_cap / rollout_count
        sensor_calls_per_rollout = step_sensor_calls / rollout_count
        sensor_verification_coverage = (
            step_sensor_verified_nodes / max(step_sensor_uncertain_nodes, 1)
        )
        sensor_post_fault_rate = (
            step_sensor_post_fault_calls / max(step_sensor_calls, 1)
        )
        reward_history.append({
            "step": step,
            "avg_reward": avg_reward,
            "loss": step_loss_value,
            "lr": scheduler.get_last_lr()[0],
            "backward_count": step_backward_count,
            "phase": phase_name,
            "train_rollout_cap": train_rollout_cap,
            "avg_tools": avg_tools,
            "no_diag_rate": no_diag_rate,
            "hit_max_steps_rate": hit_cap_rate,
            "invalid_no_action_rate": step_invalid_no_action / rollout_count,
            "sensor_calls_per_rollout": sensor_calls_per_rollout,
            "sensor_uncertain_nodes": step_sensor_uncertain_nodes,
            "sensor_verified_uncertain_nodes": step_sensor_verified_nodes,
            "sensor_verification_coverage": sensor_verification_coverage,
            "sensor_post_fault_rate": sensor_post_fault_rate,
        })

        if step % 10 == 0 or step == 1:
            elapsed = time.time() - start_time
            sec_per_step = elapsed / step
            remaining = sec_per_step * (total_steps - step)
            eta_h, eta_m = divmod(int(remaining), 3600)
            eta_m //= 60
            logger.info(
                f"Step {step}/{total_steps}: "
                f"avg_reward={avg_reward:.4f}, "
                f"avg_tools={avg_tools:.2f}, "
                f"no_diag={no_diag_rate:.1%}, "
                f"hit_cap={hit_cap_rate:.1%}, "
                f"sensor_cov={sensor_verification_coverage:.1%}, "
                f"sensor_calls={sensor_calls_per_rollout:.2f}/rollout, "
                f"loss={step_loss_value:.4f}, "
                f"lr={scheduler.get_last_lr()[0]:.2e}, "
                f"elapsed={elapsed/60:.0f}min, "
                f"ETA={eta_h}h{eta_m:02d}m"
            )

        # ---- Step-based Oracle evaluation ----
        run_diag_eval = (
            bool(config.get("run_diagnostic_eval", False))
            and int(config.get("diag_eval_episodes", 0)) > 0
        )
        if run_diag_eval and step % config["eval_steps"] == 0:
            logger.info(f"\n=== RL Diagnostic Eval at step {step} ===")
            try:
                from src.evaluation.evaluator import run_diagnostic_eval_with_model

                eval_result = run_diagnostic_eval_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    test_scenarios_path=eval_test_path,
                    max_episodes=config.get("diag_eval_episodes", 30),
                    max_steps=eval_rollout_steps,
                    max_new_tokens=config.get("diag_eval_max_new_tokens", 256),
                    output_dir=episodes_dir,
                    step=step,
                    eval_csv_nrows=int(config.get("diag_eval_csv_nrows", 10000)),
                    episode_timeout_seconds=config.get(
                        "diag_eval_episode_timeout_seconds"
                    ),
                    generate_timeout_seconds=config.get(
                        "diag_eval_generate_timeout_seconds"
                    ),
                    sampling_strategy=config.get("diag_eval_sampling", "stratified"),
                    sampling_seed=config.get("diag_eval_seed", 42),
                    samples_per_type=config.get("diag_eval_samples_per_type"),
                    scenario_types=config.get("diag_eval_scenario_types"),
                    rotate_sampling_seed=bool(
                        config.get("diag_eval_rotate_samples", False)
                    ),
                    oracle_mode=config.get("oracle_mode", "real"),
                    include_system_health=bool(
                        config.get("include_system_health", False)
                    ),
                    expose_status_summary=bool(
                        config.get("expose_status_summary", False)
                    ),
                )

                metrics = eval_result.get("overall", {})
                diag_eval_history.append({"step": step, **metrics})

                logger.info(f"  Diagnostic metrics at step {step}:")
                for k, v in metrics.items():
                    logger.info(f"    {k}: {v:.4f}")

                # Save incremental history
                save_json(
                    diag_eval_history,
                    os.path.join(output_dir, "rl_diag_eval_history.json"),
                )

                # Check for best model using a balanced diagnostic metric. This
                # avoids selecting checkpoints that look good only because
                # no-fault or formatting metrics are easy.
                selection_score = metrics.get(
                    best_model_metric,
                    metrics.get("selection_score", metrics.get("aggregate_score", avg_reward)),
                )
                min_cross_best_da = float(config.get("min_cross_best_da", 0.20))
                cross_da_for_selection = metrics.get(
                    "cross_balanced_diagnostic_accuracy", 1.0
                )
                eligible_for_best = cross_da_for_selection >= min_cross_best_da
                if not eligible_for_best:
                    logger.info(
                        "  Checkpoint not eligible for best: "
                        f"cross_balanced_diagnostic_accuracy="
                        f"{cross_da_for_selection:.4f} < {min_cross_best_da:.4f}"
                    )
                if selection_score > best_reward:
                    if eligible_for_best:
                        best_reward = selection_score
                        best_step = step
                        ckpt_dir = os.path.join(output_dir, "best")
                        _save_policy_checkpoint(
                            model,
                            tokenizer,
                            ckpt_dir,
                            is_peft_model=is_peft_model,
                            policy_adapter_name=policy_adapter_name,
                        )
                        _save_training_state(ckpt_dir, step)
                        logger.info(f"  New best checkpoint at step {step} "
                                    f"({best_model_metric}={selection_score:.4f})")

            except Exception as e:
                logger.warning(
                    f"RL eval failed at step {step}: {type(e).__name__}: {e}"
                )

            # ---- Early stopping check ----
            if len(diag_eval_history) >= 2:
                latest_metric = diag_eval_history[-1].get(
                    early_stop_metric,
                    diag_eval_history[-1].get("diagnostic_accuracy", 0),
                )
                if latest_metric < early_stop_da_thresh:
                    da_decline_count += 1
                    logger.warning(
                        f"  {early_stop_metric} below threshold "
                        f"({latest_metric:.1%} < {early_stop_da_thresh:.1%}), "
                        f"decline count: {da_decline_count}/{early_stop_patience}"
                    )
                    if da_decline_count >= early_stop_patience:
                        logger.warning(
                            f"  EARLY STOPPING at step {step}: {early_stop_metric} "
                            f"below {early_stop_da_thresh:.1%} "
                            f"for {early_stop_patience} consecutive evals."
                        )
                        break
                else:
                    da_decline_count = 0  # Reset counter on improvement

        # ---- Save periodic checkpoint ----
        if step % config["save_steps"] == 0:
            ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
            _save_policy_checkpoint(
                model,
                tokenizer,
                ckpt_dir,
                is_peft_model=is_peft_model,
                policy_adapter_name=policy_adapter_name,
            )
            _save_training_state(ckpt_dir, step)

    elapsed = time.time() - start_time

    # ---- Save final model ----
    final_dir = os.path.join(output_dir, "final")
    _save_policy_checkpoint(
        model,
        tokenizer,
        final_dir,
        is_peft_model=is_peft_model,
        policy_adapter_name=policy_adapter_name,
    )
    _save_training_state(final_dir, global_step)

    best_dir = os.path.join(output_dir, "best")
    if not os.path.exists(os.path.join(best_dir, "adapter_config.json")):
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        shutil.copytree(final_dir, best_dir)
        best_step = global_step
        if reward_history:
            best_reward = reward_history[-1].get("avg_reward", 0.0)
        logger.info(
            "No diagnostic-selected RL checkpoint was produced; "
            f"using final checkpoint as {best_dir}."
        )

    # ---- Save training summary ----
    summary = {
        "training_time_seconds": elapsed,
        "total_steps": global_step,
        "best_step": best_step,
        "best_reward": best_reward,
        "reward_history": reward_history,
        "diag_eval_history": diag_eval_history,
        "config": {k: v for k, v in config.items() if not callable(v)},
    }
    save_json(summary, os.path.join(output_dir, "training_summary.json"))

    logger.info(
        f"GRPO training complete in {elapsed / 3600:.1f}h. "
        f"Best checkpoint at step {best_step} (score={best_reward:.4f})"
    )

    return best_dir


# ============================================================================
# Helper
# ============================================================================

def _parse_diagnosis_from_text(text: str) -> Optional[Dict]:
    """Parse final diagnosis from generated text."""
    match = re.search(r"<diagnosis>\s*(\{.*?\})\s*</diagnosis>", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass
    return None
