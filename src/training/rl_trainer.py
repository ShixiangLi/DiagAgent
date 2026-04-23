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
import time
from typing import Any, Dict, List, Optional

from src.training.reward_functions import compute_total_reward
from src.utils.io_utils import load_yaml, save_json, ensure_dir, setup_logger

logger = setup_logger(__name__)


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

    # Training
    "num_train_steps": 500,
    "per_device_batch_size": 2,    # Prompts per batch (×group_size = rollouts)
    "learning_rate": 5e-6,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.05,
    "gradient_accumulation_steps": 4,
    "bf16": True,
    "max_new_tokens": 1024,        # Max tokens per generation step

    # Evaluation
    "eval_steps": 50,
    "diag_eval_episodes": 30,
    "save_steps": 50,

    # Reward weights
    "reward_weights": {
        "accuracy": 0.35,
        "efficiency": 0.20,
        "format": 0.15,
        "reasoning": 0.15,
        "completeness": 0.15,
    },

    # Data
    "train_data_path": "outputs/data/rl_train.jsonl",
    "eval_data_path": "outputs/data/rl_val.jsonl",
}


def load_rl_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load RL config, merging with defaults."""
    config = DEFAULT_RL_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        user_config = load_yaml(config_path)
        config.update(user_config)
    return config


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

    conversation = list(messages)

    for step in range(max_steps):
        prompt = tokenizer.apply_chat_template(
            conversation, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        prompt_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )

        completion_ids = output[0][prompt_len:]
        response = tokenizer.decode(completion_ids, skip_special_tokens=True)

        # Truncate at </tool_call> to prevent hallucinated tool responses
        tc_end_pos = response.find("</tool_call>")
        if tc_end_pos != -1:
            response = response[:tc_end_pos + len("</tool_call>")]
            # Also truncate the completion_ids to match
            truncated_text = response
            truncated_ids = tokenizer.encode(
                truncated_text, add_special_tokens=False,
                return_tensors="pt",
            ).to(completion_ids.device)[0]
            completion_ids = truncated_ids

        agent_outputs.append(response)
        all_completion_ids.append(completion_ids)
        conversation.append({"role": "assistant", "content": response})

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
    }


def _compute_trajectory_log_prob(
    model,
    tokenizer,
    conversation: List[Dict],
    all_completion_ids: List,
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
    ).to(model.device)

    # Truncate to avoid OOM on very long multi-turn conversations
    max_len = 2048
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

    return assistant_log_probs.mean()


def _compute_kl_divergence(
    model,
    ref_model,
    tokenizer,
    conversation: List[Dict],
    is_peft: bool = False,
) -> "torch.Tensor":
    """
    Compute KL(π_θ || π_ref) for the conversation.

    If is_peft=True and ref_model is None, uses adapter toggling:
    disable adapters to get ref logits, re-enable for policy logits.
    This avoids loading a separate reference model (~14GB saved).

    Returns scalar KL divergence tensor.
    """
    import torch
    import torch.nn.functional as F

    full_text = tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=False
    )
    full_ids = tokenizer.encode(full_text, return_tensors="pt").to(model.device)

    # Truncate to avoid OOM on very long sequences
    max_len = 2048
    if full_ids.shape[1] > max_len:
        full_ids = full_ids[:, -max_len:]

    if ref_model is None and is_peft:
        # Shared base model: toggle adapters
        with torch.no_grad():
            model.disable_adapter_layers()
            ref_logits = model(full_ids).logits
            model.enable_adapter_layers()
    elif ref_model is not None:
        with torch.no_grad():
            ref_logits = ref_model(full_ids).logits
    else:
        # No ref model available, return zero KL
        return torch.tensor(0.0, device=model.device)

    policy_logits = model(full_ids).logits

    # KL divergence on last 256 tokens (where assistant content is)
    n_tokens = min(256, policy_logits.shape[1])
    p_log = F.log_softmax(policy_logits[0, -n_tokens:, :], dim=-1)
    q_log = F.log_softmax(ref_logits[0, -n_tokens:, :], dim=-1)

    kl = F.kl_div(q_log, p_log, log_target=True, reduction="batchmean")
    return kl


# ============================================================================
# GRPO Training Loop
# ============================================================================

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

    # ---- Load policy model (shared base for policy + reference) ----
    logger.info(f"Loading SFT model from: {config['model_name_or_path']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name_or_path"],
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    is_peft_model = False
    try:
        base_model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            base_model, config["model_name_or_path"],
            is_trainable=True,
        )
        is_peft_model = True
        logger.info("Loaded as trainable PEFT model (shared base for policy + ref)")
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name_or_path"],
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        logger.info("Loaded as full model (no ref model for KL)")

    # NOTE: No separate ref_model loaded!
    # For PEFT models, KL is computed by toggling adapter layers on/off.
    # This saves ~14GB VRAM (one full 7B model).
    ref_model = None
    logger.info("Memory optimization: shared base model, no separate ref_model")

    # ---- Load Oracle environment ----
    from src.evaluation.evaluator import _create_tool_environment
    builder, tool_executor, model_registry = _create_tool_environment()
    if tool_executor is None:
        raise RuntimeError("Failed to create Oracle environment for RL training")

    # ---- Build scenario state loader ----
    import pandas as pd
    from src.environment.fault_scenario import FaultScenario, create_scenario_state
    from src.environment.diagnostic_path import DiagnosticPath, PathNode
    from src.node_models.data_loader import discover_fault_files

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
    file_path_map = {}  # (sys_id, filename) -> filepath
    for sys_id in topo_config.get("systems", {}):
        try:
            for ff in discover_fault_files("data/lbnl", sys_id):
                file_path_map[(sys_id, ff.filename)] = ff.filepath
        except Exception:
            pass

    csv_cache = {}  # (sys_id, filename) -> DataFrame

    def _load_rl_scenario_state(scenario_id: str):
        """Load scenario state for an RL prompt."""
        fs = scenario_lookup.get(scenario_id)
        if fs is None:
            return None
        key = (fs.root_cause_system, fs.source_file)
        if key not in csv_cache:
            fpath = file_path_map.get(key)
            if fpath and os.path.exists(fpath):
                df = pd.read_csv(fpath, nrows=50000, low_memory=False)
                csv_cache[key] = df.select_dtypes(include=["number"])
            else:
                return None
        if key not in csv_cache:
            return None
        try:
            return create_scenario_state(
                fs, {fs.root_cause_system: csv_cache[key]},
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
    reward_history = []
    diag_eval_history = []

    import random
    rng = random.Random(42)

    group_size = config["group_size"]
    batch_size = config["per_device_batch_size"]
    accum_steps = config["gradient_accumulation_steps"]
    kl_coeff = config.get("kl_coeff", 0.05)
    max_rollout_steps = config.get("max_rollout_steps", 15)
    temperature = config.get("temperature", 0.8)
    max_new_tokens = config.get("max_new_tokens", 1024)
    reward_weights = config.get("reward_weights")

    episodes_dir = ensure_dir(os.path.join(output_dir, "rl_episodes"))

    logger.info("Starting GRPO training with multi-turn Oracle rollouts...")
    logger.info(f"  batch_size={batch_size}, group_size={group_size}, "
                f"accum_steps={accum_steps}")
    logger.info(f"  total_steps={total_steps}, kl_coeff={kl_coeff}")
    start_time = time.time()

    model.train()
    optimizer.zero_grad()
    accumulated_loss = 0.0

    # NOTE: gradient_checkpointing is toggled per-phase:
    #   OFF during rollout (generate needs use_cache=True for speed)
    #   ON during loss computation (backward pass needs memory saving)

    for step in range(1, total_steps + 1):
        # ---- Sample a batch of prompts ----
        batch = rng.sample(train_prompts, min(batch_size, len(train_prompts)))

        step_rewards = []
        step_loss = torch.tensor(0.0, device=model.device, requires_grad=False)

        for prompt_entry in batch:
            messages = prompt_entry["messages"]
            ground_truth = prompt_entry.get("ground_truth", {})

            # ---- Generate G multi-turn rollouts ----
            group_rollouts = []
            group_rewards_raw = []

            # Load scenario state for Oracle predictions
            sid = prompt_entry.get("id", prompt_entry.get("metadata", {}).get("scenario_id", ""))
            state = _load_rl_scenario_state(sid)
            if state is not None:
                tool_executor.set_scenario_state(state)

            for g in range(group_size):
                # Disable gradient checkpointing for fast generation
                model.gradient_checkpointing_disable()
                model.eval()  # eval mode for generate (enables use_cache)

                rollout = _run_multi_turn_rollout(
                    model=model,
                    tokenizer=tokenizer,
                    messages=messages,
                    tool_executor=tool_executor,
                    max_steps=max_rollout_steps,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                )

                model.train()  # back to train mode

                # Compute reward
                reward_dict = compute_total_reward(
                    agent_outputs=rollout["agent_outputs"],
                    final_diagnosis=rollout["final_diagnosis"],
                    ground_truth=ground_truth,
                    n_tool_calls=rollout["n_tool_calls"],
                    optimal_path_length=ground_truth.get("optimal_path_length", 3),
                    weights=reward_weights,
                )

                group_rollouts.append(rollout)
                group_rewards_raw.append(reward_dict["total"])

                # Per-rollout progress log
                diag = rollout["final_diagnosis"]
                diag_str = f"{diag.get('root_cause_node','?')}/{diag.get('fault_type','?')}" if diag else "no_diag"
                logger.info(
                    f"  Step {step} rollout {g+1}/{group_size}: "
                    f"tools={rollout['n_tool_calls']}, "
                    f"reward={reward_dict['total']:.3f}, "
                    f"diag={diag_str}"
                )

            # ---- Compute group-relative advantages ----
            mean_reward = sum(group_rewards_raw) / len(group_rewards_raw)
            std_reward = max(
                (sum((r - mean_reward) ** 2 for r in group_rewards_raw) / len(group_rewards_raw)) ** 0.5,
                1e-8,
            )
            advantages = [(r - mean_reward) / std_reward for r in group_rewards_raw]

            # ---- Compute policy gradient loss (WITH gradient) ----
            # Enable gradient checkpointing for memory-efficient forward passes
            model.gradient_checkpointing_enable()
            torch.cuda.empty_cache()  # Free rollout KV cache before loss computation

            for adv, rollout in zip(advantages, group_rollouts):
                if not rollout["all_completion_ids"]:
                    continue

                # Re-compute log probability with gradient
                log_prob = _compute_trajectory_log_prob(
                    model, tokenizer,
                    rollout["conversation"],
                    rollout["all_completion_ids"],
                )

                # KL penalty (using adapter toggling if PEFT)
                kl = _compute_kl_divergence(
                    model, ref_model, tokenizer,
                    rollout["conversation"],
                    is_peft=is_peft_model,
                )

                # GRPO loss: -advantage * log_prob + kl_coeff * KL
                loss = -adv * log_prob + kl_coeff * kl
                step_loss = step_loss + loss

            step_rewards.extend(group_rewards_raw)

        # ---- Backward pass ----
        effective_loss = step_loss / (batch_size * group_size)
        effective_loss.backward()
        model.gradient_checkpointing_disable()  # disable for next rollout phase

        if step % accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        # ---- Logging ----
        global_step = step
        avg_reward = sum(step_rewards) / max(len(step_rewards), 1)
        reward_history.append({
            "step": step,
            "avg_reward": avg_reward,
            "loss": effective_loss.item(),
            "lr": scheduler.get_last_lr()[0],
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
                f"loss={effective_loss.item():.4f}, "
                f"lr={scheduler.get_last_lr()[0]:.2e}, "
                f"elapsed={elapsed/60:.0f}min, "
                f"ETA={eta_h}h{eta_m:02d}m"
            )

        # ---- Step-based Oracle evaluation ----
        if step % config["eval_steps"] == 0:
            logger.info(f"\n=== RL Diagnostic Eval at step {step} ===")
            try:
                from src.evaluation.evaluator import run_diagnostic_eval_with_model

                eval_result = run_diagnostic_eval_with_model(
                    model=model,
                    tokenizer=tokenizer,
                    test_scenarios_path=eval_test_path,
                    max_episodes=config.get("diag_eval_episodes", 30),
                    max_steps=max_rollout_steps,
                    output_dir=episodes_dir,
                    step=step,
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

                # Check for best model
                agg_score = metrics.get("aggregate_score", avg_reward)
                if agg_score > best_reward:
                    best_reward = agg_score
                    best_step = step
                    ckpt_dir = os.path.join(output_dir, "best")
                    model.save_pretrained(ckpt_dir)
                    tokenizer.save_pretrained(ckpt_dir)
                    logger.info(f"  New best checkpoint at step {step} "
                                f"(score={agg_score:.4f})")

            except Exception as e:
                logger.warning(
                    f"RL eval failed at step {step}: {type(e).__name__}: {e}"
                )

        # ---- Save periodic checkpoint ----
        if step % config["save_steps"] == 0:
            ckpt_dir = os.path.join(output_dir, f"checkpoint-{step}")
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)

    elapsed = time.time() - start_time

    # ---- Save final model ----
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

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

    return os.path.join(output_dir, "best")


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
