"""
SFT Trainer — Supervised Fine-Tuning for Qwen2.5-7B-Instruct.

Implements LoRA-based fine-tuning with:
  - Configurable LoRA rank and target modules
  - Train/eval split with step-based evaluation on held-out set
  - Best checkpoint tracking based on eval_loss
  - Loss masking (only compute loss on assistant tokens)
"""

import json
import os
import re
import shutil
import time
from typing import Any, Dict, List, Optional

from src.data_gen.sft_formatter import (
    SFT_DESIGN_VERSION,
    strip_topoitl_labels,
    system_prompt_sha256,
)
from src.training.hf_loading import cached_from_pretrained
from src.utils.io_utils import load_yaml, save_json, ensure_dir, setup_logger

logger = setup_logger(__name__)


def _fmt_metric(value: Any) -> str:
    """Format numeric metrics while keeping diagnostic callbacks robust."""
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def _metric_value(metrics: Dict[str, Any], key: str, default: float = 0.0) -> float:
    value = metrics.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _distributed_info() -> Dict[str, int | bool]:
    """Return torchrun distributed metadata from environment variables."""
    world_size = int(os.environ.get("WORLD_SIZE", "1") or "1")
    rank = int(os.environ.get("RANK", "0") or "0")
    local_rank = int(os.environ.get("LOCAL_RANK", "-1") or "-1")
    return {
        "enabled": world_size > 1,
        "world_size": world_size,
        "rank": rank,
        "local_rank": local_rank,
        "is_main": rank == 0,
    }


# ============================================================================
# Configuration Defaults
# ============================================================================

DEFAULT_SFT_CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
    "adapter_name_or_path": None,
    "output_dir": "outputs/sft",
    "data_path": "outputs/data/sft_train.jsonl",

    # LoRA config
    "lora_rank": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # Training hyperparameters
    "num_epochs": 3,
    "max_steps": -1,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "learning_rate": 2e-5,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 100,
    "max_seq_length": 16384,
    "weight_decay": 0.01,
    "bf16": True,
    "gradient_checkpointing": True,
    "attn_implementation": None,
    "group_by_length": False,
    "dataloader_num_workers": 0,
    "scale_grad_accum_by_world_size": True,

    # Evaluation
    "eval_split_ratio": 0.1,
    "eval_max_samples": None,
    "diag_eval_source_max_samples": None,
    "eval_steps": 200,
    "save_steps": 200,
    "logging_steps": 10,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "run_diagnostic_eval": False,
    "diagnostic_eval_mode": "trainer_callback",
    "select_best_by_diagnostic": True,
    "diagnostic_best_metric": "selection_score",
    "diag_eval_episodes": 64,
    "diag_eval_max_steps": 15,
    "diag_eval_max_new_tokens": 256,
    "diag_eval_episode_timeout_seconds": 180.0,
    "diag_eval_generate_timeout_seconds": 60.0,
    "run_post_train_eval": False,
    "post_train_eval_episodes": 64,
    "post_train_eval_max_new_tokens": 256,
    "post_train_eval_episode_timeout_seconds": 240.0,
    "post_train_eval_generate_timeout_seconds": 90.0,
    "post_train_eval_samples_per_type": None,
    "diag_eval_sampling": "stratified",
    "diag_eval_samples_per_type": None,
    "diag_eval_scenario_types": None,
    "diag_eval_seed": 42,
    "diag_eval_oracle_mode": "real",
    "diag_eval_include_system_health": False,
    "diag_eval_expose_status_summary": False,

    # Dataset
    "dataset_format": "sharegpt",  # "sharegpt" or "messages"
    "strip_topoitl_labels_for_training": False,

    # TopoITL three-term weighted supervision (eq:topoitl_loss).
    # state=λ_z, action=λ_act, transition=λ_tr (added on non-initial state
    # blocks), base=weight for <think> rationale tokens. All 1.0 ⇒ uniform CE.
    # Ablations: set state/transition to 0 for "w/o state supervision", or
    # base low + action high to emphasize action cloning.
    "topoitl_loss_weights": {
        "state": 1.0,
        "action": 1.0,
        "transition": 0.5,
        "base": 1.0,
    },
}


def load_sft_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load SFT config, merging with defaults."""
    config = DEFAULT_SFT_CONFIG.copy()
    if config_path and os.path.exists(config_path):
        user_config = load_yaml(config_path)
        config.update(user_config)
    return config


# ============================================================================
# Data Loading
# ============================================================================

def validate_sft_data(data: List[Dict]) -> Dict[str, int]:
    """
    Run quality pre-checks on loaded SFT data.

    Checks for issues that would degrade training quality:
    - Multiple <think> blocks in a single assistant turn
    - Missing conversations or system prompt
    """
    import re

    expected_prompt_hash = system_prompt_sha256()
    issues = {
        "multi_think": 0,
        "empty_conversations": 0,
        "missing_system": 0,
        "missing_final_diagnosis": 0,
        "stale_design_version": 0,
        "prompt_hash_mismatch": 0,
        "terminal_closure_bad_first_tool": 0,
        "topoitl_before_action": 0,
    }

    def _tool_calls(text: str) -> List[Dict[str, Any]]:
        calls: List[Dict[str, Any]] = []
        for match in re.finditer(
            r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
            str(text or ""),
            re.DOTALL,
        ):
            try:
                parsed = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                calls.append(parsed)
        return calls

    def _first_tool_name(entry: Dict[str, Any]) -> str:
        for conv in entry.get("conversations", []) or []:
            if conv.get("from") != "gpt":
                continue
            calls = _tool_calls(str(conv.get("value", "")))
            if calls:
                return str(calls[0].get("name") or "")
        return ""

    for entry in data:
        if not entry.get("conversations"):
            issues["empty_conversations"] += 1
            continue
        if not entry.get("system"):
            issues["missing_system"] += 1

        meta = entry.get("metadata", {}) or {}
        if meta.get("design_version") != SFT_DESIGN_VERSION:
            issues["stale_design_version"] += 1
        if meta.get("system_prompt_sha256") != expected_prompt_hash:
            issues["prompt_hash_mismatch"] += 1
        if "<diagnosis>" not in json.dumps(
            entry.get("conversations", []),
            ensure_ascii=False,
        ):
            issues["missing_final_diagnosis"] += 1
        if meta.get("augmentation_type") == "terminal_closure":
            if _first_tool_name(entry) != "get_system_overview":
                issues["terminal_closure_bad_first_tool"] += 1

        for conv in entry["conversations"]:
            if conv.get("from") == "gpt":
                val = conv.get("value", "")
                think_count = len(re.findall(r"<think>", val))
                if think_count > 1:
                    issues["multi_think"] += 1
                action_positions = [
                    pos for pos in (
                        str(val).find("<tool_call>"),
                        str(val).find("<diagnosis>"),
                    )
                    if pos >= 0
                ]
                label_positions = [
                    pos for pos in (
                        str(val).find("<topoitl_state>"),
                        str(val).find("<topoitl_action>"),
                    )
                    if pos >= 0
                ]
                action_pos = min(action_positions) if action_positions else -1
                label_pos = min(label_positions) if label_positions else -1
                # Action-first layout: the executable block must come BEFORE the
                # trailing TopoITL state/action labels. Flag the inverted
                # (label-before-action) layout as a defect.
                if label_pos >= 0 and action_pos >= 0 and label_pos < action_pos:
                    issues["topoitl_before_action"] += 1

    return issues


def load_sft_dataset(data_path: str, max_samples: Optional[int] = None):
    """
    Load the SFT dataset from JSONL file with quality validation.

    Returns a HuggingFace Dataset object (or list of dicts if HF not available).
    """
    data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
                if max_samples and len(data) >= max_samples:
                    break

    logger.info(f"Loaded {len(data)} SFT training examples from {data_path}")

    # Quality validation
    issues = validate_sft_data(data)
    for issue_name, count in issues.items():
        if count > 0:
            logger.warning(f"  Data quality issue: {issue_name} = {count}")
    fatal_issues = {
        "empty_conversations",
        "missing_system",
        "missing_final_diagnosis",
        "stale_design_version",
        "prompt_hash_mismatch",
        "terminal_closure_bad_first_tool",
        "topoitl_before_action",
    }
    fatal_counts = {
        name: count
        for name, count in issues.items()
        if name in fatal_issues and count > 0
    }
    if fatal_counts:
        raise ValueError(
            "SFT data failed fatal quality gates; regenerate the dataset with "
            f"the current formatter before training. Issues: {fatal_counts}"
        )
    if all(v == 0 for v in issues.values()):
        logger.info("  Data quality checks passed")

    try:
        from datasets import Dataset
        return Dataset.from_list(data)
    except ImportError:
        logger.warning("HuggingFace datasets not available; returning raw list")
        return data


def split_train_eval(dataset, eval_ratio: float = 0.1, seed: int = 42):
    """
    Split dataset into train and eval subsets.

    Args:
        dataset: HuggingFace Dataset or list.
        eval_ratio: Fraction of data to use for evaluation.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    try:
        from datasets import Dataset
        if isinstance(dataset, Dataset):
            splits = dataset.train_test_split(test_size=eval_ratio, seed=seed)
            train_ds = splits["train"]
            eval_ds = splits["test"]
            logger.info(
                f"Split dataset: {len(train_ds)} train, {len(eval_ds)} eval "
                f"(ratio={eval_ratio})"
            )
            return train_ds, eval_ds
    except ImportError:
        pass

    # Fallback for list-based datasets
    import random
    rng = random.Random(seed)
    data = list(dataset)
    rng.shuffle(data)
    split_idx = int(len(data) * (1 - eval_ratio))
    logger.info(
        f"Split dataset: {split_idx} train, {len(data) - split_idx} eval "
        f"(ratio={eval_ratio})"
    )
    return data[:split_idx], data[split_idx:]


def limit_eval_dataset(dataset, max_samples: Optional[int] = None):
    """Limit the held-out loss-eval set for fast gate runs."""
    if not max_samples or max_samples <= 0:
        return dataset
    current_size = len(dataset)
    if current_size <= max_samples:
        return dataset

    capped_size = int(max_samples)
    logger.info(
        f"Limiting eval_loss dataset: {current_size} -> {capped_size} samples"
    )
    if hasattr(dataset, "select"):
        return dataset.select(range(capped_size))
    return list(dataset)[:capped_size]


def save_dataset_jsonl(dataset, output_path: str) -> str:
    """Persist a raw dataset split as JSONL for diagnostic evaluation."""
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            f.write(json.dumps(dataset[i], ensure_ascii=False) + "\n")
    return output_path


def _log_tokenized_length_stats(
    train_tokenized,
    eval_tokenized,
    max_seq_length: int,
) -> None:
    """Log sequence-length guardrails for SFT runs.

    The trainer truncates silently when a row exceeds ``max_seq_length``.  That
    is especially damaging here because the final diagnosis often appears near
    the end of a trajectory.  Logging exact-max rows gives us a cheap warning
    that the model may be seeing search actions without terminal decisions.
    """

    def _lengths(dataset) -> List[int]:
        lengths: List[int] = []
        for i in range(len(dataset)):
            row = dataset[i]
            lengths.append(len(row.get("input_ids", [])))
        return lengths

    def _summary(name: str, lengths: List[int]) -> None:
        if not lengths:
            logger.info("%s token lengths: no rows", name)
            return
        sorted_lengths = sorted(lengths)
        p50 = sorted_lengths[int((len(sorted_lengths) - 1) * 0.50)]
        p90 = sorted_lengths[int((len(sorted_lengths) - 1) * 0.90)]
        p95 = sorted_lengths[int((len(sorted_lengths) - 1) * 0.95)]
        p99 = sorted_lengths[int((len(sorted_lengths) - 1) * 0.99)]
        max_len = sorted_lengths[-1]
        exact_max = sum(1 for n in lengths if n >= max_seq_length)
        logger.info(
            "%s token lengths: p50=%s p90=%s p95=%s p99=%s max=%s exact_max=%s/%s",
            name,
            p50,
            p90,
            p95,
            p99,
            max_len,
            exact_max,
            len(lengths),
        )
        if exact_max:
            logger.warning(
                "%s has %s rows at max_seq_length=%s; review SFT truncation risk.",
                name,
                exact_max,
                max_seq_length,
            )

    _summary("train", _lengths(train_tokenized))
    _summary("eval", _lengths(eval_tokenized))


# ============================================================================
# Tokenization
# ============================================================================

def build_tokenize_fn(
    tokenizer,
    max_seq_length: int,
    strip_auxiliary_topoitl: bool = False,
    topoitl_weights: Optional[Dict[str, float]] = None,
):
    """
    Build a tokenization function for ShareGPT conversations with TopoITL
    three-term weighted supervision (eq:topoitl_loss).

    In addition to assistant-only loss masking, every trainable token is tagged
    with a per-token ``loss_weights`` channel implementing

        L = Σ_t ( λ_z·ℓ^z_t + λ_act·ℓ^act_t + λ_tr·ℓ^tr_t )

    via the serialized ``state → action → executable`` layout:

    * ``<topoitl_state>`` block of turn t is the diagnosis-state target z_t
      (weight ``state`` = λ_z). For every turn after the first, that same state
      block is *also* the transition target z_{(t-1)+1} of the previous step,
      because the policy input for turn t already contains the previous
      observation o_{t-1}; those tokens additionally receive ``transition`` =
      λ_tr, so their effective weight is λ_z + λ_tr (an exact, non-double-counted
      realization of ℓ^z + ℓ^tr on the shared tokens).
    * ``<topoitl_action>`` block and the concrete ``<tool_call>``/``<diagnosis>``
      block form the structured action ã_t = (u_t, a_t) (weight ``action`` =
      λ_act).
    * All other assistant prose (``<think>`` rationale) receives ``base``.

    Setting every weight to 1.0 recovers standard uniform-CE behavior cloning.
    Ablations: ``state=0, transition=0`` ⇒ *w/o state supervision*;
    ``base=0, action small`` emphasizes pure state/transition supervision, etc.
    """
    weights = {
        "state": 1.0,
        "action": 1.0,
        "transition": 0.5,
        "base": 1.0,
    }
    if topoitl_weights:
        weights.update({k: float(v) for k, v in topoitl_weights.items() if k in weights})

    use_fast = bool(getattr(tokenizer, "is_fast", False))

    # Pre-compute role identifier token IDs for robust loss masking
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    assistant_role_ids = tokenizer.encode("assistant\n", add_special_tokens=False)

    _state_re = re.compile(r"<topoitl_state>.*?</topoitl_state>", re.DOTALL)
    _action_re = re.compile(r"<topoitl_action>.*?</topoitl_action>", re.DOTALL)
    _exec_re = re.compile(
        r"<tool_call>.*?</tool_call>|<diagnosis>.*?</diagnosis>", re.DOTALL
    )

    def _spans(pattern, text):
        return [(m.start(), m.end()) for m in pattern.finditer(text)]

    def _weight_for_offset(start, end, state_spans, action_spans, exec_spans,
                           first_state_end):
        """Return (weight, is_state_block) for a token's char span."""
        mid = (start + end) / 2.0 if end > start else start
        for s, e in state_spans:
            if s <= mid < e:
                # state block: λ_z, plus λ_tr for non-initial blocks
                if first_state_end is not None and s >= first_state_end:
                    return weights["state"] + weights["transition"]
                return weights["state"]
        for s, e in action_spans:
            if s <= mid < e:
                return weights["action"]
        for s, e in exec_spans:
            if s <= mid < e:
                return weights["action"]
        return weights["base"]

    def tokenize_sharegpt(examples):
        conversations = examples.get("conversations", [])
        system_prompt = examples.get("system", "")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        for turn in conversations:
            role_map = {"human": "user", "gpt": "assistant", "observation": "tool"}
            role = role_map.get(turn["from"], turn["from"])
            content = turn.get("value", "")
            if role == "assistant" and strip_auxiliary_topoitl:
                content = strip_topoitl_labels(str(content))
            messages.append({"role": role, "content": content})

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )

        if use_fast:
            tokenized = tokenizer(
                text,
                truncation=True,
                max_length=max_seq_length,
                return_offsets_mapping=True,
                return_tensors=None,
            )
            offsets = tokenized.get("offset_mapping", None)
        else:
            tokenized = tokenizer(
                text, truncation=True, max_length=max_seq_length, return_tensors=None,
            )
            offsets = None

        input_ids = tokenized["input_ids"]

        # Char spans for weighting (only meaningful with a fast tokenizer).
        if offsets is not None and not strip_auxiliary_topoitl:
            state_spans = _spans(_state_re, text)
            action_spans = _spans(_action_re, text)
            exec_spans = _spans(_exec_re, text)
            first_state_end = state_spans[0][1] if state_spans else None
        else:
            state_spans = action_spans = exec_spans = []
            first_state_end = None

        labels = [-100] * len(input_ids)
        loss_weights = [0.0] * len(input_ids)
        role_len = len(assistant_role_ids)

        in_assistant = False
        i = 0
        while i < len(input_ids):
            if input_ids[i] == im_start_id:
                after = input_ids[i + 1: i + 1 + role_len]
                if after == assistant_role_ids:
                    in_assistant = True
                    i += 1 + role_len
                    continue
                in_assistant = False
            elif input_ids[i] == im_end_id:
                if in_assistant:
                    labels[i] = input_ids[i]
                    loss_weights[i] = weights["base"]
                in_assistant = False
            elif in_assistant:
                labels[i] = input_ids[i]
                if offsets is not None:
                    start, end = offsets[i]
                    loss_weights[i] = _weight_for_offset(
                        start, end, state_spans, action_spans, exec_spans,
                        first_state_end,
                    )
                else:
                    loss_weights[i] = weights["base"]
            i += 1

        result = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": tokenized["attention_mask"],
            "loss_weights": loss_weights,
        }
        return result

    return tokenize_sharegpt


# ============================================================================
# Training Loop
# ============================================================================

# ============================================================================
# Weighted TopoITL loss: collator + trainer
# ============================================================================

def build_weighted_collator(tokenizer, base_collator):
    """Wrap a HF collator so it also pads the per-token ``loss_weights`` field.

    The base ``DataCollatorForSeq2Seq`` drops unknown keys, so we strip
    ``loss_weights`` before delegating, then right/left-pad it to match the
    collated ``labels`` length and re-attach it as a float tensor.
    """
    import torch

    pad_side = getattr(tokenizer, "padding_side", "right")

    def collate(features):
        weights = [list(f.get("loss_weights", [])) for f in features]
        stripped = [
            {k: v for k, v in f.items() if k != "loss_weights"}
            for f in features
        ]
        batch = base_collator(stripped)
        target_len = batch["labels"].shape[1]
        padded = []
        for w in weights:
            if len(w) < target_len:
                pad = [0.0] * (target_len - len(w))
                w = (pad + w) if pad_side == "left" else (w + pad)
            else:
                w = w[:target_len]
            padded.append(w)
        batch["loss_weights"] = torch.tensor(padded, dtype=torch.float32)
        return batch

    return collate


def make_weighted_loss_trainer(base_trainer_cls):
    """Return a Trainer subclass that applies per-token TopoITL loss weights."""
    import torch

    class WeightedLossTrainer(base_trainer_cls):
        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            loss_weights = inputs.pop("loss_weights", None)
            labels = inputs.get("labels")
            outputs = model(**inputs)
            logits = outputs.logits

            # Shift for next-token prediction.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            vocab = shift_logits.size(-1)
            flat_logits = shift_logits.view(-1, vocab)
            flat_labels = shift_labels.view(-1)
            token_loss = torch.nn.functional.cross_entropy(
                flat_logits, flat_labels, ignore_index=-100, reduction="none",
            )

            if loss_weights is not None:
                shift_weights = loss_weights[..., 1:].contiguous().view(-1).to(token_loss.dtype)
            else:
                shift_weights = torch.ones_like(token_loss)

            # Only count supervised (non-ignored) tokens.
            valid = (flat_labels != -100).to(token_loss.dtype)
            w = shift_weights * valid
            denom = w.sum().clamp_min(1.0)
            loss = (token_loss * w).sum() / denom

            return (loss, outputs) if return_outputs else loss

    return WeightedLossTrainer


def run_sft_training(config: Dict[str, Any]) -> str:
    """
    Run the SFT training loop with real eval-split validation.

    This function requires GPU and HuggingFace libraries. It is designed to
    run on the cloud server with the RTX PRO 6000 96GB.

    Args:
        config: SFT training configuration dict.

    Returns:
        Path to the best checkpoint directory.
    """
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
    )
    from peft import LoraConfig, PeftModel, get_peft_model, TaskType

    output_dir = ensure_dir(config["output_dir"])
    dist = _distributed_info()
    is_distributed = bool(dist["enabled"])
    is_main_process = bool(dist["is_main"])
    if is_distributed and torch.cuda.is_available():
        torch.cuda.set_device(int(dist["local_rank"]))
        logger.info(
            "Distributed SFT enabled: rank=%s local_rank=%s world_size=%s",
            dist["rank"], dist["local_rank"], dist["world_size"],
        )

    # ---- Load tokenizer ----
    logger.info(f"Loading tokenizer: {config['model_name']}")
    tokenizer = cached_from_pretrained(
        AutoTokenizer,
        config["model_name"],
        logger=logger,
        description="SFT tokenizer",
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load model ----
    logger.info(f"Loading model: {config['model_name']}")
    model_kwargs = {
        "torch_dtype": torch.bfloat16 if config.get("bf16") else torch.float16,
        "trust_remote_code": True,
    }
    if not is_distributed:
        model_kwargs["device_map"] = "auto"

    attn_impl = config.get("attn_implementation")
    if attn_impl:
        try:
            logger.info(f"Attempting to load model with attn_implementation={attn_impl}")
            model = cached_from_pretrained(
                AutoModelForCausalLM,
                config["model_name"],
                logger=logger,
                description="SFT base model",
                **model_kwargs,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            logger.warning(
                f"Failed to load model with attn_implementation={attn_impl}: {e}. "
                "Falling back to default attention implementation."
            )
            model = cached_from_pretrained(
                AutoModelForCausalLM,
                config["model_name"],
                logger=logger,
                description="SFT base model",
                **model_kwargs,
            )
    else:
        model = cached_from_pretrained(
            AutoModelForCausalLM,
            config["model_name"],
            logger=logger,
            description="SFT base model",
            **model_kwargs,
        )

    # ---- Apply or resume LoRA ----
    adapter_path = config.get("adapter_name_or_path")
    if adapter_path:
        logger.info(f"Loading trainable LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(
            model,
            adapter_path,
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["target_modules"],
            bias="none",
        )
        model = get_peft_model(model, lora_config)
    if is_main_process:
        model.print_trainable_parameters()

    # Enable input gradients for gradient checkpointing + LoRA compatibility
    if config.get("gradient_checkpointing", False):
        model.enable_input_require_grads()

    # ---- Load and split dataset ----
    dataset = load_sft_dataset(config["data_path"])
    eval_ratio = config.get("eval_split_ratio", 0.1)
    train_dataset, full_eval_dataset = split_train_eval(dataset, eval_ratio)
    eval_dataset = limit_eval_dataset(
        full_eval_dataset,
        config.get("eval_max_samples"),
    )
    diag_eval_source_dataset = limit_eval_dataset(
        full_eval_dataset,
        config.get("diag_eval_source_max_samples"),
    )
    diag_eval_source_path = os.path.join(output_dir, "diag_eval_source.jsonl")
    if is_main_process:
        save_dataset_jsonl(diag_eval_source_dataset, diag_eval_source_path)
        logger.info(
            "Diagnostic eval source saved: %s samples (loss eval uses %s)",
            len(diag_eval_source_dataset),
            len(eval_dataset),
        )

    # ---- Tokenize datasets ----
    strip_topoitl = bool(config.get("strip_topoitl_labels_for_training", False))
    topoitl_weights = config.get("topoitl_loss_weights") or {}
    tokenize_fn = build_tokenize_fn(
        tokenizer,
        config["max_seq_length"],
        strip_auxiliary_topoitl=strip_topoitl,
        topoitl_weights=topoitl_weights,
    )
    if is_main_process:
        logger.info(
            "TopoITL supervision: strip_labels=%s weights=%s",
            strip_topoitl,
            {k: topoitl_weights.get(k) for k in ("state", "action", "transition", "base")},
        )
    logger.info("Tokenizing train dataset...")

    if hasattr(train_dataset, 'map'):
        train_tokenized = train_dataset.map(
            tokenize_fn,
            remove_columns=train_dataset.column_names,
            num_proc=4,
            desc="Tokenizing train",
        )
        eval_tokenized = eval_dataset.map(
            tokenize_fn,
            remove_columns=eval_dataset.column_names,
            num_proc=4,
            desc="Tokenizing eval",
        )
    else:
        train_tokenized = [tokenize_fn(ex) for ex in train_dataset]
        eval_tokenized = [tokenize_fn(ex) for ex in eval_dataset]

    # Log tokenization stats
    if hasattr(train_tokenized, '__len__'):
        logger.info(
            f"Tokenized: {len(train_tokenized)} train, "
            f"{len(eval_tokenized)} eval"
        )
        if is_main_process:
            _log_tokenized_length_stats(
                train_tokenized,
                eval_tokenized,
                int(config["max_seq_length"]),
            )

    # ---- Training arguments ----
    grad_accum = int(config["gradient_accumulation_steps"])
    if is_distributed and config.get("scale_grad_accum_by_world_size", True):
        original_grad_accum = grad_accum
        grad_accum = max(1, grad_accum // int(dist["world_size"]))
        if is_main_process and grad_accum != original_grad_accum:
            logger.info(
                "Adjusted gradient_accumulation_steps from %s to %s "
                "to keep the global SFT batch size close under DDP.",
                original_grad_accum, grad_accum,
            )

    training_arg_kwargs = dict(
        output_dir=output_dir,
        num_train_epochs=config["num_epochs"],
        max_steps=int(config.get("max_steps", -1)),
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=grad_accum,
        learning_rate=config["learning_rate"],
        lr_scheduler_type=config["lr_scheduler_type"],
        warmup_steps=config["warmup_steps"],
        weight_decay=config["weight_decay"],
        bf16=config.get("bf16", True),
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        logging_steps=config["logging_steps"],
        save_steps=config["save_steps"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        load_best_model_at_end=config.get("load_best_model_at_end", True),
        metric_for_best_model=config.get("metric_for_best_model", "eval_loss"),
        greater_is_better=False,  # lower eval_loss is better
        save_total_limit=5,
        report_to="none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=int(config.get("dataloader_num_workers", 0)),
    )
    import inspect
    if "group_by_length" in inspect.signature(TrainingArguments.__init__).parameters:
        training_arg_kwargs["group_by_length"] = config.get("group_by_length", False)
    else:
        if is_main_process and config.get("group_by_length"):
            logger.warning("group_by_length is not supported by the current TrainingArguments library in this environment and will be ignored.")

    if is_distributed:
        training_arg_kwargs["ddp_find_unused_parameters"] = False
        training_arg_kwargs["ddp_timeout"] = int(config.get("ddp_timeout", 7200))
    training_args = TrainingArguments(**training_arg_kwargs)

    # ---- Data collator ----
    base_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )
    # Wrap so the per-token TopoITL loss_weights survive batching.
    use_weighted_loss = not strip_topoitl
    if use_weighted_loss:
        data_collator = build_weighted_collator(tokenizer, base_collator)
    else:
        data_collator = base_collator

    # ---- Diagnostic evaluation callback ----
    from transformers import TrainerCallback

    class DiagnosticEvalCallback(TrainerCallback):
        """Run full Oracle-based diagnostic evaluation every eval_steps."""

        @staticmethod
        def _distributed_barrier() -> None:
            try:
                import torch.distributed as torch_dist

                if torch_dist.is_available() and torch_dist.is_initialized():
                    torch_dist.barrier()
            except Exception as exc:
                logger.debug(f"Diagnostic eval barrier skipped: {exc}")

        def __init__(self, model, tokenizer, eval_steps, output_dir,
                     sft_data_path, diag_eval_episodes=50,
                     best_metric="selection_score",
                     sampling_strategy="stratified",
                     samples_per_type=None,
                     scenario_types=None,
                     sampling_seed=42,
                     oracle_mode="real",
                     include_system_health=False,
                     expose_status_summary=False,
                     episode_timeout_seconds=None,
                     generate_timeout_seconds=None):
            self.model = model
            self.tokenizer = tokenizer
            self.eval_steps = eval_steps
            self.output_dir = output_dir
            self.diag_eval_episodes = diag_eval_episodes
            self.best_metric = best_metric
            self.sampling_strategy = sampling_strategy
            self.samples_per_type = samples_per_type
            self.scenario_types = scenario_types
            self.sampling_seed = sampling_seed
            self.oracle_mode = oracle_mode
            self.include_system_health = include_system_health
            self.expose_status_summary = expose_status_summary
            self.episode_timeout_seconds = episode_timeout_seconds
            self.generate_timeout_seconds = generate_timeout_seconds
            self.best_score = -float("inf")
            self.best_step = 0
            self.best_diag_dir = os.path.join(output_dir, "best_diag")
            self.diag_history = []

            if not bool(_distributed_info()["is_main"]):
                self.enabled = False
                return

            # Prepare test scenarios once
            self.test_data_path = os.path.join(output_dir, "diag_eval_test.jsonl")
            try:
                from src.evaluation.evaluator import prepare_test_scenarios
                prepare_test_scenarios(
                    sft_data_path, self.test_data_path,
                    test_ratio=1.0, seed=42,
                )
                self.enabled = True
                logger.info(
                    f"DiagnosticEvalCallback: prepared test data at "
                    f"{self.test_data_path}"
                )
            except Exception as e:
                logger.warning(f"DiagnosticEvalCallback disabled: {e}")
                self.enabled = False

        def on_evaluate(self, args, state, control, **kwargs):
            """Called after each HF Trainer eval step."""
            if not state.is_world_process_zero:
                self._distributed_barrier()
                self._distributed_barrier()
                return
            if not self.enabled:
                self._distributed_barrier()
                self._distributed_barrier()
                return

            step = state.global_step
            logger.info(
                f"\n=== Diagnostic Eval at step {step} "
                f"({self.diag_eval_episodes} episodes) ==="
            )
            self._distributed_barrier()

            try:
                from src.evaluation.evaluator import (
                    run_diagnostic_eval_with_model,
                )

                episodes_dir = os.path.join(
                    self.output_dir, "diag_episodes"
                )
                result = run_diagnostic_eval_with_model(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    test_scenarios_path=self.test_data_path,
                    max_episodes=self.diag_eval_episodes,
                    max_steps=config.get("diag_eval_max_steps", 15),
                    max_new_tokens=config.get("diag_eval_max_new_tokens", 256),
                    output_dir=episodes_dir,
                    step=step,
                    episode_timeout_seconds=self.episode_timeout_seconds,
                    generate_timeout_seconds=self.generate_timeout_seconds,
                    sampling_strategy=self.sampling_strategy,
                    sampling_seed=self.sampling_seed,
                    samples_per_type=self.samples_per_type,
                    scenario_types=self.scenario_types,
                    oracle_mode=self.oracle_mode,
                    include_system_health=self.include_system_health,
                    expose_status_summary=self.expose_status_summary,
                )

                metrics = result.get("overall", {})
                self.diag_history.append({
                    "step": step,
                    **metrics,
                })

                logger.info(f"  Diagnostic metrics at step {step}:")
                for k, v in metrics.items():
                    logger.info(f"    {k}: {_fmt_metric(v)}")

                # Save incremental history
                save_json(
                    self.diag_history,
                    os.path.join(self.output_dir, "diag_eval_history.json"),
                )

                score = _metric_value(metrics, self.best_metric, 0.0)
                if score > self.best_score:
                    self.best_score = score
                    self.best_step = step
                    self.model.save_pretrained(self.best_diag_dir)
                    self.tokenizer.save_pretrained(self.best_diag_dir)
                    save_json(
                        {
                            "step": step,
                            "metric": self.best_metric,
                            "score": score,
                            "metrics": metrics,
                        },
                        os.path.join(self.best_diag_dir, "diagnostic_best.json"),
                    )
                    logger.info(
                        f"  New diagnostic best at step {step}: "
                        f"{self.best_metric}={score:.4f}"
                    )

            except Exception as e:
                logger.warning(
                    f"Diagnostic eval failed at step {step}: "
                    f"{type(e).__name__}: {e}"
                )
            finally:
                self._distributed_barrier()

    diag_callback = None
    callbacks = []
    diagnostic_eval_mode = str(
        config.get("diagnostic_eval_mode", "trainer_callback") or "trainer_callback"
    ).lower()
    enable_trainer_diag_eval = (
        bool(config.get("run_diagnostic_eval", False))
        and diagnostic_eval_mode in {"trainer_callback", "in_trainer", "inline"}
    )
    if (
        enable_trainer_diag_eval
        and int(config.get("diag_eval_episodes", 0)) > 0
    ):
        diag_callback = DiagnosticEvalCallback(
            model=model,
            tokenizer=tokenizer,
            eval_steps=config["eval_steps"],
            output_dir=output_dir,
            sft_data_path=diag_eval_source_path,
            diag_eval_episodes=config.get("diag_eval_episodes", 50),
            best_metric=config.get("diagnostic_best_metric", "selection_score"),
            sampling_strategy=config.get("diag_eval_sampling", "stratified"),
            samples_per_type=config.get("diag_eval_samples_per_type"),
            scenario_types=config.get("diag_eval_scenario_types"),
            sampling_seed=config.get("diag_eval_seed", 42),
            oracle_mode=config.get("diag_eval_oracle_mode", "real"),
            include_system_health=bool(
                config.get("diag_eval_include_system_health", False)
            ),
            expose_status_summary=bool(
                config.get("diag_eval_expose_status_summary", False)
            ),
            episode_timeout_seconds=config.get(
                "diag_eval_episode_timeout_seconds"
            ),
            generate_timeout_seconds=config.get(
                "diag_eval_generate_timeout_seconds"
            ),
        )
        callbacks.append(diag_callback)
    else:
        if bool(config.get("run_diagnostic_eval", False)):
            logger.info(
                "In-training diagnostic evaluation configured for external "
                "checkpoint sentinels (diagnostic_eval_mode=%s); Trainer "
                "checkpoint selection uses eval_loss.",
                diagnostic_eval_mode,
            )
        else:
            logger.info(
                "In-training diagnostic evaluation disabled; "
                "checkpoint selection uses eval_loss."
            )

    # ---- Train ----
    trainer_cls = make_weighted_loss_trainer(Trainer) if use_weighted_loss else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    logger.info("Starting SFT training...")
    logger.info(
        f"  Train samples: {len(train_tokenized)}, "
        f"Eval samples: {len(eval_tokenized)}"
    )
    logger.info(
        f"  Eval every {config['eval_steps']} steps, "
        f"best model by {config.get('metric_for_best_model', 'eval_loss')}"
    )

    start_time = time.time()
    trainer.train()
    elapsed = time.time() - start_time

    if not trainer.is_world_process_zero():
        return os.path.join(output_dir, "best")

    # Save final state. With HF load_best_model_at_end this is the best model
    # according to metric_for_best_model, usually eval_loss.
    final_dir = os.path.join(output_dir, "final")
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save the RL starting checkpoint. Prefer the Oracle diagnostic best
    # checkpoint because lower eval_loss has historically not guaranteed
    # better diagnostic accuracy.
    best_dir = os.path.join(output_dir, "best")
    selected_best_source = "eval_loss"
    if (
        config.get("select_best_by_diagnostic", True)
        and diag_callback is not None
        and os.path.exists(os.path.join(diag_callback.best_diag_dir, "adapter_config.json"))
    ):
        if os.path.exists(best_dir):
            shutil.rmtree(best_dir)
        shutil.copytree(diag_callback.best_diag_dir, best_dir)
        selected_best_source = "diagnostic"
        logger.info(
            f"Selected diagnostic best checkpoint from step "
            f"{diag_callback.best_step} for {best_dir}"
        )
    else:
        model.save_pretrained(best_dir)
        tokenizer.save_pretrained(best_dir)

    logger.info(
        f"SFT training complete in {elapsed/3600:.1f}h. "
        f"Best model saved to {best_dir} ({selected_best_source})"
    )

    # Save training summary
    summary = {
        "model_name": config["model_name"],
        "adapter_name_or_path": config.get("adapter_name_or_path"),
        "training_time_seconds": elapsed,
        "total_steps": trainer.state.global_step,
        "total_epochs": config["num_epochs"],
        "train_samples": len(train_tokenized),
        "eval_samples": len(eval_tokenized),
        "best_checkpoint": best_dir,
        "best_checkpoint_selection": selected_best_source,
        "diagnostic_best_checkpoint": (
            diag_callback.best_diag_dir if diag_callback is not None else None
        ),
        "diagnostic_best_step": (
            diag_callback.best_step if diag_callback is not None else 0
        ),
        "diagnostic_best_metric": config.get("diagnostic_best_metric", "selection_score"),
        "diagnostic_best_score": (
            diag_callback.best_score if diag_callback is not None else None
        ),
        "data_path": config["data_path"],
        "data_mtime": (
            os.path.getmtime(config["data_path"])
            if os.path.exists(config["data_path"]) else None
        ),
        "data_size_bytes": (
            os.path.getsize(config["data_path"])
            if os.path.exists(config["data_path"]) else None
        ),
        "diagnostic_eval_source": diag_eval_source_path,
        "final_checkpoint": final_dir,
        "max_seq_length": config["max_seq_length"],
        "eval_steps": config["eval_steps"],
        "config": {k: v for k, v in config.items() if k != "target_modules"},
    }

    # Add eval history
    eval_logs = [
        log for log in trainer.state.log_history
        if "eval_loss" in log
    ]
    if eval_logs:
        summary["eval_history"] = eval_logs
        best_eval = min(eval_logs, key=lambda x: x["eval_loss"])
        summary["best_eval_loss"] = best_eval["eval_loss"]
        summary["best_eval_step"] = best_eval.get("step", 0)
        logger.info(
            f"  Best eval_loss: {best_eval['eval_loss']:.4f} "
            f"at step {best_eval.get('step', '?')}"
        )

    # Add diagnostic eval history from callback
    if diag_callback is not None and diag_callback.diag_history:
        summary["diagnostic_eval_history"] = diag_callback.diag_history
        # Find best by the configured diagnostic selection metric.
        best_metric_name = config.get("diagnostic_best_metric", "selection_score")
        best_diag = max(
            diag_callback.diag_history,
            key=lambda x: _metric_value(x, best_metric_name, 0.0),
        )
        summary["best_diagnostic_step"] = best_diag.get("step", 0)
        summary["best_aggregate_score"] = best_diag.get("aggregate_score", 0)
        summary["best_selection_score"] = best_diag.get("selection_score", 0)
        logger.info(
            f"  Best {best_metric_name}: "
            f"{_fmt_metric(best_diag.get(best_metric_name, 0))} "
            f"at step {best_diag.get('step', '?')}"
        )

    save_json(summary, os.path.join(output_dir, "training_summary.json"))

    # ---- Post-training diagnostic evaluation ----
    if not (
        bool(config.get("run_post_train_eval", False))
        and int(config.get("post_train_eval_episodes", 0)) > 0
    ):
        logger.info(
            "Post-training diagnostic evaluation disabled; "
            "the final evaluation script will save full trajectories."
        )
        return best_dir

    logger.info("\n=== Running post-training diagnostic evaluation ===")
    try:
        from src.evaluation.evaluator import (
            run_model_evaluation,
            prepare_test_scenarios,
        )

        # Prepare test scenarios from the held-out SFT split.
        test_data_path = os.path.join(output_dir, "eval_test.jsonl")
        sft_data_path = diag_eval_source_path
        prepare_test_scenarios(
            sft_data_path, test_data_path, test_ratio=1.0, seed=42,
        )

        # Evaluate the best model
        eval_output_dir = os.path.join(output_dir, "evaluation")
        ensure_dir(eval_output_dir)

        eval_result = run_model_evaluation(
            model_name="sft_best",
            model_path=best_dir,
            test_scenarios_path=test_data_path,
            output_dir=eval_output_dir,
            max_steps=config.get("diag_eval_max_steps", 15),
            max_episodes=config.get("post_train_eval_episodes", 64),
            max_new_tokens=config.get("post_train_eval_max_new_tokens", 256),
            episode_timeout_seconds=config.get(
                "post_train_eval_episode_timeout_seconds",
                config.get("diag_eval_episode_timeout_seconds"),
            ),
            generate_timeout_seconds=config.get(
                "post_train_eval_generate_timeout_seconds",
                config.get("diag_eval_generate_timeout_seconds"),
            ),
            sampling_strategy=config.get("diag_eval_sampling", "stratified"),
            sampling_seed=config.get("diag_eval_seed", 42),
            samples_per_type=(
                config.get("post_train_eval_samples_per_type")
                or config.get("diag_eval_samples_per_type")
            ),
            scenario_types=config.get("diag_eval_scenario_types"),
            oracle_mode=config.get("diag_eval_oracle_mode", "real"),
            include_system_health=bool(
                config.get("diag_eval_include_system_health", False)
            ),
            expose_status_summary=bool(
                config.get("diag_eval_expose_status_summary", False)
            ),
        )

        # Append diagnostic metrics to summary
        overall_metrics = (
            eval_result.get("overall_metrics")
            or eval_result.get("overall")
        )
        if eval_result and overall_metrics:
            summary["diagnostic_metrics"] = overall_metrics
            logger.info("Post-training diagnostic metrics:")
            for metric, value in overall_metrics.items():
                logger.info(f"  {metric}: {_fmt_metric(value)}")

            # Per-type breakdown
            by_type = (
                eval_result.get("metrics_by_scenario_type")
                or eval_result.get("by_type")
            )
            if by_type:
                summary["diagnostic_metrics_by_type"] = by_type
                logger.info("By scenario type:")
                for stype, metrics in by_type.items():
                    da = _fmt_metric(metrics.get("diagnostic_accuracy", 0))
                    se = _fmt_metric(metrics.get("search_efficiency", 0))
                    logger.info(f"  {stype}: DA={da}  SE={se}")

            # Save updated summary with diagnostic metrics
            save_json(summary, os.path.join(output_dir, "training_summary.json"))
            logger.info(
                f"Updated training_summary.json with diagnostic metrics"
            )

    except Exception as e:
        logger.warning(
            f"Post-training evaluation failed ({type(e).__name__}: {e}). "
            f"Run 'python scripts/07_evaluate.py --models sft' manually."
        )

    return best_dir
