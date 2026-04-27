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
import shutil
import time
from typing import Any, Dict, List, Optional

from src.utils.io_utils import load_yaml, save_json, ensure_dir, setup_logger

logger = setup_logger(__name__)


# ============================================================================
# Configuration Defaults
# ============================================================================

DEFAULT_SFT_CONFIG = {
    "model_name": "Qwen/Qwen2.5-7B-Instruct",
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
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 32,
    "learning_rate": 2e-5,
    "lr_scheduler_type": "cosine",
    "warmup_steps": 100,
    "max_seq_length": 16384,
    "weight_decay": 0.01,
    "bf16": True,
    "gradient_checkpointing": True,

    # Evaluation
    "eval_split_ratio": 0.1,
    "eval_steps": 200,
    "save_steps": 200,
    "logging_steps": 10,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "select_best_by_diagnostic": True,
    "diagnostic_best_metric": "aggregate_score",
    "diag_eval_episodes": 64,
    "diag_eval_max_steps": 15,
    "diag_eval_sampling": "stratified",
    "diag_eval_seed": 42,

    # Dataset
    "dataset_format": "sharegpt",  # "sharegpt" or "messages"
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
    issues = {"multi_think": 0, "empty_conversations": 0, "missing_system": 0}

    for entry in data:
        if not entry.get("conversations"):
            issues["empty_conversations"] += 1
            continue
        if not entry.get("system"):
            issues["missing_system"] += 1

        for conv in entry["conversations"]:
            if conv.get("from") == "gpt":
                val = conv.get("value", "")
                think_count = len(re.findall(r"<think>", val))
                if think_count > 1:
                    issues["multi_think"] += 1

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
    if all(v == 0 for v in issues.values()):
        logger.info("  Data quality checks passed ✓")

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


def save_dataset_jsonl(dataset, output_path: str) -> str:
    """Persist a raw dataset split as JSONL for diagnostic evaluation."""
    ensure_dir(os.path.dirname(output_path))
    with open(output_path, "w", encoding="utf-8") as f:
        for i in range(len(dataset)):
            f.write(json.dumps(dataset[i], ensure_ascii=False) + "\n")
    return output_path


# ============================================================================
# Tokenization
# ============================================================================

def build_tokenize_fn(tokenizer, max_seq_length: int):
    """
    Build a tokenization function for ShareGPT format conversations.

    Applies Qwen chat template and masks non-assistant tokens with -100
    so that loss is only computed on the model's own generation.
    """

    # Pre-compute role identifier token IDs for robust loss masking
    im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # Encode role identifier tokens ("assistant\n" following <|im_start|>)
    assistant_role_ids = tokenizer.encode("assistant\n", add_special_tokens=False)

    def tokenize_sharegpt(examples):
        """Tokenize ShareGPT format conversations with assistant-only loss."""
        conversations = examples.get("conversations", [])
        system_prompt = examples.get("system", "")

        # Build the full conversation using Qwen-compatible roles
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        for turn in conversations:
            role_map = {"human": "user", "gpt": "assistant", "observation": "tool"}
            role = role_map.get(turn["from"], turn["from"])
            msg = {"role": role, "content": turn.get("value", "")}
            messages.append(msg)

        # Tokenize with chat template
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=max_seq_length,
            return_tensors=None,
        )

        input_ids = tokenized["input_ids"]

        # Create labels: mask non-assistant tokens with -100
        # For Qwen2.5, assistant content is between:
        #   <|im_start|>assistant\n ... <|im_end|>
        # We use token-level scanning for robustness (no string decode needed)
        labels = [-100] * len(input_ids)
        role_len = len(assistant_role_ids)

        in_assistant = False
        i = 0
        while i < len(input_ids):
            if input_ids[i] == im_start_id:
                # Check if the tokens after im_start match "assistant\n"
                after = input_ids[i + 1: i + 1 + role_len]
                if after == assistant_role_ids:
                    in_assistant = True
                    i += 1 + role_len  # Skip <|im_start|>assistant\n
                    continue
                else:
                    in_assistant = False
            elif input_ids[i] == im_end_id:
                if in_assistant:
                    labels[i] = input_ids[i]  # Include the end token
                in_assistant = False
            elif in_assistant:
                labels[i] = input_ids[i]
            i += 1

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": tokenized["attention_mask"],
        }

    return tokenize_sharegpt


# ============================================================================
# Training Loop
# ============================================================================

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
    from peft import LoraConfig, get_peft_model, TaskType

    output_dir = ensure_dir(config["output_dir"])

    # ---- Load tokenizer ----
    logger.info(f"Loading tokenizer: {config['model_name']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        trust_remote_code=True,
        padding_side="right",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- Load model ----
    logger.info(f"Loading model: {config['model_name']}")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16 if config.get("bf16") else torch.float16,
        trust_remote_code=True,
        device_map="auto",
    )

    # ---- Apply LoRA ----
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        lora_dropout=config["lora_dropout"],
        target_modules=config["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Enable input gradients for gradient checkpointing + LoRA compatibility
    if config.get("gradient_checkpointing", False):
        model.enable_input_require_grads()

    # ---- Load and split dataset ----
    dataset = load_sft_dataset(config["data_path"])
    eval_ratio = config.get("eval_split_ratio", 0.1)
    train_dataset, eval_dataset = split_train_eval(dataset, eval_ratio)
    diag_eval_source_path = save_dataset_jsonl(
        eval_dataset,
        os.path.join(output_dir, "diag_eval_source.jsonl"),
    )

    # ---- Tokenize datasets ----
    tokenize_fn = build_tokenize_fn(tokenizer, config["max_seq_length"])
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

    # ---- Training arguments ----
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_train_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
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
    )

    # ---- Data collator ----
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

    # ---- Diagnostic evaluation callback ----
    from transformers import TrainerCallback

    class DiagnosticEvalCallback(TrainerCallback):
        """Run full Oracle-based diagnostic evaluation every eval_steps."""

        def __init__(self, model, tokenizer, eval_steps, output_dir,
                     sft_data_path, diag_eval_episodes=50,
                     best_metric="aggregate_score",
                     sampling_strategy="stratified",
                     sampling_seed=42):
            self.model = model
            self.tokenizer = tokenizer
            self.eval_steps = eval_steps
            self.output_dir = output_dir
            self.diag_eval_episodes = diag_eval_episodes
            self.best_metric = best_metric
            self.sampling_strategy = sampling_strategy
            self.sampling_seed = sampling_seed
            self.best_score = -float("inf")
            self.best_step = 0
            self.best_diag_dir = os.path.join(output_dir, "best_diag")
            self.diag_history = []

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
            if not self.enabled:
                return

            step = state.global_step
            logger.info(
                f"\n=== Diagnostic Eval at step {step} "
                f"({self.diag_eval_episodes} episodes) ==="
            )

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
                    output_dir=episodes_dir,
                    step=step,
                    sampling_strategy=self.sampling_strategy,
                    sampling_seed=self.sampling_seed,
                )

                metrics = result.get("overall", {})
                self.diag_history.append({
                    "step": step,
                    **metrics,
                })

                logger.info(f"  Diagnostic metrics at step {step}:")
                for k, v in metrics.items():
                    logger.info(f"    {k}: {v:.4f}")

                # Save incremental history
                save_json(
                    self.diag_history,
                    os.path.join(self.output_dir, "diag_eval_history.json"),
                )

                score = metrics.get(self.best_metric, 0)
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

    diag_callback = DiagnosticEvalCallback(
        model=model,
        tokenizer=tokenizer,
        eval_steps=config["eval_steps"],
        output_dir=output_dir,
        sft_data_path=diag_eval_source_path,
        diag_eval_episodes=config.get("diag_eval_episodes", 50),
        best_metric=config.get("diagnostic_best_metric", "aggregate_score"),
        sampling_strategy=config.get("diag_eval_sampling", "stratified"),
        sampling_seed=config.get("diag_eval_seed", 42),
    )

    # ---- Train ----
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=eval_tokenized,
        data_collator=data_collator,
        callbacks=[diag_callback],
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
        "training_time_seconds": elapsed,
        "total_steps": trainer.state.global_step,
        "total_epochs": config["num_epochs"],
        "train_samples": len(train_tokenized),
        "eval_samples": len(eval_tokenized),
        "best_checkpoint": best_dir,
        "best_checkpoint_selection": selected_best_source,
        "diagnostic_best_checkpoint": diag_callback.best_diag_dir,
        "diagnostic_best_step": diag_callback.best_step,
        "diagnostic_best_metric": config.get("diagnostic_best_metric", "aggregate_score"),
        "diagnostic_best_score": diag_callback.best_score,
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
    if diag_callback.diag_history:
        summary["diagnostic_eval_history"] = diag_callback.diag_history
        # Find best by aggregate_score
        best_diag = max(
            diag_callback.diag_history,
            key=lambda x: x.get("aggregate_score", 0),
        )
        summary["best_diagnostic_step"] = best_diag.get("step", 0)
        summary["best_aggregate_score"] = best_diag.get("aggregate_score", 0)
        logger.info(
            f"  Best aggregate_score: {best_diag.get('aggregate_score', 0):.4f} "
            f"at step {best_diag.get('step', '?')}"
        )

    save_json(summary, os.path.join(output_dir, "training_summary.json"))

    # ---- Post-training diagnostic evaluation ----
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
            max_steps=15,
            max_episodes=200,
            sampling_strategy=config.get("diag_eval_sampling", "stratified"),
            sampling_seed=config.get("diag_eval_seed", 42),
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
                logger.info(f"  {metric}: {value:.4f}")

            # Per-type breakdown
            by_type = (
                eval_result.get("metrics_by_scenario_type")
                or eval_result.get("by_type")
            )
            if by_type:
                summary["diagnostic_metrics_by_type"] = by_type
                logger.info("By scenario type:")
                for stype, metrics in by_type.items():
                    da = metrics.get("diagnostic_accuracy", 0)
                    se = metrics.get("search_efficiency", 0)
                    logger.info(f"  {stype}: DA={da:.2f}  SE={se:.2f}")

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
