# Cloud Training Readiness

This repo is prepared for cloud SFT and RL training. Local work is limited to
code, data, topology, Oracle artifacts, and validation. Full SFT/RL training
should run on the cloud server.

## Current Gate Status

- Topology artifact: 624 nodes, 671 edges, 10 cross-system edges.
- Oracle artifacts: 85 LightGBM models, including 8 system Oracles.
- System Oracle quality: min weighted F1 about 0.623, mean weighted F1 about 0.884.
- SFT data: 3,000 ShareGPT trajectories, no-fault ground truth normalized.
- RL data: train/val/test = 5,259/657/658 prompts.
- RL curriculum:
  - Phase 1 excludes no-fault.
  - Phase 2 excludes no-fault.
  - Phase 3 uses the capped effective pool, currently 119 no-fault prompts out of 4,375.
- SFT best checkpoint selection:
  - `outputs/sft/best_diag` is saved whenever Oracle diagnostic aggregate improves.
  - `outputs/sft/best` is promoted from the diagnostic-best checkpoint.

## Regenerate Data After Code Sync

The current pipeline uses runtime Oracle calibration and additional tool-use
trajectories. Regenerate SFT/RL data after syncing this version:

```bash
pip install -r requirements.txt
python scripts/03_generate_sft_data.py --n-total 3000 --validate
python scripts/_full_audit.py
python scripts/04_generate_rl_data.py
```

Expected data properties:

- SFT contains `get_node_status_summary`, `get_related_systems`, and
  `get_node_sensors` trajectories.
- No-fault trajectories use clean baseline CSVs and runtime
  `clean_baseline_false_positive_guard` calibration when needed.
- Low-confidence scenarios preserve their `a_lc_` / `na_lc_` identity during
  evaluation, so the Oracle returns `Warning` instead of being mismatched to a
  standard fault scenario.

## Cloud Preflight

Run after syncing the repo and artifacts to the cloud:

```bash
pip install -r requirements.txt
python scripts/local_training_readiness.py --require-oracle-runtime
```

Expected result before SFT:

- `READY WITH WARNINGS` is acceptable only if the remaining warnings are:
  - weak per-node fallback models,
  - SFT audit categories,
  - missing historical SFT/RL histories,
  - missing `outputs/sft/best`.
- Any topology, Oracle runtime, no-fault cleanliness, reward, or curriculum failure must be fixed before training.

## SFT Training

```bash
python scripts/05_train_sft.py
```

After SFT completes, verify:

```bash
test -f outputs/sft/best/adapter_config.json
test -f outputs/sft/best/adapter_model.safetensors
test -f outputs/sft/best_diag/diagnostic_best.json
python scripts/local_training_readiness.py --require-oracle-runtime
```

SFT acceptance:

- `outputs/sft/best` exists and is selected by Oracle diagnostic `aggregate_score`.
- Diagnostic evaluation uses the held-out SFT split, `diag_eval_episodes >= 100`,
  and `diag_eval_max_steps >= 15`.
- Diagnostic accuracy should meet or exceed the prior reference: DA >= 0.64.
- Tool coverage should show non-zero use of `get_node_status_summary`,
  `get_related_systems`, and `get_node_sensors`.
- If SFT DA is below 0.60, inspect `outputs/sft/diag_episodes` before starting RL.

## RL Preflight

Run only after SFT has produced `outputs/sft/best`:

```bash
python scripts/_rl_preflight.py
```

Expected result:

- SFT checkpoint: PASS.
- RL data: PASS.
- Curriculum: PASS.
- Reward functions: PASS.
- Oracle env: PASS.

## RL Training

```bash
python scripts/06_train_rl.py
```

RL acceptance:

- RL must outperform SFT on the same held-out evaluation set.
- Primary target: diagnostic accuracy clearly above SFT DA.
- Secondary targets: better search efficiency and no regression in tool format validity.
- Inspect `outputs/rl/rl_episodes` if DA drops across eval steps or if no-fault predictions rise.

## Final Evaluation

```bash
python scripts/07_evaluate.py --models base,sft,rl
```

Do not accept mock evaluation. Logs must show real Oracle environment loading and real tool execution.
