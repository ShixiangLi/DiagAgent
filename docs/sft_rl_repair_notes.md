# SFT/RL Repair Notes

## Oracle Baseline False Positives

No-fault scenarios are explicitly backed by clean baseline CSV files. If the
LightGBM Oracle emits a fault on these records, the benchmark should treat it
as an Oracle false positive rather than a hidden fault. The repair therefore
applies `clean_baseline_false_positive_guard` inside `PredictionToolExecutor`
at runtime instead of silently rewriting only SFT data.

This is more suitable for paper-grade evaluation because SFT generation, RL
rollouts, and evaluation now share the same calibrated Oracle behavior. The
raw Oracle status, fault type, confidence, and probabilities are preserved in
`raw_*` fields for auditability.

Scope:

- Applied only when scenario metadata indicates no-fault ground truth and the
  source file is a clean baseline/fault-free file.
- Also applied to system-level anomaly scores so no-fault prompts do not begin
  from spurious alert systems.
- Not used for ordinary faulted scenarios, where Oracle predictions remain
  available as real evidence.

## Tool-Coverage Targets

`get_node_status_summary` should appear in:

- ambiguous single-system scenarios,
- ambiguous cross-system scenarios,
- ordinary single-system/cross-system scenarios before detailed probing,
- no-fault scenarios to avoid full blind sweeps,
- low-confidence scenarios before targeted `diagnose_node`.

`get_related_systems` should appear in:

- cross-system and ambiguous cross-system scenarios,
- cases where downstream symptoms may originate from an upstream plant,
- transitions between plant, AHU, FCU, PFPU, and SFPU systems.

`get_node_sensors` should appear in:

- low-confidence `Warning` cases,
- sensor-bias or marginal-fault cases where the model must justify a final
  diagnosis with sensor-level evidence.

The SFT readiness gate now requires non-zero coverage for all three tools.

## SFT Sweeps vs RL Efficiency

Some off-path probing is useful in SFT because the agent must learn how to
eliminate healthy nodes and recover from wrong initial hypotheses. However,
long blind sweeps are harmful as a default policy: they reduce search
efficiency, consume rollout budget, and cause no-diagnosis failures when the
root node is not reached before `max_steps`.

The repaired trajectory generator keeps a small amount of off-path verification
but first asks for a system status summary. This gives SFT a better initial
policy: overview -> summary/related-system reasoning -> targeted diagnosis.

RL can optimize efficiency, but only after SFT can already finish a diagnosis.
The reward therefore keeps accuracy dominant and gates efficiency by accuracy.
The efficiency weight is increased enough to prefer shorter correct paths, but
fast wrong or unsupported diagnoses still do not profit.

## Required Cloud Sequence

```bash
pip install -r requirements.txt
python scripts/03_generate_sft_data.py --n-total 3000 --validate
python scripts/_full_audit.py
python scripts/04_generate_rl_data.py
python scripts/local_training_readiness.py --require-oracle-runtime
python scripts/05_train_sft.py
python scripts/_rl_preflight.py
python scripts/06_train_rl.py
python scripts/07_evaluate.py --models base,sft,rl --test-size 300
```

Do not start RL until regenerated SFT data passes the tool-coverage gates and
SFT produces `outputs/sft/best` from diagnostic-best checkpoint selection.

