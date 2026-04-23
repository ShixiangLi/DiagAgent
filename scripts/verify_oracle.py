"""Verify Oracle model training results and compatibility with path-guided design."""
import json
import os
import glob

# 1. Check training summary
print("=" * 60)
print("1. TRAINING SUMMARY")
print("=" * 60)

summary_path = "outputs/models/training_summary.json"
if os.path.exists(summary_path):
    summary = json.load(open(summary_path, "r"))
    for sys_id, res in summary.items():
        print(f"  {sys_id:20s}  trained={res['n_trained']}  skipped={res['n_skipped']}  classes={len(res['label_map'])}")
else:
    print("  training_summary.json not found!")

# 2. Check all oracle model files exist
print("\n" + "=" * 60)
print("2. ORACLE MODEL FILES")
print("=" * 60)

systems = ["chiller_plant", "boiler_plant", "sdahu", "ddahu", "rtu", "fcu", "pfpu", "sfpu"]
all_ok = True
for sys_id in systems:
    oracle_dir = os.path.join("outputs/models", sys_id, f"{sys_id}__oracle")
    model_path = os.path.join(oracle_dir, "model.txt")
    meta_path = os.path.join(oracle_dir, "metadata.json")
    map_path = os.path.join(oracle_dir, "fault_node_map.json")

    has_model = os.path.exists(model_path)
    has_meta = os.path.exists(meta_path)
    has_map = os.path.exists(map_path)

    status = "OK" if (has_model and has_meta and has_map) else "MISSING"
    if status == "MISSING":
        all_ok = False

    # Read accuracy from metadata
    acc = "N/A"
    f1 = "N/A"
    n_classes = "N/A"
    n_features = "N/A"
    if has_meta:
        meta = json.load(open(meta_path, "r"))
        acc = f"{meta.get('accuracy', 0):.4f}"
        f1 = f"{meta.get('weighted_f1', 0):.4f}"
        n_classes = meta.get("n_classes", "?")
        n_features = meta.get("n_features", "?")

    print(f"  {sys_id:20s}  [{status:7s}]  acc={acc}  f1={f1}  classes={n_classes}  features={n_features}")

print(f"\n  All oracles present: {'YES' if all_ok else 'NO'}")

# 3. Check fault_node_map compatibility
print("\n" + "=" * 60)
print("3. FAULT-NODE MAP VERIFICATION")
print("=" * 60)

for sys_id in systems:
    map_path = os.path.join("outputs/models", sys_id, f"{sys_id}__oracle", "fault_node_map.json")
    if os.path.exists(map_path):
        fmap = json.load(open(map_path, "r"))
        n_faults = sum(1 for v in fmap.values() if v != "none")
        print(f"  {sys_id:20s}  {len(fmap)} entries ({n_faults} fault types mapped)")
    else:
        print(f"  {sys_id:20s}  MISSING")

# 4. Verify path-guided design compatibility
print("\n" + "=" * 60)
print("4. PATH-GUIDED DESIGN COMPATIBILITY")
print("=" * 60)

import sys
sys.path.insert(0, ".")
from src.topology.topology_builder import TopologyBuilder
from src.environment.diagnostic_path import DiagnosticPathGenerator

builder = TopologyBuilder("configs/topology_config.yaml", "data/lbnl")
builder.build()
pg = DiagnosticPathGenerator("configs/diagnostic_paths.yaml")

# Test path generation for each system's first fault type
test_cases = [
    ("coolingtower_fouling_080", "chiller_plant::Cooling_Tower_1", "chiller_plant", "sdahu"),
    ("boiler_bias_-2", "boiler_plant::Boiler_1", "boiler_plant", "sdahu"),
    ("coi_stuck_075", "sdahu::Cooling_Coil", "sdahu", None),
    ("sa_bias_2", "sdahu::AHU", "sdahu", None),
]

for fault_type, root_node, root_sys, ds_sys in test_cases:
    path = pg.generate_path(fault_type, root_node, root_sys, builder, ds_sys)
    nodes_str = " -> ".join([f"{n.node_id}({n.role[0]})" for n in path.nodes])
    print(f"  {fault_type:35s}  len={path.path_length}  {nodes_str}")

# 5. Quick end-to-end test: can we generate a trajectory?
print("\n" + "=" * 60)
print("5. END-TO-END TRAJECTORY TEST")
print("=" * 60)

from src.node_models.model_registry import ModelRegistry
from src.node_models.prediction_tools import PredictionToolExecutor
from src.topology.topology_tools import TopologyToolExecutor
from src.environment.tool_executor import UnifiedToolExecutor
from src.environment.fault_scenario import generate_single_system_scenarios
from src.data_gen.trajectory_generator import generate_trajectory
import random

registry = ModelRegistry("outputs/models")
topo_exec = TopologyToolExecutor(builder)
pred_exec = PredictionToolExecutor(registry, builder)
tool_exec = UnifiedToolExecutor(topo_exec, pred_exec)

scenarios = generate_single_system_scenarios(
    "data/lbnl", "chiller_plant", "Chiller Plant", builder,
    path_generator=pg, n_windows_per_fault=1,
)

# Pick a fault scenario
fault_scenarios = [s for s in scenarios if s.fault_type != "Normal"]
if fault_scenarios:
    test_s = fault_scenarios[0]
    traj = generate_trajectory(test_s, tool_exec, random.Random(42))
    n_tc = sum(1 for s in traj.steps if s.tool_call is not None)
    statuses = []
    for s in traj.steps:
        if s.role == "tool" and "status" in s.content:
            import re
            m = re.search(r'"status": "(\w+)"', s.content)
            if m:
                statuses.append(m.group(1))
    print(f"  Scenario: {test_s.scenario_id}")
    print(f"  Fault: {test_s.fault_type}")
    print(f"  Path: {test_s.diagnostic_path.path_length if test_s.diagnostic_path else 0} nodes")
    print(f"  Tool calls: {n_tc}")
    print(f"  Status sequence: {' -> '.join(statuses)}")
    print(f"  Result: PASS")
else:
    print("  No fault scenarios found!")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("=" * 60)
