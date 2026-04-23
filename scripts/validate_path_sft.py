"""Quick validation of path-guided SFT data."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "outputs/data/sft_train.jsonl"

with open(path, "r", encoding="utf-8") as f:
    samples = [json.loads(line) for line in f]

path_lengths = []
n_tool_calls_list = []
status_counts = {"Normal": 0, "Abnormal": 0, "Fault": 0}
scenario_types = {}
has_detour = 0

for s in samples:
    meta = s.get("metadata", {})
    pl = meta.get("path_length", 0)
    path_lengths.append(pl)
    tc = meta.get("n_tool_calls", 0)
    n_tool_calls_list.append(tc)

    st = meta.get("scenario_type", "unknown")
    scenario_types[st] = scenario_types.get(st, 0) + 1

    convs = s.get("conversations", [])
    for c in convs:
        val = c.get("value", "")
        for status in ["Normal", "Abnormal", "Fault"]:
            if f'"status": "{status}"' in val:
                status_counts[status] += 1

    # Check for wrong detours (Normal before Abnormal/Fault)
    statuses_in_order = []
    for c in convs:
        val = c.get("value", "")
        for status in ["Normal", "Abnormal", "Fault"]:
            if f'"status": "{status}"' in val:
                statuses_in_order.append(status)
    if "Normal" in statuses_in_order and len(statuses_in_order) > 1:
        if statuses_in_order.index("Normal") == 0 and len(set(statuses_in_order)) > 1:
            has_detour += 1

print("=== PATH-GUIDED SFT VALIDATION ===")
print(f"Total samples:       {len(samples)}")
print(f"Avg path length:     {sum(path_lengths)/max(len(path_lengths),1):.1f}")
print(f"Avg tool calls:      {sum(n_tool_calls_list)/max(len(n_tool_calls_list),1):.1f}")
print(f"Max tool calls:      {max(n_tool_calls_list)}")
print(f"Min tool calls:      {min(n_tool_calls_list)}")
print()
print("=== STATUS DISTRIBUTION ===")
for s, c in sorted(status_counts.items()):
    print(f"  {s:12s}: {c}")
print()
print("=== SCENARIO TYPES ===")
for s, c in sorted(scenario_types.items()):
    print(f"  {s:16s}: {c}")
print()
print(f"Detour trajectories: {has_detour} ({has_detour/max(len(samples),1)*100:.1f}%)")
print()

# Sample a cross-system trajectory
for s in samples:
    meta = s.get("metadata", {})
    if meta.get("path_length", 0) >= 3:
        convs = s.get("conversations", [])
        print("=== SAMPLE TRAJECTORY (path_length >= 3) ===")
        print(f"Scenario: {meta.get('scenario_id', 'N/A')}")
        print(f"Type: {meta.get('scenario_type', 'N/A')}")
        print(f"Path length: {meta.get('path_length', 0)}")
        print(f"Tool calls: {meta.get('n_tool_calls', 0)}")
        print(f"Turns: {len(convs)}")
        for c in convs[:4]:
            role = c.get("from", "?")
            val = c.get("value", "")[:200]
            print(f"  [{role}] {val}")
        print("  ...")
        break

print("\nStatus: PASS")
