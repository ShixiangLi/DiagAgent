"""Validate consistency + quality distribution of generated samples."""
import json, re

contradictions = 0
total_diag_checks = 0
quality = {"correct": 0, "incorrect": 0}
scenario_types = {}

with open('outputs/data/sft_train.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        entry = json.loads(line)
        convs = entry['conversations']
        meta = entry.get('metadata', {})
        q = meta.get('trajectory_quality', 'unknown')
        quality[q] = quality.get(q, 0) + 1
        st = meta.get('scenario_type', 'unknown')
        scenario_types.setdefault(st, {"correct": 0, "incorrect": 0})
        scenario_types[st][q] = scenario_types[st].get(q, 0) + 1

        for i, c in enumerate(convs):
            if c['from'] != 'observation':
                continue
            try:
                result = json.loads(c['value'])
            except:
                continue
            if 'status' not in result or 'fault_type' not in result:
                continue
            status = result['status']
            total_diag_checks += 1
            for j in range(i + 1, len(convs)):
                if convs[j]['from'] == 'gpt':
                    reasoning = convs[j]['value'].lower()
                    if status == 'Normal' and any(kw in reasoning for kw in [
                        'has a fault', 'fault has been detected', 'is reporting']):
                        contradictions += 1
                    if status == 'Fault' and any(kw in reasoning for kw in [
                        'operating normally', 'is healthy', 'no faults were detected',
                        'within normal parameters']):
                        contradictions += 1
                    break

print("=== CONSISTENCY ===")
print(f"Diagnosis checks: {total_diag_checks}")
print(f"Contradictions:   {contradictions}")
print()
print("=== QUALITY DISTRIBUTION ===")
print(f"Correct (model got it right):   {quality.get('correct',0)}")
print(f"Incorrect (model missed/false): {quality.get('incorrect',0)}")
total = quality.get('correct',0) + quality.get('incorrect',0)
if total > 0:
    print(f"Accuracy rate: {quality.get('correct',0)/total:.1%}")
print()
print("=== BY SCENARIO TYPE ===")
for st, counts in sorted(scenario_types.items()):
    t = counts.get('correct',0) + counts.get('incorrect',0)
    print(f"  {st:20s}: {counts.get('correct',0):4d} correct, "
          f"{counts.get('incorrect',0):4d} incorrect "
          f"({counts.get('correct',0)/t:.0%} accuracy)" if t > 0 else "")
print()
print(f"Status: {'PASS' if contradictions == 0 else 'FAIL'}")
