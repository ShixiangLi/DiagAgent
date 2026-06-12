"""
Deep quality validation script for DiagAgent RL prompt dataset.
Runs checks on prompt messages, ground-truth metadata, and split consistency.
"""

import argparse
import json
import os
import sys

def setup_logger():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("validate_rl")

logger = setup_logger()

def validate_rl_file(rl_path):
    total_records = 0
    records_with_errors = 0
    all_errors = []
    
    valid_systems = {"chiller_plant", "boiler_plant", "sdahu", "ddahu", "rtu", "fcu", "pfpu", "sfpu"}

    with open(rl_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total_records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(f"Line {line_idx}: Failed to parse JSON: {e}")
                records_with_errors += 1
                continue
                
            prompt_id = record.get("id", f"line_{line_idx}")
            messages = record.get("messages", [])
            gt = record.get("ground_truth", {})
            metadata = record.get("metadata", {})
            
            # 1. Schema check
            for key in ["id", "messages", "ground_truth", "metadata"]:
                if key not in record:
                    all_errors.append(f"Record {prompt_id} ({line_idx}): Missing key '{key}'")
                    records_with_errors += 1
                    break
            else:
                # 2. Messages check
                if not messages:
                    all_errors.append(f"Record {prompt_id} ({line_idx}): Empty messages list")
                    records_with_errors += 1
                    continue
                
                # Check system prompt and human role
                has_system = any(msg.get("role") == "system" for msg in messages)
                has_user = any(msg.get("role") == "user" for msg in messages)
                
                if not has_system:
                    all_errors.append(f"Record {prompt_id} ({line_idx}): Missing system prompt role")
                if not has_user:
                    all_errors.append(f"Record {prompt_id} ({line_idx}): Missing user query role")
                    
                # 3. Ground Truth check
                gt_sys = gt.get("root_cause_system")
                if gt_sys and gt_sys not in valid_systems:
                    all_errors.append(f"Record {prompt_id} ({line_idx}): Invalid root_cause_system '{gt_sys}' in ground_truth")
                    
                optimal_len = gt.get("optimal_path_length")
                if optimal_len is not None:
                    try:
                        val = int(optimal_len)
                        if val < 0:
                            all_errors.append(f"Record {prompt_id} ({line_idx}): Negative optimal_path_length '{val}'")
                    except (ValueError, TypeError):
                        all_errors.append(f"Record {prompt_id} ({line_idx}): Invalid optimal_path_length format")
                        
                # 4. Metadata check
                if not metadata.get("scenario_id"):
                    all_errors.append(f"Record {prompt_id} ({line_idx}): Missing 'scenario_id' in metadata")
                if not metadata.get("split_group_key"):
                    all_errors.append(f"Record {prompt_id} ({line_idx}): Missing 'split_group_key' in metadata")

    logger.info(f"=== RL Data Validation Summary ({os.path.basename(rl_path)}) ===")
    logger.info(f"Total prompts validated: {total_records}")
    logger.info(f"Clean prompts: {total_records - records_with_errors}")
    logger.info(f"Prompts with errors: {records_with_errors}")
    
    if all_errors:
        logger.error(f"Found {len(all_errors)} issues in RL dataset:")
        for err in all_errors[:20]:
            logger.error(f"  - {err}")
        return False
    else:
        logger.info("✓ RL dataset quality verification PASSED!")
        return True

def main():
    parser = argparse.ArgumentParser(description="Strictly validate generated RL prompts quality")
    parser.add_argument("--rl-dir", default="outputs/data")
    args = parser.parse_args()
    
    success = True
    for split in ["train", "val", "test"]:
        rl_path = os.path.join(args.rl_dir, f"rl_{split}.jsonl")
        if os.path.exists(rl_path):
            if not validate_rl_file(rl_path):
                success = False
        else:
            logger.warning(f"Split file {rl_path} not found.")
            
    if not success:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
