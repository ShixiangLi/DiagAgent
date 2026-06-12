"""
Deep quality validation script for DiagAgent SFT dataset.
Runs strict schema, syntax, tag consistency, and logical routing checks.
"""

import argparse
import json
import os
import re
import sys
from collections import Counter

def setup_logger():
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    return logging.getLogger("validate_sft")

logger = setup_logger()

def load_topology(topology_path):
    if not os.path.exists(topology_path):
        logger.warning(f"Topology JSON not found at {topology_path}. Skipping node validity checks.")
        return None
    try:
        with open(topology_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        nodes = {node["node_id"]: node for node in data.get("nodes", [])}
        logger.info(f"Loaded topology with {len(nodes)} nodes from {topology_path}")
        return nodes
    except Exception as e:
        logger.error(f"Failed to load topology: {e}")
        return None

def validate_gpt_value(val_str, scenario_id, turn_idx, topology_nodes):
    errors = []
    
    # 1. Check <think> tags
    think_matches = re.findall(r"<think>(.*?)</think>", val_str, re.DOTALL)
    if len(think_matches) != 1:
        errors.append(f"Turn {turn_idx}: Expected exactly 1 <think> block, found {len(think_matches)}")
    
    # 2. Check Action tags (must have either <tool_call> or <diagnosis>, but not both)
    has_tool_call = "<tool_call>" in val_str
    has_diagnosis = "<diagnosis>" in val_str
    if has_tool_call and has_diagnosis:
        errors.append(f"Turn {turn_idx}: Cannot have both <tool_call> and <diagnosis> in one turn")
    elif not has_tool_call and not has_diagnosis:
        errors.append(f"Turn {turn_idx}: Must contain either <tool_call> or <diagnosis>")
        
    # 3. Check TopoITL state and action tags
    topoitl_state_matches = re.findall(r"<topoitl_state>(.*?)</topoitl_state>", val_str, re.DOTALL)
    topoitl_action_matches = re.findall(r"<topoitl_action>(.*?)</topoitl_action>", val_str, re.DOTALL)
    
    if len(topoitl_state_matches) != 1:
        errors.append(f"Turn {turn_idx}: Expected exactly 1 <topoitl_state> tag, found {len(topoitl_state_matches)}")
    if len(topoitl_action_matches) != 1:
        errors.append(f"Turn {turn_idx}: Expected exactly 1 <topoitl_action> tag, found {len(topoitl_action_matches)}")
        
    # 4. Check JSON syntax and content matching
    parsed_tool_call = None
    parsed_diagnosis = None
    
    if has_tool_call:
        tc_match = re.search(r"<tool_call>\s*(\{.*?\}|\s*)\s*</tool_call>", val_str, re.DOTALL)
        if not tc_match:
            errors.append(f"Turn {turn_idx}: Malformed <tool_call> tag structure")
        else:
            try:
                parsed_tool_call = json.loads(tc_match.group(1))
            except json.JSONDecodeError as e:
                errors.append(f"Turn {turn_idx}: Failed to parse <tool_call> JSON: {e}")
                
    if has_diagnosis:
        diag_match = re.search(r"<diagnosis>\s*(\{.*?\}|\s*)\s*</diagnosis>", val_str, re.DOTALL)
        if not diag_match:
            errors.append(f"Turn {turn_idx}: Malformed <diagnosis> tag structure")
        else:
            try:
                parsed_diagnosis = json.loads(diag_match.group(1))
            except json.JSONDecodeError as e:
                errors.append(f"Turn {turn_idx}: Failed to parse <diagnosis> JSON: {e}")
                
    parsed_state = None
    if len(topoitl_state_matches) == 1:
        try:
            parsed_state = json.loads(topoitl_state_matches[0])
        except json.JSONDecodeError as e:
            errors.append(f"Turn {turn_idx}: Failed to parse <topoitl_state> JSON: {e}")
            
    parsed_action = None
    if len(topoitl_action_matches) == 1:
        try:
            parsed_action = json.loads(topoitl_action_matches[0])
        except json.JSONDecodeError as e:
            errors.append(f"Turn {turn_idx}: Failed to parse <topoitl_action> JSON: {e}")

    # 5. Check if action and tool_call / diagnosis parameters align
    if parsed_action and parsed_tool_call:
        action_args = parsed_action.get("arguments", {})
        tool_args = parsed_tool_call.get("arguments", {})
        action_tool = parsed_action.get("tool")
        tool_name = parsed_tool_call.get("name")
        
        if action_tool != tool_name:
            errors.append(f"Turn {turn_idx}: Tool name mismatch! topoitl_action tool='{action_tool}', tool_call name='{tool_name}'")
            
        # Check node_id or system_id arguments matching
        for key in ["node_id", "system_id"]:
            if key in tool_args or key in action_args:
                if tool_args.get(key) != action_args.get(key):
                    errors.append(f"Turn {turn_idx}: Argument mismatch for key '{key}'! topoitl_action={action_args.get(key)}, tool_call={tool_args.get(key)}")
                    
        # Node validation check
        node_id = tool_args.get("node_id")
        if node_id and topology_nodes:
            if node_id not in topology_nodes and not node_id.startswith("system::"):
                errors.append(f"Turn {turn_idx}: Hallucinated node_id '{node_id}' not found in topology!")

    if parsed_action and parsed_diagnosis:
        action_type = parsed_action.get("type")
        action_primitive = parsed_action.get("primitive")
        if action_type != "final_diagnosis" or action_primitive != "conclude":
            errors.append(f"Turn {turn_idx}: Diagnosis topoitl_action mismatch! Expected type='final_diagnosis', primitive='conclude', got type='{action_type}', primitive='{action_primitive}'")
            
        action_diag = parsed_action.get("diagnosis", {})
        for key in ["root_cause_node", "fault_type", "status"]:
            if action_diag.get(key) != parsed_diagnosis.get(key):
                errors.append(f"Turn {turn_idx}: Diagnosis field '{key}' mismatch! topoitl_action={action_diag.get(key)}, diagnosis={parsed_diagnosis.get(key)}")

    # 6. Verify action-first tag ordering:
    #    <think> -> Action(tool_call/diagnosis) -> <topoitl_state> -> <topoitl_action>
    # The executable block is emitted FIRST so multi-turn rollouts/eval stop at
    # </tool_call> | </diagnosis>; the TopoITL state/action labels are trailing
    # training-only supervision (eq:topoitl_loss three-term loss on those tokens).
    think_start = val_str.find("<think>")
    action_start = val_str.find("<tool_call>") if has_tool_call else val_str.find("<diagnosis>")
    state_start = val_str.find("<topoitl_state>")
    act_tag_start = val_str.find("<topoitl_action>")

    # Require: executable block BEFORE state label BEFORE action label.
    if not (action_start >= 0 and state_start >= 0 and act_tag_start >= 0
            and action_start < state_start < act_tag_start):
        errors.append(
            f"Turn {turn_idx}: Invalid tag ordering! Expected "
            f"<think> -> Action -> <topoitl_state> -> <topoitl_action> "
            f"(got exec={action_start}, state={state_start}, "
            f"action_label={act_tag_start})"
        )

    # 7. Check there is no SECOND executable block (one action per turn).
    #    Trailing TopoITL labels after the first executable block are expected.
    if has_tool_call and val_str.count("<tool_call>") > 1:
        errors.append(f"Turn {turn_idx}: More than one <tool_call> block")
    if has_diagnosis and val_str.count("<diagnosis>") > 1:
        errors.append(f"Turn {turn_idx}: More than one <diagnosis> block")

    return errors, parsed_diagnosis

def validate_sft_file(sft_path, topology_path):
    topology_nodes = load_topology(topology_path)
    
    total_records = 0
    records_with_errors = 0
    all_errors = []
    
    with open(sft_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            total_records += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                all_errors.append(f"Line {line_idx}: Failed to parse JSON record: {e}")
                records_with_errors += 1
                continue
                
            scenario_id = record.get("id", f"line_{line_idx}")
            metadata = record.get("metadata", {})
            gt = metadata.get("ground_truth", {})
            conversations = record.get("conversations", [])
            
            if not conversations:
                all_errors.append(f"Record {scenario_id}: Empty conversations array")
                records_with_errors += 1
                continue
                
            record_errors = []
            final_diagnosis = None
            
            # Track discovered nodes to check for logical step skips
            discovered_nodes = set()
            
            for turn_idx, turn in enumerate(conversations):
                from_role = turn.get("from")
                val_str = turn.get("value", "")
                
                if from_role == "human":
                    if not val_str.strip():
                        record_errors.append(f"Turn {turn_idx} (human): Empty prompt")
                elif from_role == "observation":
                    if not val_str.strip():
                        record_errors.append(f"Turn {turn_idx} (observation): Empty observation content")
                    else:
                        try:
                            obs_data = json.loads(val_str)
                            if isinstance(obs_data, dict):
                                # 1. get_node_children children
                                for child in obs_data.get("children", []):
                                    if isinstance(child, dict) and "node_id" in child:
                                        discovered_nodes.add(child["node_id"])
                                # 2. get_system_overview systems
                                for sys_item in obs_data.get("systems", []):
                                    if isinstance(sys_item, dict) and "node_id" in sys_item:
                                        discovered_nodes.add(sys_item["node_id"])
                                # 3. get_upstream_nodes upstream
                                for up in obs_data.get("upstream", []):
                                    if isinstance(up, dict) and "node_id" in up:
                                        discovered_nodes.add(up["node_id"])
                                # 4. get_downstream_nodes downstream
                                for down in obs_data.get("downstream", []):
                                    if isinstance(down, dict) and "node_id" in down:
                                        discovered_nodes.add(down["node_id"])
                                # 5. get_related_systems connections
                                for conn in obs_data.get("upstream_connections", []) + obs_data.get("downstream_connections", []):
                                    if isinstance(conn, dict):
                                        if "connected_via" in conn and conn["connected_via"]:
                                            discovered_nodes.add(conn["connected_via"])
                                        if "target_component" in conn and conn["target_component"]:
                                            discovered_nodes.add(conn["target_component"])
                        except json.JSONDecodeError:
                            pass
                elif from_role == "gpt":
                    gpt_errs, parsed_diag = validate_gpt_value(val_str, scenario_id, turn_idx, topology_nodes)
                    record_errors.extend(gpt_errs)
                    if parsed_diag:
                        final_diagnosis = parsed_diag
                        
                    # Check for logical step-skips (calling tool on node that hasn't been discovered yet)
                    tc_match = re.search(r"<tool_call>\s*(\{.*?\}|\s*)\s*</tool_call>", val_str, re.DOTALL)
                    if tc_match:
                        try:
                            tc_json = json.loads(tc_match.group(1))
                            tool_name = tc_json.get("name")
                            tool_args = tc_json.get("arguments", {})
                            node_id = tool_args.get("node_id")
                            if node_id and tool_name in ["get_node_children", "get_downstream_nodes", "get_upstream_nodes", "diagnose_node", "get_node_sensors"]:
                                if node_id not in discovered_nodes and not node_id.startswith("system::"):
                                    # Relax parent systems check
                                    record_errors.append(f"Turn {turn_idx}: Logical Step Skip! Querying node '{node_id}' before discovering it via get_node_children.")
                        except Exception:
                            pass
            
            # Check if final diagnosis matches GT
            if final_diagnosis and gt:
                gt_node = gt.get("root_cause_node")
                gt_fault = gt.get("fault_type")
                
                diag_node = final_diagnosis.get("root_cause_node")
                diag_fault = final_diagnosis.get("fault_type")
                
                # Normal/No fault mapping comparison
                if gt_node == "none" or not gt_node:
                    if diag_node != "none" or diag_fault != "no_fault":
                        record_errors.append(f"Final diagnosis mismatch: GT is no-fault, but diagnosed root={diag_node}, fault={diag_fault}")
                else:
                    if diag_node != gt_node:
                        record_errors.append(f"Final diagnosis mismatch: GT root cause is '{gt_node}', but diagnosed '{diag_node}'")
                    if diag_fault != gt_fault:
                        record_errors.append(f"Final diagnosis mismatch: GT fault type is '{gt_fault}', but diagnosed '{diag_fault}'")

            if record_errors:
                records_with_errors += 1
                all_errors.extend([f"Record {scenario_id} ({line_idx}): {err}" for err in record_errors])
                
    logger.info("=== SFT Data Validation Summary ===")
    logger.info(f"Total scenarios validated: {total_records}")
    logger.info(f"Clean scenarios: {total_records - records_with_errors}")
    logger.info(f"Scenarios with errors: {records_with_errors}")
    
    if all_errors:
        logger.error(f"Found {len(all_errors)} issues in SFT dataset:")
        for err in all_errors[:50]:  # Show first 50 errors
            logger.error(f"  - {err}")
        if len(all_errors) > 50:
            logger.error(f"  ... and {len(all_errors) - 50} more issues.")
        return False
    else:
        logger.info("✓ Strict SFT data quality verification PASSED! No issues found.")
        return True

def main():
    parser = argparse.ArgumentParser(description="Strictly validate generated SFT data quality")
    parser.add_argument("--sft-path", default="outputs/data/sft_train.jsonl")
    parser.add_argument("--topology-path", default="outputs/topology/topology.json")
    args = parser.parse_args()
    
    if not os.path.exists(args.sft_path):
        logger.error(f"SFT data file not found at {args.sft_path}")
        sys.exit(1)
        
    success = validate_sft_file(args.sft_path, args.topology_path)
    if not success:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()
