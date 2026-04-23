"""
SFT Formatter — Convert diagnostic trajectories to Qwen2.5 SFT format.

Produces training data in the ShareGPT/Qwen conversational format with:
  - System prompt with tool definitions
  - Multi-turn conversations with <|im_start|>/<|im_end|> tokens
  - Loss masking metadata (only compute loss on assistant turns)
  - Tool call formatting with <tool_call> tags
"""

import json
import os
from typing import Any, Dict, List, Optional

from src.data_gen.trajectory_generator import DiagnosticTrajectory, TrajectoryStep
from src.environment.tool_executor import ALL_TOOL_SCHEMAS
from src.utils.io_utils import save_json, setup_logger

logger = setup_logger(__name__)

# System prompt template
SYSTEM_PROMPT = """You are an expert HVAC fault diagnosis agent for building automation systems. Your role is to systematically diagnose faults in HVAC systems by investigating the building topology, checking component statuses, and tracing fault propagation paths to identify root causes.

## Available Tools
You have access to the following diagnostic tools:

{tool_schemas}

## Diagnostic Protocol
1. Start by getting an overview of the building's HVAC systems.
2. Based on reported symptoms, identify the most likely affected system(s).
3. Drill into the system hierarchy to examine specific components.
4. Use the diagnose_node tool to check component health status.
5. If a fault is found, trace upstream to find the root cause and downstream to assess impact.
6. If no fault is found, investigate alternative components or systems.
7. Provide a final diagnosis with root cause, fault type, and confidence level.

## Response Format
- Use <think>...</think> tags for your reasoning process before each action.
- Use <tool_call>{{"name": "tool_name", "arguments": {{...}}}}</tool_call> to invoke tools.
- When you've identified the root cause, provide your final diagnosis using:
  <diagnosis>{{"root_cause_node": "...", "fault_type": "...", "confidence": 0.XX, "affected_systems": [...]}}</diagnosis>

Always reason step-by-step. Never guess or fabricate information — only use data from tool results."""


def format_system_prompt() -> str:
    """Generate the system prompt with tool schemas."""
    tool_str = json.dumps(ALL_TOOL_SCHEMAS, indent=2)
    return SYSTEM_PROMPT.format(tool_schemas=tool_str)


def trajectory_to_sft_messages(
    trajectory: DiagnosticTrajectory,
) -> List[Dict[str, str]]:
    """
    Convert a DiagnosticTrajectory into a list of chat messages.

    Returns:
        List of dicts with 'role' and 'content' keys, compatible with
        Qwen2.5 chat template.
    """
    messages = [
        {"role": "system", "content": format_system_prompt()},
    ]

    for step in trajectory.steps:
        if step.role == "user":
            messages.append({"role": "user", "content": step.content})
        elif step.role == "assistant":
            messages.append({"role": "assistant", "content": step.content})
        elif step.role == "tool":
            # Tool results are formatted as a special user message or tool role
            # Qwen2.5 uses the "tool" role for tool responses
            messages.append({
                "role": "tool",
                "name": step.tool_name or "unknown",
                "content": step.content,
            })

    return messages


def trajectory_to_sharegpt(
    trajectory: DiagnosticTrajectory,
) -> Dict[str, Any]:
    """
    Convert a trajectory to ShareGPT format for LLaMA-Factory compatibility.

    Merges consecutive gpt turns into a single turn to ensure clean
    turn-taking boundaries (every assistant turn followed by observation
    or end-of-conversation).

    Returns:
        Dict with 'conversations' key containing the formatted dialogue.
    """
    raw_conversations = []

    for step in trajectory.steps:
        if step.role == "user":
            raw_conversations.append({
                "from": "human",
                "value": step.content,
            })
        elif step.role == "assistant":
            raw_conversations.append({
                "from": "gpt",
                "value": step.content,
            })
        elif step.role == "tool":
            raw_conversations.append({
                "from": "observation",
                "value": step.content,
            })

    # Merge consecutive gpt turns into a single turn
    # This prevents the model from learning to generate multi-turn content
    # in a single output (which causes hallucinated tool responses).
    conversations = []
    for conv in raw_conversations:
        if (conversations
                and conversations[-1]["from"] == "gpt"
                and conv["from"] == "gpt"):
            # Merge: append content with a newline separator
            conversations[-1]["value"] += "\n" + conv["value"]
        else:
            conversations.append(conv)

    return {
        "id": trajectory.scenario_id,
        "system": format_system_prompt(),
        "conversations": conversations,
        "metadata": {
            "scenario_id": trajectory.scenario_id,
            "scenario_type": trajectory.scenario_type,
            "ground_truth": trajectory.ground_truth,
            "n_tool_calls": trajectory.metadata.get("n_tool_calls", 0),
            "path_length": trajectory.metadata.get("path_length", 0),
            "difficulty": trajectory.metadata.get("difficulty", "medium"),
            "trajectory_quality": trajectory.metadata.get("trajectory_quality", "unknown"),
            "tool_faithful": trajectory.metadata.get("tool_faithful", True),
        },
    }


def format_sft_dataset(
    trajectories: List[DiagnosticTrajectory],
    output_path: str,
    format_type: str = "sharegpt",
) -> str:
    """
    Format all trajectories into a SFT training dataset file.

    Args:
        trajectories: List of generated trajectories.
        output_path: Path to save the formatted dataset.
        format_type: "sharegpt" for LLaMA-Factory, "messages" for raw messages.

    Returns:
        Path to the saved dataset file.
    """
    dataset = []

    for traj in trajectories:
        if format_type == "sharegpt":
            entry = trajectory_to_sharegpt(traj)
        else:
            entry = {
                "id": traj.scenario_id,
                "messages": trajectory_to_sft_messages(traj),
                "metadata": {
                    "scenario_type": traj.scenario_type,
                    "ground_truth": traj.ground_truth,
                },
            }
        dataset.append(entry)

    # Save as JSONL
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(dataset)} SFT examples to {output_path}")

    # Also save a summary
    summary = {
        "total_examples": len(dataset),
        "format": format_type,
        "scenario_type_counts": {},
        "avg_turns": 0,
        "avg_tool_calls": 0,
    }

    from collections import Counter
    type_counts = Counter()
    total_turns = 0
    total_tools = 0

    for entry in dataset:
        if format_type == "sharegpt":
            meta = entry.get("metadata", {})
            type_counts[meta.get("scenario_type", "unknown")] += 1
            total_turns += len(entry.get("conversations", []))
            total_tools += meta.get("n_tool_calls", 0)
        else:
            meta = entry.get("metadata", {})
            type_counts[meta.get("scenario_type", "unknown")] += 1
            total_turns += len(entry.get("messages", []))

    summary["scenario_type_counts"] = dict(type_counts)
    summary["avg_turns"] = total_turns / max(len(dataset), 1)
    summary["avg_tool_calls"] = total_tools / max(len(dataset), 1)

    summary_path = output_path.replace(".jsonl", "_summary.json")
    save_json(summary, summary_path)

    return output_path
