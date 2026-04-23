"""
Oracle Environment — Gymnasium-style environment for agent diagnostic episodes.

Wraps the topology tools and prediction models into a step-based environment
where the agent can:
  1. Observe: See the conversation history and current state
  2. Act: Generate text that may contain tool calls
  3. Receive: Tool execution results
  4. Terminate: Issue a final diagnosis or exhaust max steps
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.environment.fault_scenario import FaultScenario, FaultScenarioState
from src.topology.topology_tools import TopologyToolExecutor
from src.node_models.prediction_tools import PredictionToolExecutor
from src.utils.io_utils import setup_logger

logger = setup_logger(__name__)

# Maximum number of tool calls per episode
DEFAULT_MAX_STEPS = 15


@dataclass
class EpisodeStep:
    """A single step in a diagnostic episode."""
    step_num: int
    agent_text: str                      # Full agent output (thought + tool call)
    thought: str                         # Extracted reasoning/thought
    tool_call: Optional[Dict] = None     # Parsed tool call (name + arguments)
    tool_result: Optional[Dict] = None   # Tool execution result
    is_final_diagnosis: bool = False     # Whether this step contains a final diagnosis
    final_diagnosis: Optional[Dict] = None  # Extracted diagnosis if is_final_diagnosis


@dataclass
class EpisodeResult:
    """Complete result of a diagnostic episode."""
    scenario: FaultScenario
    steps: List[EpisodeStep]
    final_diagnosis: Optional[Dict]
    n_tool_calls: int
    is_correct: bool
    is_complete: bool
    terminated_reason: str              # "diagnosis", "max_steps", "error"


class OracleEnvironment:
    """
    Simulated diagnostic environment using oracle node models.

    Parses agent text output for tool calls, executes them against the
    topology and prediction tools, and returns results.
    """

    def __init__(
        self,
        topology_tool_executor: TopologyToolExecutor,
        prediction_tool_executor: PredictionToolExecutor,
        max_steps: int = DEFAULT_MAX_STEPS,
    ):
        self.topo_executor = topology_tool_executor
        self.pred_executor = prediction_tool_executor
        self.max_steps = max_steps

        # Combine tool maps
        self._all_tools = {}
        self._all_tools.update(self.topo_executor._tool_map)
        self._all_tools.update(self.pred_executor._tool_map)

        self._current_scenario: Optional[FaultScenario] = None
        self._steps: List[EpisodeStep] = []
        self._n_tool_calls = 0

    def reset(self, scenario: FaultScenario, scenario_state: FaultScenarioState) -> str:
        """
        Reset the environment for a new episode.

        Args:
            scenario: The fault scenario to diagnose.
            scenario_state: Pre-loaded scenario state with sensor data.

        Returns:
            The initial user message (symptom description).
        """
        self._current_scenario = scenario
        self._steps = []
        self._n_tool_calls = 0
        self.pred_executor.set_scenario_state(scenario_state)
        return scenario.description

    def step(self, agent_text: str) -> Tuple[Optional[Dict], bool, Dict]:
        """
        Process one agent action.

        Args:
            agent_text: The agent's text output, potentially containing
                        tool calls and/or a final diagnosis.

        Returns:
            Tuple of:
              - tool_result: Dict with tool execution result, or None if final diagnosis
              - done: Whether the episode is finished
              - info: Additional info dict
        """
        step_num = len(self._steps) + 1

        # Parse thought and tool call from agent text
        thought = self._extract_thought(agent_text)
        tool_call = self._extract_tool_call(agent_text)
        final_diag = self._extract_final_diagnosis(agent_text)

        step = EpisodeStep(
            step_num=step_num,
            agent_text=agent_text,
            thought=thought,
            tool_call=tool_call,
            is_final_diagnosis=final_diag is not None,
            final_diagnosis=final_diag,
        )

        info = {"step_num": step_num}

        # If final diagnosis detected
        if final_diag is not None:
            step.is_final_diagnosis = True
            self._steps.append(step)
            return None, True, {"terminated_reason": "diagnosis", **info}

        # If tool call detected, execute it
        if tool_call is not None:
            tool_name = tool_call.get("name", "")
            tool_args = tool_call.get("arguments", {})

            # Route to appropriate executor
            if tool_name in self.topo_executor._tool_map:
                result = self.topo_executor.execute(tool_name, tool_args)
            elif tool_name in self.pred_executor._tool_map:
                result = self.pred_executor.execute(tool_name, tool_args)
            else:
                result = {"status": "error", "error": f"Unknown tool: {tool_name}"}

            step.tool_result = result
            self._n_tool_calls += 1
            self._steps.append(step)

            # Check max steps
            if self._n_tool_calls >= self.max_steps:
                return result, True, {"terminated_reason": "max_steps", **info}

            return result, False, info

        # No tool call and no final diagnosis — agent should continue
        self._steps.append(step)
        if step_num >= self.max_steps * 2:  # Safety limit
            return None, True, {"terminated_reason": "max_steps", **info}

        return None, False, info

    def get_episode_result(self) -> EpisodeResult:
        """Build the episode result after completion."""
        final_diag = None
        for step in reversed(self._steps):
            if step.is_final_diagnosis:
                final_diag = step.final_diagnosis
                break

        is_correct = self._check_correctness(final_diag)
        is_complete = self._check_completeness(final_diag)
        terminated = "diagnosis" if final_diag else "max_steps"

        return EpisodeResult(
            scenario=self._current_scenario,
            steps=self._steps,
            final_diagnosis=final_diag,
            n_tool_calls=self._n_tool_calls,
            is_correct=is_correct,
            is_complete=is_complete,
            terminated_reason=terminated,
        )

    def _check_correctness(self, diagnosis: Optional[Dict]) -> bool:
        """Check if the final diagnosis matches ground truth."""
        if diagnosis is None or self._current_scenario is None:
            return False

        scenario = self._current_scenario

        # For no-fault scenarios, correct if agent says "Normal"
        if scenario.fault_type == "Normal":
            diag_status = str(diagnosis.get("status", "")).lower()
            return "normal" in diag_status or "no fault" in diag_status

        # Check root cause node
        root_node = str(diagnosis.get("root_cause_node", "")).lower()
        expected_node = scenario.root_cause_node.lower()

        # Check fault type
        diag_fault = str(diagnosis.get("fault_type", "")).lower()
        expected_fault = scenario.fault_type.lower()

        # Partial match: correct system at minimum
        node_match = expected_node in root_node or root_node in expected_node
        fault_match = expected_fault in diag_fault or diag_fault in expected_fault

        return node_match and fault_match

    def _check_completeness(self, diagnosis: Optional[Dict]) -> bool:
        """Check if the diagnosis includes all required fields."""
        if diagnosis is None:
            return False
        required = ["root_cause_node", "fault_type", "confidence"]
        return all(diagnosis.get(f) is not None for f in required)

    @staticmethod
    def _extract_thought(text: str) -> str:
        """Extract reasoning/thought from agent text."""
        # Pattern: <think>...</think>
        match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        # If no think tags, text before tool call is the thought
        tool_match = re.search(r"<tool_call>", text)
        if tool_match:
            return text[:tool_match.start()].strip()
        return text.strip()

    @staticmethod
    def _extract_tool_call(text: str) -> Optional[Dict]:
        """Extract a tool call from agent text."""
        # Pattern: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
        match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse tool call JSON: {match.group(1)[:100]}")
                return None

        # Alternative pattern: ```tool_call\n{...}\n```
        match = re.search(r"```tool_call\s*\n(\{.*?\})\s*\n```", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None

        return None

    @staticmethod
    def _extract_final_diagnosis(text: str) -> Optional[Dict]:
        """Extract a final diagnosis from agent text."""
        # Pattern: <diagnosis>{"root_cause_node": "...", ...}</diagnosis>
        match = re.search(r"<diagnosis>\s*(\{.*?\})\s*</diagnosis>", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                return None

        # Pattern: **Final Diagnosis** or **Root Cause** followed by structured info
        if any(marker in text.lower() for marker in [
            "final diagnosis", "root cause identified", "diagnosis complete"
        ]):
            # Try to extract structured info from natural language
            diag = {}
            node_match = re.search(r"root.?cause.*?node[:\s]+([^\n,]+)", text, re.IGNORECASE)
            if node_match:
                diag["root_cause_node"] = node_match.group(1).strip()
            fault_match = re.search(r"fault.?type[:\s]+([^\n,]+)", text, re.IGNORECASE)
            if fault_match:
                diag["fault_type"] = fault_match.group(1).strip()
            conf_match = re.search(r"confidence[:\s]+([\d.]+)", text, re.IGNORECASE)
            if conf_match:
                diag["confidence"] = float(conf_match.group(1))

            status_match = re.search(r"status[:\s]+(\w+)", text, re.IGNORECASE)
            if status_match:
                diag["status"] = status_match.group(1).strip()

            if diag:
                return diag

        return None
