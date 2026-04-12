# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Bug Triage Environment Implementation.

An RL-based environment featuring a 3-phase triage system:
1. Forensic Routing
2. RAG-based Deduplication
3. Live Queue Preemption Management
"""

import os
import json
from uuid import uuid4
from typing import Optional, Dict, Any

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import BugTriageAction, BugTriageObservation
    from .vector_store import BugVectorStore
except ImportError:
    from models import BugTriageAction, BugTriageObservation
    from vector_store import BugVectorStore


class BugTriageEnvironment(Environment):
    """
    Stateful RL environment for simulating an automated bug triage pipeline.
    
    This environment processes a pipeline of bugs through forensic analysis,
    historical duplicate searching via Vector DB, and developer resource allocation.
    """

    # Enable concurrent WebSocket sessions for parallel evaluation.
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self, max_steps: Optional[int] = None):
        """
        Initialize the Bug Triage environment and load datasets.
        
        Args:
            max_steps: Maximum number of bugs to process before ending the episode.
        """
        super().__init__()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.search_attempts = 0

        # Resolve absolute paths for Docker compatibility
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(os.path.dirname(current_dir), "data")

        # Load Environment Data
        with open(os.path.join(data_dir, "historical_bugs.json"), 'r', encoding='utf-8') as f:
            self.historical_dataset = json.load(f)
            
        with open(os.path.join(data_dir, "ongoing_pipeline.json"), 'r', encoding='utf-8') as f:
            self.pipeline_dataset = json.load(f)
            
        with open(os.path.join(data_dir, "developer_queues.json"), 'r', encoding='utf-8') as f:
            self.team_states = json.load(f)

        # Initialize Forensic Tools
        self.vector_store = BugVectorStore(self.historical_dataset)
        
        # Configuration
        self.max_steps = max_steps or len(self.pipeline_dataset)
        self.available_teams = list(set(bug["route_to"] for bug in self.historical_dataset))
        self.available_severities = list(set(bug["severity"] for bug in self.historical_dataset))
        
        self.current_phase = "forensic_routing"
        self.bug_index = 0

    def _get_payload(self) -> Optional[Dict[str, Any]]:
        """Constructs the forensic context for the current target bug."""
        if self.bug_index >= len(self.pipeline_dataset):
            return None
            
        bug = self.pipeline_dataset[self.bug_index]
        payload = bug["forensics"].copy()
        payload.update({
            "description": bug["user_description"],
            "valid_teams": self.available_teams,
            "valid_severities": self.available_severities,
            "current_phase": self.current_phase 
        })
        return payload

    def reset(self) -> BugTriageObservation:
        """
        Reset the environment state for a new triage episode.

        Returns:
            BugTriageObservation initialized for Phase 1.
        """
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self.search_attempts = 0
        self.current_phase = "forensic_routing"
        self.bug_index = 0
        
        return BugTriageObservation(
            message="Phase 1: Forensic Routing started.",
            forensics=self._get_payload(),
            done=False,
            reward=0.0
        )

    def step(self, action: BugTriageAction) -> BugTriageObservation:  # type: ignore[override]
        """
        Execute a triage action across the 3-phase pipeline.

        Args:
            action: The BugTriageAction containing routing, RAG, or queue decisions.

        Returns:
            BugTriageObservation containing the new system state and rewards.
        """
        self._state.step_count += 1
        
        # 0. Handle Agent Hallucinations/Formatting Errors
        if action.phase == "error":
            return BugTriageObservation(
                message="FATAL RL ERROR: Agent generated invalid JSON or hallucinated a schema.",
                forensics=self._get_payload(),
                reward=-5.0, 
                done=True 
            )

        target_bug = self.pipeline_dataset[self.bug_index]
        
        # PHASE 1: FORENSIC ROUTING
        if self.current_phase == "forensic_routing":
            reward = 0.0
            if action.route_to == target_bug["route_to"]: reward += 0.5
            if action.severity == target_bug["severity"]: reward += 0.5

            self.current_phase = "rag_deduplication"
            return BugTriageObservation(
                message=f"Routed to {action.route_to}. Transitioning to Phase 2.",
                forensics=self._get_payload(),
                metadata={"phase": "rag_deduplication", "step": self._state.step_count},
                reward=reward, 
                done=False
            )

        # PHASE 2: RAG DEDUPLICATION
        elif self.current_phase == "rag_deduplication":
            if action.search_query:
                self.search_attempts += 1
                penalty = -0.1 if self.search_attempts <= 1 else -1.0
                results = self.vector_store.search(action.search_query)
                return BugTriageObservation(
                    message="Search results returned. Submit group_id.",
                    forensics=self._get_payload(),
                    search_results=results,
                    reward=penalty, 
                    done=False
                )
            
            elif action.group_id:
                self.search_attempts = 0
                ground_truth = target_bug["duplicate_of"]
                reward = 0.0
                
                if action.group_id == ground_truth:
                    reward += 0.5 if ground_truth == "new_bug" else 2.0
                    msg = "Deduplication Correct."
                else:
                    reward -= 1.5 if action.group_id == "new_bug" else 2.0
                    msg = f"Deduplication Failed. Truth: {ground_truth}."

                if action.group_id != "new_bug" and ground_truth != "new_bug":
                    self.bug_index += 1
                    self.current_phase = "forensic_routing"
                    done = (self.bug_index >= self.max_steps)
                    return BugTriageObservation(
                        message=msg + (" | Dataset exhausted." if done else " | Moving to next bug."),
                        forensics=self._get_payload() if not done else None,
                        reward=reward, 
                        done=done
                    )
                else:
                    self.current_phase = "live_queue"
                    return BugTriageObservation(
                        message=msg + " | Transitioning to Phase 3.",
                        forensics=self._get_payload(),
                        queue_status=self.team_states,
                        reward=reward, 
                        done=False
                    )
            
            return BugTriageObservation(message="Error: Missing search_query or group_id.", forensics=self._get_payload(), reward=-0.5, done=False)

        # PHASE 3: LIVE QUEUE MANAGEMENT
        elif self.current_phase == "live_queue":
            target_team = target_bug["route_to"]
            team_state = self.team_states.get(target_team, next(iter(self.team_states.values())))
            
            is_new_bug_critical = (target_bug["severity"] == "critical")
            is_dev_already_critical = (team_state["active_task_severity"] == "critical")
            reward = 0.0
            
            if action.preempt_active_task is True:
                if is_new_bug_critical and not is_dev_already_critical:
                    reward += 2.0
                    msg = "Preempted for P0 Fire."
                    team_state["backlog_size"] += 1 
                    team_state["active_task_severity"] = "critical"
                elif is_new_bug_critical and is_dev_already_critical:
                    reward -= 1.5
                    msg = "BURNOUT! Dev is already on a Critical fire!"
                else:
                    reward -= 1.0
                    msg = "DEVELOPER RAGE! Unnecessary preemption."
            else:
                if is_new_bug_critical and not is_dev_already_critical:
                    reward -= 2.0
                    msg = "SYSTEM FIRE! Critical bug queued quietly."
                elif is_new_bug_critical and is_dev_already_critical:
                    reward += 2.0
                    msg = "SMART ROUTING! Dev busy on P0, bug queued safely."
                else:
                    reward += 0.5
                    msg = "Non-critical bug queued."

            self.bug_index += 1
            self.current_phase = "forensic_routing"
            done = (self.bug_index >= self.max_steps)

            return BugTriageObservation(
                message=msg + (" | Dataset exhausted." if done else " | Moving to next bug."),
                forensics=self._get_payload() if not done else None,
                reward=reward, 
                done=done
            )

    @property
    def state(self) -> State:
        """Get the current environment state (episode_id and step_count)."""
        return self._state