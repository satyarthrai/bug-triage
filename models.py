# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Bug Triage Environment.

Defines the schemas for forensic actions and observations across the 
three-phase triage pipeline (Routing, Deduplication, and Queue Management).
"""

from typing import Optional, Dict, Any, List
from pydantic import Field, ConfigDict
from openenv.core.env_server.types import Action, Observation


class BugTriageAction(Action):
    """
    Action schema supporting Forensic Routing, RAG, and Live Queues.
    
    This model enforces a strict schema for the RL agent to interact with 
    historical bug data and real-time developer resources.
    """

    thought: str = Field(
        ..., 
        description="Forensic reasoning: Verify teams, check history, and assess developer capacity."
    )
    
    phase: str = Field(
        ..., 
        description="Target phase: 'forensic_routing', 'rag_deduplication', or 'live_queue'"
    )
    
    route_to: Optional[str] = Field(
        None, 
        description="The specific Component/Team from 'valid_teams' (e.g., 'ui', 'team', 'swt')."
    )
    
    severity: Optional[str] = Field(
        None, 
        description="Severity assessment from 'valid_severities'."
    )
    
    search_query: Optional[str] = Field(
        None, 
        description="Technical keywords to search for historical duplicates."
    )
    
    group_id: Optional[str] = Field(
        None, 
        description="Assign a historical ID or 'new_bug'."
    )
    
    assign_bug_index: Optional[int] = Field(
        None, 
        description="Set to 0 to process the current target bug."
    )
    
    preempt_active_task: Optional[bool] = Field(
        False, 
        description="Set to True ONLY to interrupt for P0 critical failures."
    )

    # Pydantic V2 configuration to forbid hallucinated extra fields
    model_config = ConfigDict(extra="forbid")


class BugTriageObservation(Observation):
    """
    Observation schema returning rich system state data.
    
    Returns the current forensic context, vector search results, and 
    live developer queue statuses to the agent.
    """

    message: str = Field(
        ..., 
        description="Main system message or event trigger."
    )
    
    forensics: Optional[Dict[str, Any]] = Field(
        None, 
        description="Simulated system logs, stack traces, and OS metadata."
    )
    
    search_results: Optional[List[Dict[str, Any]]] = Field(
        None, 
        description="Top-K results returned from the Vector DB."
    )
    
    queue_status: Optional[Dict[str, Any]] = Field(
        None, 
        description="State of the live developer queue and active tasks."
    )
    
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict, 
        description="Step and phase tracking."
    )