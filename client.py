# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bug Triage Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

try:
    from .models import BugTriageAction, BugTriageObservation
except (ImportError, ModuleNotFoundError):
    from models import BugTriageAction, BugTriageObservation


class BugTriageEnv(
    EnvClient[BugTriageAction, BugTriageObservation, State]
):
    """
    Client for the Bug Triage Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with BugTriageEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.message)
        ...
        ...     action = BugTriageAction(thought="Initial routing", phase="forensic_routing", route_to="ui")
        ...     result = client.step(action)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = BugTriageEnv.from_docker_image("bug_triage_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: BugTriageAction) -> Dict:
        """
        Convert BugTriageAction to JSON payload for step message.

        Args:
            action: BugTriageAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        # Using model_dump to handle the complex Phase 1-3 RL action schema
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[BugTriageObservation]:
        """
        Parse server response into StepResult[BugTriageObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with BugTriageObservation
        """
        obs_data = payload.get("observation", {})
        
        # Mapping the rich forensic payloads from the environment server
        observation = BugTriageObservation(
            message=obs_data.get("message", ""),
            forensics=obs_data.get("forensics", {}),
            search_results=obs_data.get("search_results"),
            queue_status=obs_data.get("queue_status"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward", 0.0),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )