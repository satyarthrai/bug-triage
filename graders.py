# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

"""
Hackathon Evaluation Graders.

These graders decouple the evaluation logic from the environment physics.
They ensure all rewards adhere to the strict 0.0 - 1.0 limit and isolate 
scoring based on the active task difficulty.
"""

class TriageGrader:
    def __init__(self, task_name: str):
        self.task_name = task_name

    def grade_step(self, action_phase: str, raw_reward: float) -> float:
        """
        Takes the raw environment reward and scales/filters it 
        for the specific hackathon task.
        """
        # 1. Eliminate negative penalties (Floor at 0.0)
        clipped_reward = max(0.0, raw_reward)
        
        # 2. Map large bonuses (like +2.0) down to the strict cap (Ceiling at 1.0)
        # Partial rewards (like +0.5) remain intact.
        normalized_reward = min(1.0, clipped_reward)

        # 3. Isolate the scoring based on task difficulty
        if self.task_name == "triage_easy":
            # EASY: Only score points during Phase 1 (Routing)
            if action_phase == "forensic_routing":
                return normalized_reward
            return 0.0

        elif self.task_name == "triage_medium":
            # MEDIUM: Score points during Phase 1 and 2 (Routing + RAG)
            if action_phase in ["forensic_routing", "rag_deduplication"]:
                return normalized_reward
            return 0.0

        elif self.task_name == "triage_hard":
            # HARD: The agent is evaluated on the full 3-phase pipeline
            return normalized_reward

        return 0.0