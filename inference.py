# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Inference Script for the Bug Triage Environment.
Orchestrates a 3-task evaluation loop across the bug pipeline.
"""

import os
import json
import textwrap
import asyncio
from typing import List, Optional

from openai import OpenAI
from dotenv import load_dotenv

# Load the modular grader to decouple physics from scoring
from graders import TriageGrader

load_dotenv()

try:
    from bug_triage_env import BugTriageEnv, BugTriageAction
except ImportError:
    from client import BugTriageEnv
    from models import BugTriageAction

# =========================================================
# CONFIGURATION & MANDATORY ENVS
# =========================================================
DEBUG_MODE = False  # Set to True only for local testing without HF_TOKEN

API_BASE_URL = os.getenv("API_BASE_URL", "https://integrate.api.nvidia.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta/llama3-70b-instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN") or "nvapi-dummy-key"

if not HF_TOKEN and not DEBUG_MODE:
    raise ValueError("HF_TOKEN environment variable is missing. Mandatory for official inference.")

BENCHMARK = "bug_triage_env"
MAX_STEPS = 50 
SUCCESS_SCORE_THRESHOLD = 0.3  # Normalized score threshold [0, 1]

# Max reward calculation for normalization (Task dependent)
# 15 bugs * ~3.0 potential max points per bug (Routing + Deduplication + Queue)
MAX_TOTAL_REWARD = 45.0 

SYSTEM_PROMPT = textwrap.dedent(
    """
    You are the Lead Forensic Systems Architect. You MUST process technical failures using the following absolute constraints.

    ### 1. MANDATORY STRUCTURE (REQUIRED EVERY STEP)
    You MUST output a valid JSON object. Every response MUST include these two keys regardless of the phase:
    - "thought": (string) Chain-of-thought reasoning. Start by stating: "Current Step: [Phase Name] | Goal: [Action]".
    - "phase": (string) The current active phase you are interacting with.

    ### 2. PHASE-SPECIFIC PROTOCOLS
    - PHASE: 'forensic_routing'
      - Action: Provide 'route_to' (team name) and 'severity' (low/medium/high/critical).
    
    - PHASE: 'rag_deduplication'
      - ACTION A (Search): If 'Search Results' is "None", provide 'search_query'.
      - ACTION B (Match): If 'Search Results' HAS DATA, provide 'group_id' (a specific ID or "new_bug"). 
      - CRITICAL: Never provide 'search_query' and 'group_id' in the same step.
    
    - PHASE: 'live_queue'
      - Action: Provide 'preempt_active_task' (true/false).
      - LOGIC: Only set to true if (New Bug == 'critical') AND (Active Dev Task != 'critical').

    ### 3. THE "NO-LOOP" RULES
    - If you already searched once and the results did not yield a perfect match, you MUST provide "group_id": "new_bug" immediately. 
    - Do not search twice for the same bug. 
    - Your priority is to process the bug and move to the next index.

    ### 4. OUTPUT FORMAT
    - ONLY raw JSON. No markdown blocks, no conversational filler, no apologies.
    """
).strip()

# =========================================================
# LOGGING UTILITIES (Hackathon Compliant)
# =========================================================
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    # MANDATORY: String format must include 'score' field as per sample
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)

# =========================================================
# AGENT LOGIC
# =========================================================
def build_user_prompt(step: int, obs: any, history: List[str]) -> str:
    search_str = json.dumps(obs.search_results, indent=2) if obs.search_results else "None"
    queue_str = json.dumps(obs.queue_status, indent=2) if obs.queue_status else "None"
    
    # Extract phase from environmental forensics payload
    current_phase = obs.forensics.get('current_phase', 'forensic_routing') if obs.forensics else "forensic_routing"
    
    return textwrap.dedent(
        f"""
        Step: {step}
        Current Phase: {current_phase}
        Forensics: {json.dumps(obs.forensics) if obs.forensics else "None"}
        Search Results from DB: {search_str}
        Developer Queues: {queue_str}
        
        Recent Action History: {history[-3:] if history else "None"}
        """
    ).strip()

def get_model_action(client: OpenAI, step: int, obs: any, history: List[str]) -> tuple:
    user_prompt = build_user_prompt(step, obs, history)
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=250,
        )
        raw_text = (completion.choices[0].message.content or "").strip()
        if "```json" in raw_text:
            raw_text = raw_text.split("```json")[1].split("```")[0].strip()
            
        action = BugTriageAction(**json.loads(raw_text))
        return action, None
        
    except Exception as exc:
        # Fallback to a valid schema on error to prevent total script crash
        return BugTriageAction(thought="JSON parse failure.", phase="error"), str(exc).replace('\n', ' ')

# =========================================================
# EVALUATION ORCHESTRATOR
# =========================================================
async def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    tasks = ["triage_easy", "triage_medium", "triage_hard"]
    
    async with BugTriageEnv(base_url="http://localhost:8000") as env:
        for current_task in tasks:
            grader = TriageGrader(current_task)
            history: List[str] = []
            rewards: List[float] = []
            steps_taken = 0
            score = 0.0
            success = False
            
            log_start(task=current_task, env=BENCHMARK, model=MODEL_NAME)

            try:
                # Pure env reset takes no task_name argument
                result = await env.reset()
                obs = result.observation

                for step in range(1, MAX_STEPS + 1):
                    if result.done:
                        break
                        
                    action, step_error = get_model_action(client, step, obs, history)
                    action_str = json.dumps(action.model_dump(exclude_none=True, exclude={'metadata'}))
                    
                    result = await env.step(action)
                    
                    raw_reward = result.reward or 0.0
                    task_reward = grader.grade_step(action.phase, raw_reward)
                    
                    rewards.append(task_reward)
                    steps_taken = step
                    
                    # --- NEW: CALCULATE SCORE EVERY STEP ---
                    current_sum = sum(rewards)
                    score = current_sum / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
                    score = min(max(score, 0.0), 1.0)
                    success = score >= SUCCESS_SCORE_THRESHOLD
                    # ---------------------------------------

                    log_step(step=step, action=action_str, reward=task_reward, done=result.done, error=step_error)
                    
                    history.append(f"Phase {action.phase}: Score {task_reward:.2f}")
                    obs = result.observation 

                    if result.done:
                        break

                # NORMALIZE SCORE [0, 1] (Hackathon Sample Requirement)
                score = sum(rewards) / MAX_TOTAL_REWARD if MAX_TOTAL_REWARD > 0 else 0.0
                score = min(max(score, 0.0), 1.0)
                success = score >= SUCCESS_SCORE_THRESHOLD

            except Exception:
                success = False
            finally:
                # Mandatory: Close the environment and print final task summary
                log_end(success=success, steps=steps_taken, score=score, rewards=rewards)
                print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    asyncio.run(main())