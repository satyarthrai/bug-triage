---
title: Bug Triage Environment
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - nlp
---

# 🐛 RL Bug Triage Environment

An autonomous, Reinforcement Learning (RL) environment simulating a real-world enterprise bug triage pipeline. This project provides a robust, stateful testbed for evaluating LLM agents on their ability to perform forensic routing, semantic deduplication via RAG, and high-stakes developer resource allocation.

Built for the **OpenEnv Framework**, this environment challenges agents to balance accuracy with operational efficiency in a live engineering queue.

---

## 🌟 The Problem

In large-scale enterprise software, bug triage is a massive resource sink. Human managers spend hours reading stack traces, identifying if a bug has been reported before, and deciding which developer should drop their current work to fix it.

This environment automates the evaluation of AI agents attempting to solve this problem by simulating a **3-Phase Triage Pipeline**:

- **Forensic Routing:** Analyzing system logs and user descriptions to assign the correct team and severity  
- **Semantic Deduplication:** Using a Vector DB to query historical data and link duplicates  
- **Live Queue Preemption:** Deciding whether a critical bug warrants interrupting a developer's current active task  

---

## ⚙️ Robust Environment Physics

The core environment (`bug_triage_env_environment.py`) operates as a **Pure RL State Machine**. It does not know it is being evaluated; it simply simulates the physics of an engineering team.

The environment issues raw, unbounded reward signals to train robust policies:

- **Positive Reinforcement:** High-value bonuses (+2.0) for complex, correct decisions (e.g., matching a novel bug, correctly preempting a developer for a P0 fire)  
- **Negative Reinforcement:** Strict penalties for inefficient or dangerous actions:
  - Excessive database queries (-1.0)  
  - Hallucinating JSON schemas (-5.0)  
  - Burning out developers by unnecessary preemption (-1.5)  

---

## 📊 Dataset & Real-World Grounding

To ensure agents are evaluated on realistic technical debt, this environment is powered by real industry data:

- **Historical Dataset:** Built using authenticated bug reports from the Eclipse Platform GitHub/Bugzilla Repository  
- **Ongoing Pipeline (`ongoing_pipeline.json`):** Simulates a live influx of 10–15 mixed-priority bugs with real metadata  
- **Developer Queues (`developer_queues.json`):** Snapshot of team capacity, backlog, and active task severity  

---

## 🧠 Offline RL & Human-Labeled Data
This environment is specifically architected to support Offline Reinforcement Learning. The data/ folder contains high-fidelity JSON structures that represent "Expert Trajectories" derived from human engineering decisions in the Eclipse ecosystem.

### Labeling for Offline Training
Unlike standard "Online" RL where an agent must explore blindly, our data provides a supervised signal for pre-training:

- historical_bugs.json (The Knowledge Base): Contains over 5,000 human-triaged reports. Each entry includes the final route_to (Team) and severity labels assigned by human project leads, serving as the "State Space" for the RAG-based deduplication engine.
- ongoing_pipeline.json (The Expert Signal): Every bug in the test pipeline is pre-labeled with "Ground Truth" outcomes. These labels include the correct component destination and the duplicate_of ID, allowing researchers to train agents using Behavioral Cloning or Conservative Q-Learning (CQL).
- developer_queues.json (Contextual Constraints): Provides the "System State" at the time of triage, allowing the agent to learn the reward function for resource preemption based on real-world developer bandwidth.

### Training Workflow
1. Pre-training: Use the historical_bugs.json to train a transformer-based policy to predict the human-labeled route_to and severity.
2. Offline Evaluation: Run the agent against the ongoing_pipeline.json without the live server to calculate "Offline Accuracy" against the human labels.
3. Online Fine-tuning: Deploy the agent via inference.py to the OpenEnv server to refine the policy using dynamic RL reward signals, ranging from +2.0 for high-impact forensic hits to -5.0 for fatal schema hallucinations.

---

## 🏆 Progressive Difficulty (The Graders)

To comply with hackathon evaluation standards (normalized scores between 0.0 and 1.0) without polluting core environment physics, we implemented a **Decoupled Grader Architecture (`graders.py`)**.

### Tasks Overview

| Task ID       | Skill Focus            | Grader Logic |
|--------------|----------------------|-------------|
| triage_easy   | Basic Routing         | Evaluated only on Phase 1. Phase 2/3 actions yield 0.00 |
| triage_medium | Context & RAG         | Evaluated on Phases 1 & 2 (Vector DB + duplicate detection) |
| triage_hard   | Resource Allocation   | Full pipeline including live queue preemption |

---

## 🤖 LLM Inference Performance

- **Agent Tested:** `meta/llama3-70b-instruct` (via NVIDIA NIM)  
- **Max Total Reward Potential:** 45.0  

---

## 🚀 Quick Start Guide

### 1. Setup Environment Variables

Create a `.env` file:

```env
API_BASE_URL="https://integrate.api.nvidia.com/v1"
MODEL_NAME="meta/llama3-70b-instruct"
HF_TOKEN="your_huggingface_or_api_key_here"
```

### 2. Action & Observation Spaces

To enable structured forensic reasoning, the environment utilizes strict **Pydantic models** to define interaction boundaries.

#### Action Space (`BugTriageAction`)

The agent interacts with the pipeline using the following schema:

- **thought**: String field for forensic reasoning and assessing developer capacity  
- **phase**: Specifies the target phase:
  - `forensic_routing`
  - `rag_deduplication`
  - `live_queue`  
- **route_to**: Target team/component from `valid_teams` (e.g., `ui`, `runtime`, `swt`)  
- **severity**: Selected from `valid_severities`  
- **search_query**: Keywords used to query the Vector DB for duplicates  
- **group_id**: Assigns a historical bug ID or `new_bug`  
- **preempt_active_task**: Boolean flag to interrupt a developer (only for P0 cases)  

#### Observation Space (`BugTriageObservation`)

The environment returns structured system state data:

- **message**: System status or transition message  
  *(e.g., "Routed to runtime. Transitioning to Phase 2.")*  

- **forensics**: Dictionary containing:
  - System logs  
  - Stack traces  
  - OS metadata  

- **search_results**: Top-K similar bug records from Vector DB  

- **queue_status**: Real-time developer state:
  - Active task severity  
  - Backlog size  

---

### 3. Complete Setup & Execution

#### Step 1: Install Dependencies

This project uses **uv** for high-performance dependency management:

```bash
uv sync
```
#### Step 3: Build the Environment Container
Build the Docker image to ensure the environment server is isolated and portable.

```
docker build -t bug_triage_env .
```

#### Step 4: Launch the Server
The environment server must be running to handle WebSocket requests from the agent.

```
uv run server
```

#### Step 5: Run the Evaluation Inference
In a separate terminal, execute the 3-task evaluation tournament. This will generate the required [START], [STEP], and [END] logs for submission.

```
uv run python inference.py
```

---