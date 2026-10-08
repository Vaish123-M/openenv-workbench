# openenv-workbench

openenv-workbench is a small OpenEnv-compliant benchmark for real-world agent workflows. It simulates three non-game tasks: email classification, data cleaning, and customer support reply generation.

## Project Overview

The environment is intentionally lightweight and deterministic. Each task exposes a clear objective, a step-based reward signal, and a deterministic grader that returns a normalized score between `0.0` and `1.0`.

## Structure

- `environment/` contains the OpenEnv runtime, FastAPI app, and shared Pydantic models.
- `tasks/` contains the task definitions and deterministic graders.
- `models/` contains the OpenAI-compatible client helper and prompting utilities.
- `inference.py` runs all tasks using an OpenAI-compatible endpoint.
- `openenv.yaml` describes the environment metadata.
- `Dockerfile` builds and runs the API service.

## Tasks

### Easy: Email Classification

Classify a suspicious email as `spam` or `important` and justify the decision with evidence from the message.

### Medium: Data Cleaning

Normalize a messy customer table by removing duplicates and fixing names, emails, dates, and phone numbers.

### Hard: Customer Support Reply Generation

Draft a helpful support response that acknowledges the issue, follows policy, and gives a concrete next step.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the FastAPI app locally:

```bash
uvicorn environment.app:app --host 0.0.0.0 --port 8000
```

Run in Docker:

```bash
docker build -t openenv-workbench .
docker run -p 8000:8000 openenv-workbench
```

## API Usage

### Reset

```bash
curl -X POST http://localhost:8000/reset -H "Content-Type: application/json" -d "{\"task_name\": \"email_classification\"}"
```

### Step

```bash
curl -X POST http://localhost:8000/step -H "Content-Type: application/json" -d "{\"action\": {\"content\": \"{\\\"label\\\": \\\"spam\\\", \\\"reason\\\": \\\"It asks for card details and uses urgent language.\\\"}\", \"format\": \"json\"}}"
```

### State

```bash
curl http://localhost:8000/state
```

## Baseline Inference

`inference.py` uses an OpenAI-compatible client and reads these environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Example:

```bash
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4o-mini
set HF_TOKEN=your_token_here
python inference.py
```

## Agent Runner

Phase 1 includes a reusable runner for executing any agent that implements
`observe(observation) -> action` against the existing `OpenEnv` environment.
The deterministic `MockAgent` is useful for local checks without an LLM:

```python
from agent import AgentRunner, MockAgent
from environment.core import OpenEnv

agent = MockAgent([
    {
        "content": "{\"label\": \"spam\", \"reason\": \"Urgent request for card details.\"}",
        "format": "json",
    }
])
result = AgentRunner().run(OpenEnv(), agent, task_id="email_classification", max_steps=3)
print(result.model_dump())
```

The runner returns a structured `RunResult` for task completion, max-step
termination, invalid actions, agent/environment errors, missing observations,
and cooperative timeout checks. Timeout checks occur between calls; a call
already in progress cannot be forcefully interrupted in this phase.

The script logs exactly in this format for each task:

```text
[START]
task: <task_name>

[STEP]
action: <action>
reward: <value>

[END]
final_score: <value>
```

## Validation

The repository is structured to be compatible with `openenv validate` by exposing the required OpenEnv methods, deterministic task graders, and a FastAPI API surface.
