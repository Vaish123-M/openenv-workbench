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

### Controlled Agent Run

The internal `/runs` endpoint only accepts registered environment and agent
names. It does not import or execute caller-supplied classes or code.

```bash
curl -X POST http://localhost:8000/runs \
  -H "Content-Type: application/json" \
  -d "{\"environment\":\"openenv\",\"task\":\"email_classification\",\"agent\":\"mock\",\"max_steps\":3,\"actions\":[{\"content\":\"{\\\"label\\\":\\\"spam\\\",\\\"reason\\\":\\\"Verify suspension card click immediately form.\\\"}\",\"format\":\"json\"}]}"
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

## Benchmarks

`BenchmarkRunner` executes named tasks through the existing `AgentRunner`,
records every `RunResult`, and aggregates success rate, score, steps,
execution time, failures, and timeouts. It supports repeated runs, filters,
and JSON/CSV export. A minimal CLI is available:

```bash
python -m benchmark.cli run --environment openenv \
  --tasks email_classification --agent mock --runs 2 \
  --actions mock-actions.json --output-json results.json --output-csv results.csv
```

The actions file is a JSON object keyed by task ID, with each value containing
the action objects supplied to `MockAgent`.

## Coding Environment

The Phase 3 coding environment is available as the registered `coding`
environment for controlled runs. It creates a temporary task workspace and
accepts only JSON tool actions: `list_files`, `read_file`, `edit_file`,
`run_tests`, and `submit`. Paths are restricted to that workspace and tests
run through a fixed Python/pytest invocation with a timeout.

An LLM-backed agent can use any provider implementing `complete(prompt) -> str`.
`OpenAIProvider` is an OpenAI-compatible implementation that reads credentials
from `OPENAI_API_KEY` or `HF_TOKEN`:

```python
from agent import AgentRunner, LLMBackedAgent, OpenAIProvider
from environment.core import OpenEnv

agent = LLMBackedAgent(OpenAIProvider())
result = AgentRunner().run(OpenEnv(), agent, task_id="email_classification")
```

For asynchronous agents/providers or environments, use
`await AgentRunner().run_async(...)`. Awaitable calls are cancelled when the
overall timeout expires, and external task cancellation is propagated after
best-effort cleanup. Synchronous calls used through `run_async` remain
non-preemptible because Python cannot safely interrupt a blocking call.

Each `RunResult` also includes `metrics` and `grading`. Metrics report observed
step counts, failures, invalid actions, elapsed time, timeout status, and
provider model/token usage when available. Grading is populated from the
environment's existing `score`, `reward`, `penalty`, and `breakdown` fields.
Every run also contains a structured `trajectory` with each observation,
action, environment response, timestamp, duration, and error. Deterministic
failure summaries classify invalid actions, agent/environment errors,
constraint violations, incorrect output, incomplete tasks, timeouts, and
repeated failed actions. Benchmark summaries aggregate these categories and
track repeated actions and wasted steps.

## Multi-agent benchmark comparison

The benchmark runner can execute the same tasks and evaluation settings for
multiple registered agents. Agent/provider construction remains in the agent
registry; benchmark configuration only names the registered agents:

```python
from benchmark import BenchmarkConfig, BenchmarkRunner

comparison = BenchmarkRunner().run_comparison(
    BenchmarkConfig(
        environment="openenv",
        task_ids=["email_classification"],
        agents=["mock", "openai"],
        runs=3,
        max_steps=10,
    )
)

for agent_name, summary in comparison.summaries.items():
    print(agent_name, summary.success_rate, summary.average_score)
```

Each record includes the registered agent and observed model name when the
agent exposes one. Per-agent summaries include success rate, average score,
average steps, average execution time, failure rate, and timeout rate. A
single agent failure is recorded in that agent's results without stopping
other agents. Comparison results can be exported with
`BenchmarkRunner.export_comparison_json` and
`BenchmarkRunner.export_comparison_csv`.

The CLI supports the same flow:

```bash
python -m benchmark.cli run \
  --environment openenv \
  --tasks email_classification \
  --agents mock openai \
  --runs 3 \
  --output-json comparison.json \
  --output-csv comparison.csv
```

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
