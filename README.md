---
title: Ontology Architect Environment Server
emoji: 🎪
colorFrom: purple
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
---

# Ontology Architect

Ontology Architect is an OpenEnv-compatible research environment for code-driven scientific discovery. The agent receives noisy text logs from a simulated alien universe, submits a full Python theory module, and is rewarded for compact theories that predict hidden future observations.

## What Was Implemented

The echo scaffold has been replaced with a research loop:

- **Environment purpose:** discover compact ontologies for hidden latent ODE/SDE systems from raw sensor logs.
- **Action contract:** `OntologyArchitectAction(theory_module=..., revision_note=..., paradigm_shift_claim=...)`.
- **Observation contract:** one schema-guided text envelope with raw sensor logs, last execution output, peer review, and non-revealing metadata.
- **Theory API:** submitted code must define `class Theory` with `fit(history)`, `predict(window)`, and `log_prob(observations)`. Optional methods are `detect_drift(history)` and `describe()`.
- **Reward:** environment-owned Gaussian future-window log likelihood minus `mdl_lambda * len(theory_module)`, plus rare-anomaly and hidden-drift adaptation bonuses.
- **Sandbox:** theory code runs in subprocess mode for local smoke tests or Docker container mode for full runs, with timeout, memory/container settings, and import checks. Allowed imports are stdlib plus NumPy/SciPy names configured in JSON.
- **Training stages:** oracle curriculum generation, supervised fine-tuning entrypoint, and group reward optimization entrypoint.
- **Evaluation outputs:** JSON reports with discovery score, log likelihood, execution failure rate, and baseline comparisons.

## How To Run

Install dependencies:

```bash
uv sync
```

Install dev/test dependencies:

```bash
uv sync --extra dev
```

Install optional Hugging Face training dependencies:

```bash
uv sync --extra train
```

Run tests:

```bash
uv run pytest
```

Start the OpenEnv server:

```bash
uv run uvicorn server.app:app --reload
```

Run a local environment smoke test:

```bash
uv run python -m ontology_architect.scripts.smoke --config configs/tiny_smoke.json --baseline linear
```

Generate oracle curriculum data:

```bash
uv run python -m ontology_architect.scripts.generate_curriculum \
  --config configs/tiny_smoke.json \
  --output artifacts/curriculum/oracle.jsonl \
  --episodes 3
```

Launch SFT training with a Hugging Face model:

```bash
uv run python -m ontology_architect.scripts.train_sft \
  --config configs/full_research.json \
  --model-id <HF_MODEL_ID> \
  --data artifacts/curriculum/oracle.jsonl \
  --output-dir artifacts/checkpoints/sft
```

Dry-run SFT without downloading a model:

```bash
uv run python -m ontology_architect.scripts.train_sft \
  --config configs/tiny_smoke.json \
  --model-id dry-run-model \
  --data artifacts/curriculum/oracle.jsonl \
  --output-dir artifacts/checkpoints/sft-smoke \
  --dry-run
```

Launch group reward optimization:

```bash
uv run python -m ontology_architect.scripts.train_gro \
  --config configs/full_research.json \
  --model-id <HF_MODEL_ID> \
  --output-dir artifacts/checkpoints/gro \
  --group-size 4 \
  --max-steps 100
```

Dry-run group reward optimization:

```bash
uv run python -m ontology_architect.scripts.train_gro \
  --config configs/tiny_smoke.json \
  --model-id dry-run-model \
  --output-dir artifacts/checkpoints/gro-smoke \
  --dry-run
```

Run benchmark evaluation and report generation:

```bash
uv run python -m ontology_architect.scripts.evaluate \
  --config configs/tiny_smoke.json \
  --output artifacts/reports/baseline_report.json
```

Build and run the Docker image:

```bash
docker build -t ontology_architect-env:latest -f server/Dockerfile .
docker run --rm -p 8000:8000 ontology_architect-env:latest
```

## Configuration

Two example configs are included:

- `configs/tiny_smoke.json`: small local runs, subprocess sandbox, short horizons.
- `configs/full_research.json`: larger benchmark defaults, container sandbox, longer training loops.

Set the exact Hugging Face model at runtime with `--model-id`. The project does not hardcode a checkpoint.

## Theory Module Example

```python
import math


class Theory:
    def fit(self, history):
        self.last = dict(history[-1]["sensors"]) if history else {}
        return {"records": len(history)}

    def predict(self, window):
        return [{"sensors": dict(self.last), "anomaly_prob": 0.05} for _ in range(window["steps"])]

    def log_prob(self, observations):
        return -float(len(observations))

    def describe(self):
        return "Static persistence theory."
```

The environment computes the real reward from predictions and hidden future observations; `log_prob` is recorded as theory output but is not trusted as the reward.
