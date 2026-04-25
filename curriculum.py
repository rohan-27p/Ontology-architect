"""Oracle curriculum generation for staged code-agent training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

try:
    from .baselines import get_baseline
    from .config import ExperimentConfig
    from .models import OntologyArchitectAction
    from .server.ontology_architect_environment import OntologyArchitectEnvironment
except ImportError:  # pragma: no cover
    from baselines import get_baseline
    from config import ExperimentConfig
    from models import OntologyArchitectAction
    from server.ontology_architect_environment import OntologyArchitectEnvironment


def generate_oracle_curriculum(
    config: ExperimentConfig,
    output_path: str | Path,
    episodes: int = 3,
    teacher_name: str = "teacher",
) -> list[dict]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    examples: list[dict] = []
    teacher_module = get_baseline(teacher_name)

    with output.open("w", encoding="utf-8") as handle:
        for episode_index in range(episodes):
            env = OntologyArchitectEnvironment(config)
            observation = env.reset()
            for step_index in range(config.universe.max_steps):
                example = {
                    "episode": episode_index,
                    "step": step_index,
                    "prompt": observation.text,
                    "completion": teacher_module,
                    "teacher": teacher_name,
                }
                result = env.step(
                    OntologyArchitectAction(
                        theory_module=teacher_module,
                        revision_note="oracle curriculum teacher rewrite",
                    )
                )
                example["reward"] = result.reward
                example["metrics"] = result.metadata.get("last_metrics", {})
                handle.write(json.dumps(example, sort_keys=True) + "\n")
                examples.append(example)
                observation = result
                if result.done:
                    break
    return examples


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(rows: Iterable[dict], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
