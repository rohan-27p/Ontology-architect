from pathlib import Path

import pytest

pytest.importorskip("openenv_core")

from ontology_architect.config import ExperimentConfig, UniverseConfig
from ontology_architect.curriculum import generate_oracle_curriculum
from ontology_architect.evaluation import evaluate_baselines


def test_curriculum_generation_writes_jsonl(tmp_path: Path):
    config = ExperimentConfig(
        universe=UniverseConfig(seed=17, family="thermal_split", observation_window=8, future_window=3, max_steps=2)
    )
    output = tmp_path / "oracle.jsonl"

    examples = generate_oracle_curriculum(config, output, episodes=1)

    assert output.exists()
    assert len(examples) == 2
    assert examples[0]["completion"].strip().startswith("import")


def test_evaluation_is_reproducible():
    config = ExperimentConfig(
        universe=UniverseConfig(seed=19, family="thermal_split", observation_window=8, future_window=3, max_steps=2)
    )

    first = evaluate_baselines(config, ["static"], [19])
    second = evaluate_baselines(config, ["static"], [19])

    assert first["baselines"]["static"]["discovery_score"] == second["baselines"]["static"]["discovery_score"]
