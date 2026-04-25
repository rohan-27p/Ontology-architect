import pytest

pytest.importorskip("openenv_core")

from ontology_architect.baselines import get_baseline
from ontology_architect.config import ExperimentConfig, UniverseConfig
from ontology_architect.models import OntologyArchitectAction
from ontology_architect.server.ontology_architect_environment import OntologyArchitectEnvironment


def test_environment_reset_and_step_returns_structured_observation():
    config = ExperimentConfig(
        universe=UniverseConfig(seed=3, family="thermal_split", observation_window=8, future_window=3, max_steps=2)
    )
    env = OntologyArchitectEnvironment(config)
    observation = env.reset()

    assert "RAW SENSOR LOG" in observation.text
    assert "PEER REVIEW" in observation.text

    result = env.step(
        OntologyArchitectAction(
            theory_module=get_baseline("static"),
            revision_note="unit test",
        )
    )

    assert isinstance(result.reward, float)
    assert "LAST THEORY EXECUTION" in result.text
    assert "last_metrics" in result.metadata
