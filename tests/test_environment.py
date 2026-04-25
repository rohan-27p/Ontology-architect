import pytest

pytest.importorskip("openenv_core")

from ontology_architect.baselines import get_baseline
from ontology_architect.config import ExperimentConfig, FeedbackConfig, UniverseConfig
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


def test_feedback_window_and_lineage_are_configurable():
    config = ExperimentConfig(
        universe=UniverseConfig(seed=3, family="thermal_split", observation_window=8, future_window=3, max_steps=3),
        feedback=FeedbackConfig(peer_review_window=2, lineage_window=1),
    )
    env = OntologyArchitectEnvironment(config)
    env.reset()
    module = get_baseline("static")

    result = None
    for step in range(3):
        result = env.step(
            OntologyArchitectAction(
                theory_module=module,
                revision_note=f"unit test step={step}",
            )
        )

    assert result is not None
    assert "THEORY LINEAGE" in result.text
    assert "Sensor error:" in result.peer_review
    assert len(result.peer_review.splitlines()) == 2
    assert len(result.metadata["theory_lineage"]) == 1
    assert "prediction_diagnostics" in result.metadata["last_metrics"]
