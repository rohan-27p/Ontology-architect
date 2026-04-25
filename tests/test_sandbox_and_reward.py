from ontology_architect.baselines import get_baseline
from ontology_architect.config import RewardConfig, SandboxConfig, UniverseConfig
from ontology_architect.reward import score_theory
from ontology_architect.sandbox import TheorySandbox
from ontology_architect.universe import ProceduralAlienUniverse


def _windows():
    records = ProceduralAlienUniverse(
        UniverseConfig(seed=11, family="thermal_split", observation_window=8, future_window=3, anomaly_rate=0.0)
    ).generate(14)
    return records[:8], records[8:11]


def test_sandbox_executes_valid_theory():
    history, future = _windows()
    result = TheorySandbox(SandboxConfig(timeout_seconds=3.0)).execute(
        get_baseline("linear"),
        history,
        future,
        ("sigma", "tau", "lambda"),
    )

    assert result.ok
    assert len(result.predictions) == len(future)


def test_sandbox_rejects_disallowed_import():
    history, future = _windows()
    result = TheorySandbox(SandboxConfig(timeout_seconds=3.0)).execute(
        "import os\n\nclass Theory:\n    pass\n",
        history,
        future,
        ("sigma", "tau", "lambda"),
    )

    assert not result.ok
    assert "not allowed" in result.error


def test_reward_penalizes_execution_errors():
    history, future = _windows()
    result = TheorySandbox(SandboxConfig(timeout_seconds=3.0)).execute(
        "class Theory:\n    pass\n",
        history,
        future,
        ("sigma", "tau", "lambda"),
    )
    reward = score_theory(result, future, "class Theory:\n    pass\n", RewardConfig())

    assert reward.reward == RewardConfig().execution_error_penalty
    assert not reward.execution_ok
