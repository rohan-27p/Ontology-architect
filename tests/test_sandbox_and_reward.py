import json

from ontology_architect.baselines import get_baseline
from ontology_architect.config import RewardConfig, SandboxConfig, UniverseConfig
from ontology_architect.reward import score_theory, theory_complexity
from ontology_architect.sandbox import SandboxResult, TheorySandbox
from ontology_architect.universe import ProceduralAlienUniverse, SensorRecord


def _windows():
    records = ProceduralAlienUniverse(
        UniverseConfig(seed=11, family="thermal_split", observation_window=8, future_window=3, anomaly_rate=0.0)
    ).generate(14)
    return records[:8], records[8:11]


def _dsl_module():
    return json.dumps(
        {
            "dsl_version": 1,
            "name": "sensor persistence DSL",
            "state": ["sigma", "tau", "lambda"],
            "dynamics": {"sigma": 0.0, "tau": 0.0, "lambda": 0.0},
            "observations": {
                "sigma": {"var": "sigma"},
                "tau": {"var": "tau"},
                "lambda": {"var": "lambda"},
            },
            "fit": {"lookback": 4, "trend_weight": 1.0},
            "integrator": {"dt": 1.0, "substeps": 1},
            "noise": 0.12,
        },
        sort_keys=True,
    )


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


def test_sandbox_executes_structured_theory_dsl():
    history, future = _windows()
    result = TheorySandbox(SandboxConfig(timeout_seconds=3.0)).execute(
        _dsl_module(),
        history,
        future,
        ("sigma", "tau", "lambda"),
    )

    assert result.ok
    assert len(result.predictions) == len(future)
    assert "Structured Theory DSL" in result.description


def test_sandbox_rejects_invalid_theory_dsl_before_execution():
    history, future = _windows()
    invalid_dsl = json.dumps(
        {
            "dsl_version": 1,
            "state": ["sigma"],
            "observations": {"sigma": {"var": "unknown_latent"}},
        }
    )

    result = TheorySandbox(SandboxConfig(timeout_seconds=3.0)).execute(
        invalid_dsl,
        history,
        future,
        ("sigma", "tau", "lambda"),
    )

    assert not result.ok
    assert "TheoryDSLValidationError" in result.error


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


def test_sandbox_rejects_unlisted_numpy_submodule_import():
    history, future = _windows()
    result = TheorySandbox(SandboxConfig(timeout_seconds=3.0)).execute(
        "from numpy import linalg\n\nclass Theory:\n    pass\n",
        history,
        future,
        ("sigma", "tau", "lambda"),
    )

    assert not result.ok
    assert "numpy.linalg" in result.error


def test_sandbox_rejects_unlisted_numpy_submodule_attribute_access():
    history, future = _windows()
    result = TheorySandbox(SandboxConfig(timeout_seconds=3.0)).execute(
        "import numpy as np\nvalue = np.linalg.norm([1.0])\n\nclass Theory:\n    pass\n",
        history,
        future,
        ("sigma", "tau", "lambda"),
    )

    assert not result.ok
    assert "numpy.linalg" in result.error


def test_sandbox_rejects_unlisted_numpy_submodule_getattr_access():
    history, future = _windows()
    result = TheorySandbox(SandboxConfig(timeout_seconds=3.0)).execute(
        "import numpy as np\nvalue = getattr(np, 'linalg').norm([1.0])\n\nclass Theory:\n    pass\n",
        history,
        future,
        ("sigma", "tau", "lambda"),
    )

    assert not result.ok
    assert "numpy.linalg" in result.error


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


def test_ast_mdl_ignores_comments_and_whitespace():
    compact = "class Theory:\n    pass\n"
    padded = "# explanatory comment\n\nclass Theory:\n\n    pass\n"

    assert theory_complexity(compact).score == theory_complexity(padded).score


def test_dsl_mdl_uses_semantic_complexity():
    complexity = theory_complexity(_dsl_module())

    assert complexity.mode == "dsl"
    assert complexity.score > 0.0


def test_drift_bonus_requires_structural_change_for_repeated_theory():
    future = [SensorRecord(t=0.0, sensors={"sigma": 1.0}, drift=True)]
    result = SandboxResult(
        ok=True,
        error="",
        stdout="",
        predictions=[{"sensors": {"sigma": 1.0}}],
        reported_log_prob=0.0,
        description="",
        drift_detected=False,
        runtime_ms=0.0,
    )
    module = "class Theory:\n    pass\n"

    reward = score_theory(
        result,
        future,
        module,
        RewardConfig(),
        paradigm_shift_claim=True,
        previous_theory_module=module,
    )

    assert reward.drift_bonus == 0.0
    assert reward.weak_paradigm_shift
