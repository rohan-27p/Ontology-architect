"""Configuration objects for Ontology Architect experiments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class UniverseConfig:
    seed: int = 13
    family: str = "thermal_split"
    split: str = "train"
    observation_window: int = 18
    future_window: int = 6
    max_steps: int = 8
    dt: float = 0.15
    sensor_noise: float = 0.04
    process_noise: float = 0.01
    anomaly_rate: float = 0.08
    anomaly_scale: float = 1.4
    drift_interval: int = 4


@dataclass(frozen=True)
class SandboxConfig:
    mode: str = "subprocess"
    container_image: str = "ontology_architect-env:latest"
    timeout_seconds: float = 3.0
    memory_mb: int = 256
    cpus: float = 1.0
    allowed_imports: tuple[str, ...] = (
        "abc",
        "collections",
        "dataclasses",
        "enum",
        "functools",
        "itertools",
        "json",
        "math",
        "numpy",
        "random",
        "scipy",
        "statistics",
        "typing",
    )


@dataclass(frozen=True)
class RewardConfig:
    mdl_lambda: float = 0.0005
    mdl_complexity_mode: str = "ast"
    prediction_sigma: float = 0.12
    anomaly_bonus: float = 2.0
    drift_bonus: float = 1.0
    false_paradigm_shift_penalty: float = 0.25
    paradigm_shift_min_structural_delta: float = 0.15
    execution_error_penalty: float = -25.0
    max_theory_chars: int = 12000


@dataclass(frozen=True)
class FeedbackConfig:
    peer_review_window: int = 8
    lineage_window: int = 8
    divergence_sigma: float = 2.0


@dataclass(frozen=True)
class TrainingConfig:
    model_id: str = ""
    output_dir: str = "artifacts/checkpoints"
    curriculum_path: str = "artifacts/curriculum/oracle.jsonl"
    report_dir: str = "artifacts/reports"
    group_size: int = 4
    max_steps: int = 100
    batch_size: int = 1
    checkpoint_steps: int = 10


@dataclass(frozen=True)
class ExperimentConfig:
    universe: UniverseConfig = UniverseConfig()
    sandbox: SandboxConfig = SandboxConfig()
    reward: RewardConfig = RewardConfig()
    feedback: FeedbackConfig = FeedbackConfig()
    training: TrainingConfig = TrainingConfig()


def _coerce_section(section: dict[str, Any] | None, cls: type) -> Any:
    section = section or {}
    allowed = {field.name for field in fields(cls)}
    clean = {key: value for key, value in section.items() if key in allowed}
    if cls is SandboxConfig and "allowed_imports" in clean:
        clean["allowed_imports"] = tuple(clean["allowed_imports"])
    return cls(**clean)


def config_from_dict(data: dict[str, Any] | None) -> ExperimentConfig:
    data = data or {}
    return ExperimentConfig(
        universe=_coerce_section(data.get("universe"), UniverseConfig),
        sandbox=_coerce_section(data.get("sandbox"), SandboxConfig),
        reward=_coerce_section(data.get("reward"), RewardConfig),
        feedback=_coerce_section(data.get("feedback"), FeedbackConfig),
        training=_coerce_section(data.get("training"), TrainingConfig),
    )


def load_config(path: str | Path | None = None) -> ExperimentConfig:
    if path is None:
        return ExperimentConfig()
    with Path(path).open("r", encoding="utf-8") as handle:
        return config_from_dict(json.load(handle))


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2, sort_keys=True)
