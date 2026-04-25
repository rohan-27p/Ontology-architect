"""Benchmark evaluation helpers for Ontology Architect agents and baselines."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from statistics import mean

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


def evaluate_baselines(
    config: ExperimentConfig,
    baseline_names: list[str] | None = None,
    seeds: list[int] | None = None,
) -> dict:
    baseline_names = baseline_names or ["static", "linear", "teacher"]
    seeds = seeds or [config.universe.seed, config.universe.seed + 1]
    report = {
        "baselines": {},
        "seeds": seeds,
        "universe_family": config.universe.family,
        "split": config.universe.split,
    }
    for baseline in baseline_names:
        module = get_baseline(baseline)
        runs = []
        for seed in seeds:
            run_config = replace(config, universe=replace(config.universe, seed=seed))
            runs.append(_run_module(run_config, module, baseline))
        rewards = [run["total_reward"] for run in runs]
        log_likelihoods = [
            metric["log_likelihood"]
            for run in runs
            for metric in run["step_metrics"]
            if metric.get("execution_ok")
        ]
        report["baselines"][baseline] = {
            "runs": runs,
            "mean_total_reward": mean(rewards) if rewards else 0.0,
            "mean_log_likelihood": mean(log_likelihoods) if log_likelihoods else float("-inf"),
            "execution_failure_rate": _failure_rate(runs),
            "discovery_score": mean(rewards) if rewards else 0.0,
        }
    return report


def write_report(report: dict, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)


def _run_module(config: ExperimentConfig, module: str, label: str) -> dict:
    env = OntologyArchitectEnvironment(config)
    observation = env.reset()
    total_reward = 0.0
    step_metrics = []
    for step in range(config.universe.max_steps):
        observation = env.step(
            OntologyArchitectAction(
                theory_module=module,
                revision_note=f"benchmark baseline={label} step={step}",
            )
        )
        total_reward += float(observation.reward or 0.0)
        step_metrics.append(observation.metadata.get("last_metrics", {}))
        if observation.done:
            break
    return {
        "steps": len(step_metrics),
        "total_reward": total_reward,
        "step_metrics": step_metrics,
    }


def _failure_rate(runs: list[dict]) -> float:
    metrics = [metric for run in runs for metric in run["step_metrics"]]
    if not metrics:
        return 1.0
    failures = [metric for metric in metrics if not metric.get("execution_ok")]
    return len(failures) / len(metrics)
