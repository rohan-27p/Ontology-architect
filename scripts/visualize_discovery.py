"""Visualize latent variable discovery: true vs inferred hidden states.

Usage:
    python -m ontology_architect.scripts.visualize_discovery \
        --config configs/dual_fluid_demo.json \
        --baseline dual_fluid_dsl \
        --output artifacts/discovery_plot.png

    python -m ontology_architect.scripts.visualize_discovery \
        --config configs/dual_fluid_demo.json \
        --theory-file path/to/generated_theory.json \
        --output artifacts/trained_discovery.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from ontology_architect.baselines import get_baseline
    from ontology_architect.config import load_config
    from ontology_architect.models import OntologyArchitectAction
    from ontology_architect.sandbox import TheorySandbox
    from ontology_architect.universe import ProceduralAlienUniverse
except ImportError:  # pragma: no cover
    from baselines import get_baseline
    from config import load_config
    from models import OntologyArchitectAction
    from sandbox import TheorySandbox
    from universe import ProceduralAlienUniverse


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize latent variable discovery")
    parser.add_argument("--config", default="configs/dual_fluid_demo.json")
    parser.add_argument("--baseline", default=None, help="Name of baseline to use as theory")
    parser.add_argument("--theory-file", default=None, help="Path to theory file (DSL JSON or Python)")
    parser.add_argument("--seed", type=int, default=None, help="Override universe seed")
    parser.add_argument("--steps", type=int, default=80, help="Number of timesteps to simulate")
    parser.add_argument("--output", default="artifacts/discovery_plot.png")
    parser.add_argument("--compare-baseline", default=None, help="Second baseline for comparison")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        from dataclasses import replace
        config = replace(config, universe=replace(config.universe, seed=args.seed))

    # Load theory module
    if args.theory_file:
        theory_module = Path(args.theory_file).read_text(encoding="utf-8")
    elif args.baseline:
        theory_module = get_baseline(args.baseline)
    else:
        theory_module = get_baseline("dual_fluid_dsl")

    comparison_module = get_baseline(args.compare_baseline) if args.compare_baseline else None

    # Generate ground truth with hidden latent states
    universe = ProceduralAlienUniverse(config.universe)
    records, latent_trajectory = universe.generate_with_latents(args.steps)

    # Extract true latent time series
    times = [lr.t for lr in latent_trajectory]
    latent_names = universe.latent_names
    true_latents = {name: [lr.latent_state[name] for lr in latent_trajectory] for name in latent_names}
    true_sensors = {name: [lr.sensors[name] for lr in latent_trajectory] for name in universe.sensor_names}
    noisy_sensors = {name: [r.sensors[name] for r in records] for name in universe.sensor_names}
    drift_times = [lr.t for lr in latent_trajectory if lr.drift]
    anomaly_times = [lr.t for lr in latent_trajectory if lr.anomaly]

    # Run theory through sandbox to get predictions
    sandbox = TheorySandbox(config.sandbox)
    # Use a small initial window as history so the oracle can initialize correctly
    split = 8
    history = records[:split]
    future = records[split:]

    result = sandbox.execute(theory_module, history, future, universe.sensor_names)
    predictions = result.predictions if result.ok else []

    # Also run comparison baseline if provided
    comparison_predictions = []
    if comparison_module:
        comp_result = sandbox.execute(comparison_module, history, future, universe.sensor_names)
        comparison_predictions = comp_result.predictions if comp_result.ok else []

    _plot(
        times=times,
        true_latents=true_latents,
        true_sensors=true_sensors,
        noisy_sensors=noisy_sensors,
        predictions=predictions,
        comparison_predictions=comparison_predictions,
        split_index=split,
        drift_times=drift_times,
        anomaly_times=anomaly_times,
        sensor_names=list(universe.sensor_names),
        latent_names=list(latent_names),
        output_path=args.output,
        theory_ok=result.ok,
        theory_label=args.baseline or "theory",
        compare_label=args.compare_baseline or "",
    )
    print(f"Saved discovery plot to {args.output}")
    if result.ok:
        print(f"  Theory executed OK | predictions={len(predictions)}")
    else:
        print(f"  Theory FAILED: {result.error}")


def _plot(
    times, true_latents, true_sensors, noisy_sensors,
    predictions, comparison_predictions,
    split_index, drift_times, anomaly_times,
    sensor_names, latent_names, output_path,
    theory_ok, theory_label, compare_label,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping plot generation")
        return

    n_latents = len(latent_names)
    n_sensors = len(sensor_names)
    fig, axes = plt.subplots(n_latents + n_sensors, 1, figsize=(14, 3 * (n_latents + n_sensors)),
                             sharex=True)
    fig.suptitle("Ontology Architect — Latent Variable Discovery", fontsize=16, fontweight="bold")

    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    pred_color = "#9b59b6"
    comp_color = "#95a5a6"

    # Plot hidden latent variables
    for i, name in enumerate(latent_names):
        ax = axes[i]
        ax.plot(times, true_latents[name], color=colors[i % len(colors)],
                linewidth=2, label=f"True latent {name}")
        for dt in drift_times:
            ax.axvline(dt, color="red", alpha=0.3, linestyle="--", linewidth=1)
        ax.set_ylabel(f"Latent {name}", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.2)
        ax.set_title(f"Hidden Latent Variable: {name}", fontsize=11, fontweight="bold")

    # Plot sensor predictions vs ground truth
    pred_times = times[split_index:split_index + len(predictions)]
    for i, name in enumerate(sensor_names):
        ax = axes[n_latents + i]
        # True clean sensors
        ax.plot(times, true_sensors[name], color=colors[i % len(colors)],
                linewidth=1.5, alpha=0.7, label=f"True {name}")
        # Noisy observed sensors
        ax.scatter(times, noisy_sensors[name], color=colors[i % len(colors)],
                   s=6, alpha=0.3, label=f"Observed {name}")
        # Theory predictions
        if predictions:
            pred_values = [p.get("sensors", {}).get(name, 0.0) for p in predictions]
            ax.plot(pred_times[:len(pred_values)], pred_values,
                    color=pred_color, linewidth=2, linestyle="-",
                    label=f"{theory_label} prediction")
        # Comparison predictions
        if comparison_predictions:
            comp_values = [p.get("sensors", {}).get(name, 0.0) for p in comparison_predictions]
            ax.plot(pred_times[:len(comp_values)], comp_values,
                    color=comp_color, linewidth=1.5, linestyle=":",
                    label=f"{compare_label} prediction")
        # Mark prediction boundary
        if split_index < len(times):
            ax.axvline(times[split_index], color="black", alpha=0.4,
                       linestyle="-.", linewidth=1, label="Prediction horizon")
        # Mark drift events
        for dt in drift_times:
            ax.axvline(dt, color="red", alpha=0.2, linestyle="--", linewidth=1)
        ax.set_ylabel(name, fontsize=10)
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(alpha=0.2)
        ax.set_title(f"Sensor: {name}", fontsize=11)

    axes[-1].set_xlabel("Time (t)", fontsize=11)
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()


if __name__ == "__main__":
    main()
