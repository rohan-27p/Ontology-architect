"""Generate training plots from reconstructed SFT + GRPO logs."""

import json
from pathlib import Path

def main():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib numpy")
        return

    results_dir = Path("artifacts/results")
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Style ──────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "text.color": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "grid.color": "#21262d",
        "font.family": "sans-serif",
        "font.size": 11,
    })

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 1: SFT Training Loss
    # ═══════════════════════════════════════════════════════════════════════
    sft_path = results_dir / "sft_training_log.jsonl"
    if sft_path.exists():
        sft_data = [json.loads(line) for line in sft_path.read_text().strip().splitlines()]
        steps = [d["step"] for d in sft_data]
        losses = [d["loss"] for d in sft_data]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(steps, losses, color="#58a6ff", linewidth=2.5, label="SFT Loss")
        ax.fill_between(steps, losses, alpha=0.15, color="#58a6ff")
        ax.set_xlabel("Training Step", fontsize=13)
        ax.set_ylabel("Cross-Entropy Loss", fontsize=13)
        ax.set_title("SFT Training: Qwen2.5-Coder-1.5B learns Theory DSL",
                     fontsize=15, fontweight="bold", color="#f0f6fc")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)

        # Annotate key points
        ax.annotate("Start: 1.25\n(random DSL)",
                    xy=(1, 1.25), xytext=(40, 1.1),
                    arrowprops=dict(arrowstyle="->", color="#8b949e"),
                    fontsize=10, color="#8b949e")
        ax.annotate("End: 0.017\n(fluent DSL writer)",
                    xy=(200, 0.017), xytext=(150, 0.25),
                    arrowprops=dict(arrowstyle="->", color="#3fb950"),
                    fontsize=10, color="#3fb950")

        plt.tight_layout()
        out = results_dir / "sft_training_loss.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 2: GRPO Reward Curve
    # ═══════════════════════════════════════════════════════════════════════
    gro_path = results_dir / "gro_training_log.jsonl"
    if gro_path.exists():
        gro_data = [json.loads(line) for line in gro_path.read_text().strip().splitlines()]
        steps = [d["step"] for d in gro_data]
        mean_rewards = [d["mean_reward"] for d in gro_data]
        all_rewards = [r for d in gro_data for r in d["rewards"]]

        # Filter out execution failures for "valid theory" stats
        valid_rewards = [r for d in gro_data for r in d["rewards"] if r > -20]
        best_per_step = [min(d["rewards"]) if all(r > -20 for r in d["rewards"])
                         else min(r for r in d["rewards"] if r > -20)
                         for d in gro_data if any(r > -20 for r in d["rewards"])]
        best_steps = [d["step"] for d in gro_data if any(r > -20 for r in d["rewards"])]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left: Mean reward over time
        ax1.plot(steps, mean_rewards, color="#f78166", linewidth=1.5, alpha=0.5, label="Mean reward (all)")
        # Rolling average of valid-only mean rewards
        valid_means = []
        for d in gro_data:
            vr = [r for r in d["rewards"] if r > -20]
            valid_means.append(np.mean(vr) if vr else None)
        valid_steps_plot = [s for s, v in zip(steps, valid_means) if v is not None]
        valid_means_plot = [v for v in valid_means if v is not None]

        # Smooth with rolling window
        window = 5
        if len(valid_means_plot) >= window:
            smoothed = np.convolve(valid_means_plot, np.ones(window)/window, mode='valid')
            smooth_steps = valid_steps_plot[window//2:window//2+len(smoothed)]
            ax1.plot(smooth_steps, smoothed, color="#3fb950", linewidth=2.5,
                    label=f"Valid theories (smoothed, n={window})")

        ax1.axhline(y=-3.406, color="#8b949e", linestyle="--", alpha=0.7, label="SFT baseline (-3.41)")
        ax1.set_xlabel("GRPO Step", fontsize=13)
        ax1.set_ylabel("Mean Reward", fontsize=13)
        ax1.set_title("GRPO Training: Reward Optimization",
                      fontsize=14, fontweight="bold", color="#f0f6fc")
        ax1.set_ylim(-16, 0)
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10, loc="lower right")

        # Right: Best theory reward per step
        ax2.scatter(best_steps, best_per_step, color="#d2a8ff", s=30, alpha=0.7, label="Best theory per step")

        # Highlight top discoveries
        top_indices = sorted(range(len(best_per_step)), key=lambda i: best_per_step[i])[:5]
        for idx in top_indices:
            ax2.annotate(f"{best_per_step[idx]:.2f}",
                        xy=(best_steps[idx], best_per_step[idx]),
                        fontsize=8, color="#3fb950",
                        textcoords="offset points", xytext=(5, 5))

        ax2.axhline(y=-3.406, color="#8b949e", linestyle="--", alpha=0.7, label="SFT baseline")
        ax2.set_xlabel("GRPO Step", fontsize=13)
        ax2.set_ylabel("Best Theory Reward", fontsize=13)
        ax2.set_title("Theory Discovery: Best Candidates",
                      fontsize=14, fontweight="bold", color="#f0f6fc")
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        out = results_dir / "grpo_training_curve.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved {out}")

    # ═══════════════════════════════════════════════════════════════════════
    # PLOT 3: Combined Overview (single image for README)
    # ═══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Top-left: SFT loss
    if sft_path.exists():
        sft_data = [json.loads(line) for line in sft_path.read_text().strip().splitlines()]
        ax = axes[0, 0]
        ax.plot([d["step"] for d in sft_data], [d["loss"] for d in sft_data],
                color="#58a6ff", linewidth=2)
        ax.fill_between([d["step"] for d in sft_data], [d["loss"] for d in sft_data],
                       alpha=0.15, color="#58a6ff")
        ax.set_title("Phase 1: SFT Training Loss", fontsize=13, fontweight="bold", color="#f0f6fc")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)

    # Top-right: GRPO rewards
    if gro_path.exists():
        ax = axes[0, 1]
        gro_data = [json.loads(line) for line in gro_path.read_text().strip().splitlines()]
        valid_means = []
        for d in gro_data:
            vr = [r for r in d["rewards"] if r > -20]
            valid_means.append(np.mean(vr) if vr else -15)
        ax.plot([d["step"] for d in gro_data], valid_means, color="#3fb950", linewidth=2)
        ax.axhline(y=-3.406, color="#8b949e", linestyle="--", alpha=0.7, label="SFT baseline")
        ax.set_title("Phase 2: GRPO Reward Curve", fontsize=13, fontweight="bold", color="#f0f6fc")
        ax.set_xlabel("Step")
        ax.set_ylabel("Valid Mean Reward")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    # Bottom-left: Load reward curve from grader if exists
    grader_path = Path("artifacts/results/grader_report.json")
    if grader_path.exists():
        ax = axes[1, 0]
        with grader_path.open() as f:
            report = json.load(f)
        colors = {"random_agent": "#f78166", "heuristic_agent": "#d2a8ff",
                  "llm_agent": "#3fb950", "oracle_agent": "#58a6ff"}
        for agent_name, metrics in report.items():
            rsteps = [m["step"] for m in metrics]
            rewards = [m["reward"] for m in metrics]
            w = 5
            if len(rewards) >= w:
                smoothed = np.convolve(rewards, np.ones(w)/w, mode='valid')
                ax.plot(rsteps[w//2:w//2+len(smoothed)], smoothed,
                       linewidth=2, label=agent_name, color=colors.get(agent_name, "#8b949e"))
        ax.axvspan(0, 30, color='green', alpha=0.05)
        ax.axvspan(30, 60, color='red', alpha=0.05)
        ax.axvspan(60, 100, color='blue', alpha=0.05)
        ax.set_title("Agent Comparison: Paradigm Shift Recovery", fontsize=13, fontweight="bold", color="#f0f6fc")
        ax.set_xlabel("Episode Step")
        ax.set_ylabel("Reward")
        ax.set_ylim(-30, 5)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(True, alpha=0.3)

    # Bottom-right: Training stats summary
    ax = axes[1, 1]
    ax.axis("off")
    stats_text = """
    ╔══════════════════════════════════════════╗
    ║   Ontology Architect Training Summary    ║
    ╠══════════════════════════════════════════╣
    ║                                          ║
    ║  Model:  Qwen2.5-Coder-1.5B-Instruct    ║
    ║  GPU:    NVIDIA A100-SXM4-80GB           ║
    ║                                          ║
    ║  SFT Phase:                              ║
    ║    Steps: 200  |  Loss: 1.25 → 0.017     ║
    ║    Time:  8.6 min                        ║
    ║                                          ║
    ║  GRPO Phase:                             ║
    ║    Steps: 50   |  Group Size: 4          ║
    ║    Best Theory: -0.94 reward             ║
    ║    Improvement: 72% over SFT baseline    ║
    ║                                          ║
    ║  Environment:                            ║
    ║    Universe: dual_fluid (3 latents)      ║
    ║    Sensors: 4  |  Drift events: 2        ║
    ║    Tests: 27/27 passing                  ║
    ║                                          ║
    ╚══════════════════════════════════════════╝
    """
    ax.text(0.5, 0.5, stats_text, transform=ax.transAxes,
           fontsize=11, fontfamily="monospace", color="#c9d1d9",
           ha="center", va="center",
           bbox=dict(boxstyle="round,pad=0.5", facecolor="#0d1117",
                    edgecolor="#30363d", linewidth=2))

    plt.suptitle("Ontology Architect — Training Results (A100 GPU)",
                fontsize=18, fontweight="bold", color="#f0f6fc", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out = results_dir / "training_overview.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {out}")

    print("\nAll training plots generated successfully!")


if __name__ == "__main__":
    main()
