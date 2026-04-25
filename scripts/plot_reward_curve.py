"""Visualize the reward curve and phase transitions."""

import argparse
import json
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Plot reward curves from grader output.")
    parser.add_argument("--input", default="artifacts/grader_report.json")
    parser.add_argument("--output", default="artifacts/reward_curve.png")
    args = parser.parse_args()

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Skipping plot.")
        return

    report_path = Path(args.input)
    if not report_path.exists():
        print(f"Grader report not found at {args.input}")
        return

    with report_path.open() as f:
        report = json.load(f)

    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Phase Transitions
    # For dual_fluid config: drift_interval = 5 (so phase shift depends on config, but roughly:)
    # We will just mark some standard known bounds for demonstration or let the user configure.
    # Assuming standard demo shifts at t=30 and t=60:
    ax.axvspan(0, 30, color='green', alpha=0.1, label='Phase 1: Normal Science')
    ax.axvspan(30, 60, color='red', alpha=0.1, label='Phase 2: Crisis')
    ax.axvspan(60, 100, color='blue', alpha=0.1, label='Phase 3: Revolution')

    for agent_name, metrics in report.items():
        steps = [m["step"] for m in metrics]
        rewards = [m["reward"] for m in metrics]
        
        # Smooth the curve slightly for readability
        window = 3
        if len(rewards) >= window:
            import numpy as np
            smoothed_rewards = np.convolve(rewards, np.ones(window)/window, mode='valid')
            smooth_steps = steps[(window-1)//2 : -(window//2)]
        else:
            smoothed_rewards = rewards
            smooth_steps = steps

        ax.plot(smooth_steps, smoothed_rewards, linewidth=2, label=agent_name)

    ax.set_title("Reward Curve Over Time: Agent Recovery During Paradigm Shifts", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (Steps / Episodes)", fontsize=12)
    ax.set_ylabel("Total Reward (Log Likelihood - MDL + Bonuses)", fontsize=12)
    
    # Optional: limit y axis if RandomAgent crashes it too much
    ax.set_ylim(-50, 10) 
    
    ax.grid(alpha=0.3)
    ax.legend(loc='lower left')

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved reward curve plot to {args.output}")

if __name__ == "__main__":
    main()
