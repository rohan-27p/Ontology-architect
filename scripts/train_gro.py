"""Group reward optimization entrypoint.

Dry-run writes a reproducible manifest. Full mode runs a lightweight
group-relative policy-gradient loop with Hugging Face Transformers/PyTorch.
"""

from __future__ import annotations

import argparse

try:
    from ontology_architect.config import load_config
    from ontology_architect.training import run_group_reward_optimization, write_gro_manifest
except ImportError:  # pragma: no cover
    from config import load_config
    from training import run_group_reward_optimization, write_gro_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_smoke.json")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir or f"{config.training.output_dir}/gro"
    group_size = args.group_size or config.training.group_size
    max_steps = args.max_steps or config.training.max_steps

    if args.dry_run:
        manifest = write_gro_manifest(args.model_id, output_dir, group_size, max_steps, dry_run=True)
        print(
            f"validated GRO config: group_size={manifest['group_size']} "
            f"max_steps={manifest['max_steps']}"
        )
        return

    result = run_group_reward_optimization(
        args.model_id,
        config,
        output_dir,
        group_size,
        max_steps,
        max_new_tokens=args.max_new_tokens,
        learning_rate=args.learning_rate,
    )
    write_gro_manifest(args.model_id, output_dir, group_size, max_steps, dry_run=False)
    print(f"saved GRO checkpoint to {result['output_dir']}")


if __name__ == "__main__":
    main()
