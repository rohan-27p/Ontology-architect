"""Supervised fine-tuning entrypoint for oracle curriculum traces."""

from __future__ import annotations

import argparse

try:
    from ontology_architect.config import load_config
    from ontology_architect.training import run_transformers_sft, write_sft_manifest
except ImportError:  # pragma: no cover
    from config import load_config
    from training import run_transformers_sft, write_sft_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_smoke.json")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--data", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    data_path = args.data or config.training.curriculum_path
    output_dir = args.output_dir or f"{config.training.output_dir}/sft"
    max_steps = args.max_steps or config.training.max_steps
    batch_size = args.batch_size or config.training.batch_size

    if args.dry_run:
        manifest = write_sft_manifest(args.model_id, data_path, output_dir, dry_run=True)
        print(f"validated SFT inputs: {manifest['examples']} examples")
        return

    run_transformers_sft(args.model_id, data_path, output_dir, max_steps, batch_size)
    write_sft_manifest(args.model_id, data_path, output_dir, dry_run=False)
    print(f"saved SFT checkpoint to {output_dir}")


if __name__ == "__main__":
    main()
