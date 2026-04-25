"""Generate oracle curriculum JSONL data."""

from __future__ import annotations

import argparse

try:
    from ontology_architect.config import load_config
    from ontology_architect.curriculum import generate_oracle_curriculum
except ImportError:  # pragma: no cover
    from config import load_config
    from curriculum import generate_oracle_curriculum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_smoke.json")
    parser.add_argument("--output", default="artifacts/curriculum/oracle.jsonl")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--teacher", default="teacher")
    args = parser.parse_args()

    config = load_config(args.config)
    examples = generate_oracle_curriculum(config, args.output, args.episodes, args.teacher)
    print(f"wrote {len(examples)} curriculum examples to {args.output}")


if __name__ == "__main__":
    main()
