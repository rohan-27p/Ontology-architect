"""Run benchmark evaluation and write a JSON report."""

from __future__ import annotations

import argparse

try:
    from ontology_architect.config import load_config
    from ontology_architect.evaluation import evaluate_baselines, write_report
except ImportError:  # pragma: no cover
    from config import load_config
    from evaluation import evaluate_baselines, write_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_smoke.json")
    parser.add_argument("--output", default="artifacts/reports/baseline_report.json")
    parser.add_argument("--baselines", nargs="+", default=["static", "linear", "teacher"])
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    report = evaluate_baselines(config, args.baselines, args.seeds)
    write_report(report, args.output)
    print(f"wrote benchmark report to {args.output}")
    for name, result in report["baselines"].items():
        print(f"{name}: discovery_score={result['discovery_score']:.4f}")


if __name__ == "__main__":
    main()
