"""Run a tiny local Ontology Architect rollout."""

from __future__ import annotations

import argparse

try:
    from ontology_architect.baselines import get_baseline
    from ontology_architect.config import load_config
    from ontology_architect.models import OntologyArchitectAction
    from ontology_architect.server.ontology_architect_environment import OntologyArchitectEnvironment
except ImportError:  # pragma: no cover
    from baselines import get_baseline
    from config import load_config
    from models import OntologyArchitectAction
    from server.ontology_architect_environment import OntologyArchitectEnvironment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tiny_smoke.json")
    parser.add_argument("--baseline", default="linear", choices=["static", "linear", "teacher", "oracle"])
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    config = load_config(args.config)
    env = OntologyArchitectEnvironment(config)
    observation = env.reset()
    module = get_baseline(args.baseline)
    print(observation.text.splitlines()[0])
    for step in range(args.steps):
        observation = env.step(
            OntologyArchitectAction(
                theory_module=module,
                revision_note=f"smoke baseline={args.baseline} step={step}",
            )
        )
        metrics = observation.metadata.get("last_metrics", {})
        print(
            f"step={step} reward={observation.reward:.4f} "
            f"log_likelihood={metrics.get('log_likelihood')} done={observation.done}"
        )
        if observation.done:
            break


if __name__ == "__main__":
    main()
