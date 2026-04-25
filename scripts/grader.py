"""Master script to run multiple agents and evaluate them."""

import argparse
import json
from pathlib import Path

from ontology_architect.config import load_config
from ontology_architect.server.ontology_architect_environment import OntologyArchitectEnvironment

from ontology_architect.agents.random_agent import RandomAgent
from ontology_architect.agents.heuristic_agent import HeuristicAgent
from ontology_architect.agents.llm_agent import LLMAgent
from ontology_architect.agents.base_agent import BaseAgent
from ontology_architect.reward import theory_structural_distance
from ontology_architect.baselines import get_baseline

class OracleAgent(BaseAgent):
    def act(self, observation):
        from ontology_architect.models import OntologyArchitectAction
        return OntologyArchitectAction(
            theory_module=get_baseline("dual_fluid_dsl"),
            revision_note="Oracle theory",
            paradigm_shift_claim=False
        )

def run_agent(agent: BaseAgent, config, steps: int):
    env = OntologyArchitectEnvironment(config)
    obs = env.reset()
    
    metrics = []
    previous_theory = None
    
    for step in range(steps):
        action = agent.act(obs)
        obs = env.step(action)
        
        # Track Diversity
        diversity_score = 1.0
        if previous_theory:
            diversity_score = theory_structural_distance(action.theory_module, previous_theory)
        previous_theory = action.theory_module
        
        step_metadata = obs.metadata.get("last_metrics", {})
        metrics.append({
            "step": step,
            "reward": obs.reward,
            "execution_ok": step_metadata.get("execution_ok", False),
            "log_likelihood": step_metadata.get("log_likelihood", 0.0),
            "mdl_penalty": step_metadata.get("mdl_penalty", 0.0),
            "diversity_score": diversity_score
        })
        
        if obs.done:
            break
            
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Evaluate agents.")
    parser.add_argument("--config", default="configs/dual_fluid_demo.json")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--output", default="artifacts/grader_report.json")
    args = parser.parse_args()

    config = load_config(args.config)
    
    agents = [
        RandomAgent(),
        HeuristicAgent(),
        LLMAgent(),
        OracleAgent(name="oracle_agent")
    ]
    
    report = {}
    for agent in agents:
        print(f"Running {agent.name}...")
        metrics = run_agent(agent, config, args.steps)
        report[agent.name] = metrics
        
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(report, f, indent=2)
        
    print(f"Grader report saved to {args.output}")

if __name__ == "__main__":
    main()
