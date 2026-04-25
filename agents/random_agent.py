import json
import random

from ontology_architect.agents.base_agent import BaseAgent
from ontology_architect.models import OntologyArchitectAction, OntologyArchitectObservation

class RandomAgent(BaseAgent):
    """An agent that generates valid but mathematically random JSON DSLs."""

    def __init__(self, name: str = "random_agent"):
        super().__init__(name)

    def act(self, observation: OntologyArchitectObservation) -> OntologyArchitectAction:
        # A simple valid random theory structure
        a_coef = random.uniform(-1.0, 1.0)
        b_coef = random.uniform(-1.0, 1.0)
        noise = random.uniform(0.01, 0.5)

        theory_dict = {
            "dsl_version": 1,
            "name": f"random_theory_{random.randint(1000, 9999)}",
            "state": ["A", "B"],
            "dynamics": {
                "A": {"linear": {"terms": {"A": a_coef, "B": b_coef}}},
                "B": {"add": [{"sin": "A"}, {"linear": {"terms": {"A": -0.1}}}]}
            },
            "observations": {
                "pressure": {"var": "A"},
                "turbulence": {"pow": [{"var": "B"}, 2]},
                "thermal_radiation": {"mul": [{"var": "B"}, {"exp": {"var": "A"}}]},
                "magnetic_flux": {"mul": [0.1, {"var": "A"}, {"var": "B"}]}
            },
            "integrator": {"dt": 0.15, "substeps": 1},
            "noise": noise
        }
        
        theory_module = json.dumps(theory_dict)
        paradigm_shift_claim = random.random() < 0.05 # Rarely guess a paradigm shift
        
        return OntologyArchitectAction(
            theory_module=theory_module,
            revision_note="Randomly generated coefficients.",
            paradigm_shift_claim=paradigm_shift_claim
        )
