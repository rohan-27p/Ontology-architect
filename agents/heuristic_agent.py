import json
import statistics

from ontology_architect.agents.base_agent import BaseAgent
from ontology_architect.models import OntologyArchitectAction, OntologyArchitectObservation

class HeuristicAgent(BaseAgent):
    """An agent that computes basic linear trends and outputs a DSL based on them."""

    def __init__(self, name: str = "heuristic_agent"):
        super().__init__(name)

    def _parse_sensor_log(self, text: str):
        """Extract the most recent rows from the sensor log text."""
        # lines look like: 't=001.000 | pressure=+0.50 turbulence=-0.2 ...'
        lines = text.strip().split("\n")
        history = []
        for line in lines:
            if "|" not in line or "t=" not in line:
                continue
            parts = line.split("|")
            sensor_part = parts[1].strip()
            sensors = {}
            for token in sensor_part.split():
                if "=" in token:
                    k, v = token.split("=")
                    try:
                        sensors[k] = float(v)
                    except ValueError:
                        pass
            history.append(sensors)
        return history

    def act(self, observation: OntologyArchitectObservation) -> OntologyArchitectAction:
        history = self._parse_sensor_log(observation.sensor_log)
        
        if not history:
            return OntologyArchitectAction(
                theory_module=json.dumps({"dsl_version": 1, "state": ["A"]}),
                revision_note="No history available.",
                paradigm_shift_claim=False
            )

        names = list(history[-1].keys())
        last_vals = history[-1]
        
        slopes = {}
        volatility = 0.0
        
        if len(history) >= 5:
            deltas = []
            for name in names:
                series = [row.get(name, 0.0) for row in history[-5:]]
                step_deltas = [b - a for a, b in zip(series[:-1], series[1:])]
                if step_deltas:
                    slopes[name] = statistics.mean(step_deltas)
                    deltas.extend(abs(value) for value in step_deltas)
            volatility = statistics.mean(deltas) if deltas else 0.0

        paradigm_shift_claim = volatility > 0.45
        
        # We will build a theory where the state variables are exactly the sensors,
        # and the dynamics just extrapolate their recent linear trend + some noise.
        # This mirrors a simple persistence/linear model but forces it into the DSL format.
        
        theory_dict = {
            "dsl_version": 1,
            "name": "heuristic_linear_extrapolator",
            "state": names,
            "init": {name: last_vals.get(name, 0.0) for name in names},
            "dynamics": {
                name: {"const": slopes.get(name, 0.0)} for name in names
            },
            "observations": {
                name: {"var": name} for name in names
            },
            "integrator": {"dt": 0.15, "substeps": 1},
            "noise": max(0.01, volatility)
        }
        
        theory_module = json.dumps(theory_dict)

        return OntologyArchitectAction(
            theory_module=theory_module,
            revision_note=f"Extrapolating based on last 5 steps. Volatility: {volatility:.3f}",
            paradigm_shift_claim=paradigm_shift_claim
        )
