"""Data models for the Ontology Architect research environment."""

try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:  # pragma: no cover - older OpenEnv compatibility.
    from openenv_core.env_server.types import Action, Observation


class OntologyArchitectAction(Action):
    """A full Python theory-module rewrite submitted by the agent."""

    theory_module: str
    revision_note: str = ""
    paradigm_shift_claim: bool = False


class OntologyArchitectObservation(Observation):
    """Schema-guided text observation returned to the theory-writing agent."""

    text: str
    sensor_log: str = ""
    execution_output: str = ""
    peer_review: str = ""
