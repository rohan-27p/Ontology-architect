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

    def __init__(
        self,
        theory_module: str,
        revision_note: str = "",
        paradigm_shift_claim: bool = False,
        **kwargs,
    ):
        # Newer OpenEnv variants accept subclass fields in __init__; older ones don't.
        try:
            super().__init__(
                theory_module=theory_module,
                revision_note=revision_note,
                paradigm_shift_claim=paradigm_shift_claim,
                **kwargs,
            )
            return
        except TypeError:
            pass

        try:
            super().__init__(**kwargs)
        except TypeError:
            super().__init__()
            for key, value in kwargs.items():
                object.__setattr__(self, key, value)

        object.__setattr__(self, "theory_module", theory_module)
        object.__setattr__(self, "revision_note", revision_note)
        object.__setattr__(self, "paradigm_shift_claim", paradigm_shift_claim)


class OntologyArchitectObservation(Observation):
    """Schema-guided text observation returned to the theory-writing agent."""

    text: str
    sensor_log: str = ""
    execution_output: str = ""
    peer_review: str = ""

    def __init__(
        self,
        text: str,
        sensor_log: str = "",
        execution_output: str = "",
        peer_review: str = "",
        **kwargs,
    ):
        # Newer OpenEnv variants accept subclass fields in __init__; older ones don't.
        try:
            super().__init__(
                text=text,
                sensor_log=sensor_log,
                execution_output=execution_output,
                peer_review=peer_review,
                **kwargs,
            )
            return
        except TypeError:
            pass

        try:
            super().__init__(**kwargs)
        except TypeError:
            super().__init__()
            for key, value in kwargs.items():
                object.__setattr__(self, key, value)

        object.__setattr__(self, "text", text)
        object.__setattr__(self, "sensor_log", sensor_log)
        object.__setattr__(self, "execution_output", execution_output)
        object.__setattr__(self, "peer_review", peer_review)
