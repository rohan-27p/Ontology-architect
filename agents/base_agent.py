from abc import ABC, abstractmethod

from ontology_architect.models import OntologyArchitectAction, OntologyArchitectObservation

class BaseAgent(ABC):
    """Abstract base class for all Ontology Architect agents."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def act(self, observation: OntologyArchitectObservation) -> OntologyArchitectAction:
        """Given an observation from the environment, return an action."""
        pass
