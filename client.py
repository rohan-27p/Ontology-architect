# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HTTP client for the Ontology Architect research environment."""

from typing import Any, Dict

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient
    from openenv.core.env_server.types import State
except ImportError:  # pragma: no cover - older OpenEnv compatibility.
    from openenv_core.client_types import StepResult
    from openenv_core.env_client import EnvClient
    from openenv_core.env_server.types import State

try:
    from .models import OntologyArchitectAction, OntologyArchitectObservation
except ImportError:  # pragma: no cover - source-root import during pytest collection.
    from models import OntologyArchitectAction, OntologyArchitectObservation


class OntologyArchitectEnv(EnvClient[OntologyArchitectAction, OntologyArchitectObservation, State]):
    """HTTP client for reset/step/state interactions."""

    def _step_payload(self, action: OntologyArchitectAction) -> Dict:
        """
        Convert OntologyArchitectAction to JSON payload for step request.

        Args:
            action: OntologyArchitectAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "theory_module": action.theory_module,
            "revision_note": action.revision_note,
            "paradigm_shift_claim": action.paradigm_shift_claim,
        }

    def _parse_result(self, payload: Dict) -> StepResult[OntologyArchitectObservation]:
        """
        Parse server response into StepResult[OntologyArchitectObservation].

        Args:
            payload: JSON response from server

        Returns:
            StepResult with OntologyArchitectObservation
        """
        obs_data = payload.get("observation", {})
        observation = OntologyArchitectObservation(
            text=obs_data.get("text", ""),
            sensor_log=obs_data.get("sensor_log", ""),
            execution_output=obs_data.get("execution_output", ""),
            peer_review=obs_data.get("peer_review", ""),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from /state endpoint

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
