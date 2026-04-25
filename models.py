# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the Ontology Architect Environment.

The ontology_architect environment is a simple test environment that echoes back messages.
"""

from dataclasses import dataclass

from openenv_core.env_server.types import Action, Observation


@dataclass(kw_only=True)
class OntologyArchitectAction(Action):
    """Action for the Ontology Architect environment - just a message to echo."""

    message: str


@dataclass(kw_only=True)
class OntologyArchitectObservation(Observation):
    """Observation from the Ontology Architect environment - the echoed message."""

    echoed_message: str
    message_length: int = 0

