# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ontology Architect environment for code-driven scientific discovery."""

try:
    from .models import OntologyArchitectAction, OntologyArchitectObservation
except ImportError:  # pragma: no cover - source-root import during pytest collection.
    from models import OntologyArchitectAction, OntologyArchitectObservation


def __getattr__(name: str):
    """Lazily import the client to avoid hard dependency at package import time."""
    if name == "OntologyArchitectEnv":
        try:
            from .client import OntologyArchitectEnv
        except ImportError:  # pragma: no cover - source-root import during pytest collection.
            from client import OntologyArchitectEnv
        return OntologyArchitectEnv
    raise AttributeError(name)

__all__ = ["OntologyArchitectAction", "OntologyArchitectObservation", "OntologyArchitectEnv"]

