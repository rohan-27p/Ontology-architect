# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Ontology Architect environment for code-driven scientific discovery."""

try:
    from .client import OntologyArchitectEnv
    from .models import OntologyArchitectAction, OntologyArchitectObservation
except ImportError:  # pragma: no cover - source-root import during pytest collection.
    from client import OntologyArchitectEnv
    from models import OntologyArchitectAction, OntologyArchitectObservation

__all__ = ["OntologyArchitectAction", "OntologyArchitectObservation", "OntologyArchitectEnv"]

