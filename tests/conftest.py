"""Pytest configuration for local source-layout imports."""

from __future__ import annotations

import sys
from pathlib import Path


# Ensure imports like `ontology_architect.config` resolve when pytest runs from
# the repository package directory.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))