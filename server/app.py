# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the Ontology Architect Environment.

This module exposes the research environment over HTTP, making it compatible
with HTTPEnvClient.

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as first_error:
    try:
        from openenv_core.env_server.http_server import create_app
    except Exception as e:  # pragma: no cover
        raise ImportError("OpenEnv is required for the web interface. Install dependencies with '\n    uv sync\n'") from first_error

from .ontology_architect_environment import OntologyArchitectEnvironment

try:
    from ontology_architect.models import OntologyArchitectAction, OntologyArchitectObservation
except ImportError:  # pragma: no cover
    from models import OntologyArchitectAction, OntologyArchitectObservation

# Create the app with web interface and README integration.
# Current OpenEnv expects an environment factory so every session gets fresh state.
app = create_app(
    OntologyArchitectEnvironment,
    OntologyArchitectAction,
    OntologyArchitectObservation,
    env_name="ontology_architect",
)


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m ontology_architect.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn ontology_architect.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
