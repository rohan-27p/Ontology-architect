"""Sandbox adapters for executing submitted theory modules."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

try:
    from .config import SandboxConfig
    from .theory_dsl import TheoryDSLValidationError, render_theory_module
    from .universe import SensorRecord, records_to_json
except ImportError:  # pragma: no cover
    from config import SandboxConfig
    from theory_dsl import TheoryDSLValidationError, render_theory_module
    from universe import SensorRecord, records_to_json


@dataclass(frozen=True)
class SandboxResult:
    ok: bool
    error: str
    stdout: str
    predictions: list[Any]
    reported_log_prob: float
    description: str
    drift_detected: bool
    runtime_ms: float
    traceback: str = ""


class TheorySandbox:
    """Run theory modules in either local subprocess or Docker container mode."""

    def __init__(self, config: SandboxConfig):
        self.config = config
        self.runner_path = Path(__file__).with_name("sandbox_runner.py").resolve()

    def execute(
        self,
        theory_module: str,
        history: list[SensorRecord],
        future: list[SensorRecord],
        sensor_names: tuple[str, ...],
    ) -> SandboxResult:
        try:
            executable_theory_module = render_theory_module(theory_module)
        except TheoryDSLValidationError as exc:
            return SandboxResult(
                ok=False,
                error=f"TheoryDSLValidationError: {exc}",
                stdout="",
                predictions=[],
                reported_log_prob=float("-inf"),
                description="",
                drift_detected=False,
                runtime_ms=0.0,
            )
        payload = {
            "theory_module": executable_theory_module,
            "history": records_to_json(history),
            "future": records_to_json(future),
            "sensor_names": list(sensor_names),
            "allowed_imports": list(self.config.allowed_imports),
        }
        if self.config.mode == "container":
            return self._execute_container(payload)
        return self._execute_subprocess(payload)

    def _execute_subprocess(self, payload: dict[str, Any]) -> SandboxResult:
        command = [sys.executable, str(self.runner_path)]
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                ok=False,
                error=f"TimeoutExpired: theory exceeded {self.config.timeout_seconds:.2f}s",
                stdout="",
                predictions=[],
                reported_log_prob=float("-inf"),
                description="",
                drift_detected=False,
                runtime_ms=self.config.timeout_seconds * 1000,
            )
        if completed.returncode != 0:
            return SandboxResult(
                ok=False,
                error=f"Sandbox process exited with code {completed.returncode}",
                stdout=completed.stdout[-4000:],
                predictions=[],
                reported_log_prob=float("-inf"),
                description="",
                drift_detected=False,
                runtime_ms=0.0,
                traceback=completed.stderr[-4000:],
            )
        return _parse_runner_output(completed.stdout)

    def _execute_container(self, payload: dict[str, Any]) -> SandboxResult:
        package_dir = self.runner_path.parent
        command = [
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--memory",
            f"{self.config.memory_mb}m",
            "--cpus",
            str(self.config.cpus),
            "-i",
            "-v",
            f"{package_dir}:/ontology_architect:ro",
            self.config.container_image,
            "python",
            "/ontology_architect/sandbox_runner.py",
        ]
        try:
            completed = subprocess.run(
                command,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds + 1.0,
                check=False,
            )
        except FileNotFoundError:
            return SandboxResult(
                ok=False,
                error="Docker executable was not found for container sandbox mode",
                stdout="",
                predictions=[],
                reported_log_prob=float("-inf"),
                description="",
                drift_detected=False,
                runtime_ms=0.0,
            )
        except subprocess.TimeoutExpired:
            return SandboxResult(
                ok=False,
                error=f"TimeoutExpired: container exceeded {self.config.timeout_seconds:.2f}s",
                stdout="",
                predictions=[],
                reported_log_prob=float("-inf"),
                description="",
                drift_detected=False,
                runtime_ms=self.config.timeout_seconds * 1000,
            )
        if completed.returncode != 0:
            return SandboxResult(
                ok=False,
                error=f"Container sandbox exited with code {completed.returncode}",
                stdout=completed.stdout[-4000:],
                predictions=[],
                reported_log_prob=float("-inf"),
                description="",
                drift_detected=False,
                runtime_ms=0.0,
                traceback=completed.stderr[-4000:],
            )
        return _parse_runner_output(completed.stdout)


def _parse_runner_output(stdout: str) -> SandboxResult:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        return SandboxResult(
            ok=False,
            error=f"Invalid sandbox output: {exc}",
            stdout=stdout[-4000:],
            predictions=[],
            reported_log_prob=float("-inf"),
            description="",
            drift_detected=False,
            runtime_ms=0.0,
        )
    return SandboxResult(
        ok=bool(payload.get("ok")),
        error=str(payload.get("error", "")),
        stdout=str(payload.get("stdout", "")),
        predictions=payload.get("predictions") or [],
        reported_log_prob=float(payload.get("reported_log_prob", float("-inf"))),
        description=str(payload.get("description", "")),
        drift_detected=bool(payload.get("drift_detected", False)),
        runtime_ms=float(payload.get("runtime_ms", 0.0)),
        traceback=str(payload.get("traceback", "")),
    )
