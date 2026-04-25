"""Procedural alien universes with hidden latent ODE/SDE dynamics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

try:  # Package import when installed, top-level import during local scripts/tests.
    from .config import UniverseConfig
except ImportError:  # pragma: no cover
    from config import UniverseConfig


TRAIN_FAMILIES = ("thermal_split", "coupled_oscillator", "dual_fluid")
EVAL_ONLY_FAMILIES = ("plasma_relay",)
ALL_FAMILIES = TRAIN_FAMILIES + EVAL_ONLY_FAMILIES


@dataclass(frozen=True)
class SensorRecord:
    """One rendered sensor row plus hidden labels used only by scoring."""

    t: float
    sensors: dict[str, float]
    anomaly: bool = False
    drift: bool = False

    def to_json(self) -> dict:
        return {
            "t": self.t,
            "sensors": dict(self.sensors),
            "anomaly": self.anomaly,
            "drift": self.drift,
        }


@dataclass(frozen=True)
class LatentTrajectoryRecord:
    """Debug/evaluation-only record that exposes hidden latent state.

    NEVER included in agent observations — only used by visualization
    scripts, oracle baselines, and evaluation diagnostics.
    """

    t: float
    latent_state: dict[str, float]
    sensors: dict[str, float]
    params: dict[str, float]
    anomaly: bool = False
    drift: bool = False

    def to_json(self) -> dict:
        return {
            "t": self.t,
            "latent_state": dict(self.latent_state),
            "sensors": dict(self.sensors),
            "params": {k: v for k, v in self.params.items() if not k.startswith("_")},
            "anomaly": self.anomaly,
            "drift": self.drift,
        }


def records_to_json(records: Iterable[SensorRecord]) -> list[dict]:
    return [record.to_json() for record in records]


def window_has_anomaly(records: Iterable[SensorRecord | dict]) -> bool:
    return any(bool(_get_flag(record, "anomaly")) for record in records)


def window_has_drift(records: Iterable[SensorRecord | dict]) -> bool:
    return any(bool(_get_flag(record, "drift")) for record in records)


def format_sensor_log(records: Iterable[SensorRecord | dict]) -> str:
    lines = ["RAW SENSOR LOG", "columns: t | sensor=value ..."]
    for record in records:
        t = _get_time(record)
        sensors = _get_sensors(record)
        values = " ".join(f"{name}={value:+.5f}" for name, value in sensors.items())
        lines.append(f"t={t:08.3f} | {values}")
    return "\n".join(lines)


class ProceduralAlienUniverse:
    """Seeded universe generator with latent states and noisy sensors."""

    def __init__(self, config: UniverseConfig):
        self.config = config
        self.family = self._resolve_family(config.family, config.split, config.seed)
        self.rng = np.random.default_rng(config.seed)
        self._params = self._initial_params()

    @property
    def sensor_names(self) -> tuple[str, ...]:
        if self.family == "thermal_split":
            return ("sigma", "tau", "lambda")
        if self.family == "coupled_oscillator":
            return ("field", "momentum", "glow")
        if self.family == "plasma_relay":
            return ("ion", "shear", "echo")
        if self.family == "dual_fluid":
            return ("pressure", "turbulence", "thermal_radiation", "magnetic_flux")
        raise ValueError(f"Unknown universe family: {self.family}")

    @property
    def latent_names(self) -> tuple[str, ...]:
        """Names of hidden latent variables (debug/eval only)."""
        if self.family == "thermal_split":
            return ("a", "b", "c")
        if self.family == "coupled_oscillator":
            return ("x", "v", "q")
        if self.family == "plasma_relay":
            return ("ion", "shear", "echo")
        if self.family == "dual_fluid":
            return ("A", "B", "C")
        raise ValueError(f"Unknown universe family: {self.family}")

    def _state_to_dict(self, state: np.ndarray) -> dict[str, float]:
        """Convert raw state array to named dict using latent_names."""
        return {name: float(state[i]) for i, name in enumerate(self.latent_names)}

    def generate(self, n_records: int) -> list[SensorRecord]:
        state = self._initial_state()
        params = dict(self._params)
        records: list[SensorRecord] = []
        drift_every = max(1, self.config.drift_interval * self.config.future_window)

        for index in range(n_records):
            t = round(index * self.config.dt, 6)
            drift = index > 0 and index % drift_every == 0
            if drift:
                params = self._drift_params(params)

            state = self._step_state(state, params, t)
            sensors = self._project_sensors(state, params)
            sensors = {
                name: float(value + self.rng.normal(0.0, self.config.sensor_noise))
                for name, value in sensors.items()
            }

            anomaly = self.rng.random() < self.config.anomaly_rate
            if anomaly:
                target = self.rng.choice(list(sensors))
                direction = self.rng.choice((-1.0, 1.0))
                sensors[target] += float(direction * self.config.anomaly_scale)

            records.append(
                SensorRecord(
                    t=t,
                    sensors=sensors,
                    anomaly=bool(anomaly),
                    drift=bool(drift),
                )
            )
        return records

    def generate_with_latents(self, n_records: int) -> tuple[list[SensorRecord], list[LatentTrajectoryRecord]]:
        """Generate records AND hidden latent trajectories.

        **DEBUG/EVAL ONLY** — the LatentTrajectoryRecord list must NEVER
        be included in agent observations. Use this for:
        - Oracle baseline construction
        - Post-training visualization (true vs inferred latent overlay)
        - Evaluation diagnostics
        """
        state = self._initial_state()
        params = dict(self._params)
        records: list[SensorRecord] = []
        latent_records: list[LatentTrajectoryRecord] = []
        drift_every = max(1, self.config.drift_interval * self.config.future_window)

        for index in range(n_records):
            t = round(index * self.config.dt, 6)
            drift = index > 0 and index % drift_every == 0
            if drift:
                params = self._drift_params(params)

            state = self._step_state(state, params, t)
            clean_sensors = self._project_sensors(state, params)
            noisy_sensors = {
                name: float(value + self.rng.normal(0.0, self.config.sensor_noise))
                for name, value in clean_sensors.items()
            }

            anomaly = self.rng.random() < self.config.anomaly_rate
            if anomaly:
                target = self.rng.choice(list(noisy_sensors))
                direction = self.rng.choice((-1.0, 1.0))
                noisy_sensors[target] += float(direction * self.config.anomaly_scale)

            records.append(
                SensorRecord(
                    t=t,
                    sensors=noisy_sensors,
                    anomaly=bool(anomaly),
                    drift=bool(drift),
                )
            )
            latent_records.append(
                LatentTrajectoryRecord(
                    t=t,
                    latent_state=self._state_to_dict(state),
                    sensors=clean_sensors,
                    params={k: v for k, v in params.items()},
                    anomaly=bool(anomaly),
                    drift=bool(drift),
                )
            )
        return records, latent_records

    @staticmethod
    def _resolve_family(family: str, split: str, seed: int) -> str:
        if family in ALL_FAMILIES:
            return family
        if family == "auto":
            candidates = TRAIN_FAMILIES if split == "train" else ALL_FAMILIES
            return candidates[seed % len(candidates)]
        raise ValueError(f"Unknown universe family: {family}")

    def _initial_state(self) -> np.ndarray:
        if self.family == "thermal_split":
            return self.rng.normal([0.8, -0.2, 0.4], [0.12, 0.12, 0.08])
        if self.family == "coupled_oscillator":
            return self.rng.normal([0.6, 0.0, 0.3], [0.1, 0.08, 0.08])
        if self.family == "dual_fluid":
            # State: [A, B] — two coupled latent fields
            # Phase 1 (normal): autocatalytic A + forced oscillator B (omega=1.0)
            # Phase 2 (drift):  omega shifts to 1.3, old theory breaks
            # Phase 3 (crisis): third latent C emerges, affects thermal_radiation
            return self.rng.normal([0.5, 0.3, 0.0], [0.1, 0.1, 0.05])
        return self.rng.normal([0.4, -0.1, 0.2], [0.08, 0.08, 0.08])

    def _initial_params(self) -> dict[str, float]:
        if self.family == "thermal_split":
            return {"coupling": 0.06, "leak": 0.11, "phase": 0.08}
        if self.family == "coupled_oscillator":
            return {"omega": 1.12, "damping": 0.10, "charge": 0.16}
        if self.family == "dual_fluid":
            # k1, k2: autocatalytic coupling; k3, k4: forced oscillator params; omega: forcing freq
            return {"k1": 0.3, "k2": 0.5, "k3": 0.4, "k4": 0.2, "omega": 1.0, "c_active": 0.0}
        return {"drive": 0.18, "decay": 0.09, "mix": 0.12}

    def _drift_params(self, params: dict[str, float]) -> dict[str, float]:
        if self.family == "dual_fluid":
            # Phase 2: omega shifts from 1.0 → 1.3 (first drift)
            # Phase 3: c_active turns on (second drift — new latent variable emerges)
            drift_count = params.get("_drift_count", 0) + 1
            shifted = dict(params)
            shifted["_drift_count"] = drift_count
            if drift_count == 1:
                shifted["omega"] = 1.3   # paradigm shift — forcing frequency changes
            elif drift_count == 2:
                shifted["c_active"] = 1.0  # crisis — third latent C activates
            else:
                shifted["omega"] = float(params["omega"] * (1.0 + self.rng.normal(0.05, 0.02)))
            return shifted
        shifted = {}
        for key, value in params.items():
            shifted[key] = float(value * (1.0 + self.rng.normal(0.12, 0.04)))
        return shifted

    def _step_state(self, state: np.ndarray, params: dict[str, float], t: float = 0.0) -> np.ndarray:
        dt = self.config.dt
        if self.family == "thermal_split":
            a, b, c = state
            derivative = np.array(
                [
                    -params["leak"] * a + params["coupling"] * b + 0.03 * np.sin(c),
                    -0.05 * b + 0.04 * a * a - 0.02 * c,
                    params["phase"] * a - 0.07 * c,
                ]
            )
        elif self.family == "coupled_oscillator":
            x, v, q = state
            derivative = np.array(
                [
                    v,
                    -(params["omega"] ** 2) * x - params["damping"] * v + params["charge"] * q,
                    -0.08 * q + 0.06 * np.sin(x),
                ]
            )
        elif self.family == "dual_fluid":
            A, B, C = state
            omega = float(params["omega"])
            c_active = float(params.get("c_active", 0.0))
            # Hidden autocatalytic + forced oscillator ODEs
            # The agent must discover A, B, and (after drift 2) C
            dA = -params["k1"] * A + params["k2"] * B * A
            dB = params["k3"] * np.sin(omega * t) - params["k4"] * A * A
            dC = 0.15 * np.sin(0.4 * A) - 0.12 * C
            derivative = np.array([dA, dB, c_active * dC])
        else:
            ion, shear, echo = state
            derivative = np.array(
                [
                    params["drive"] * np.tanh(echo) - params["decay"] * ion,
                    -0.07 * shear + params["mix"] * ion * echo,
                    0.05 * ion - 0.04 * echo + 0.03 * np.sin(shear),
                ]
            )

        noise = self.rng.normal(0.0, self.config.process_noise, size=state.shape)
        return state + dt * derivative + noise

    def _project_sensors(self, state: np.ndarray, params: dict[str, float]) -> dict[str, float]:
        if self.family == "thermal_split":
            a, b, c = state
            return {
                "sigma": a + b,
                "tau": 0.8 * a - 0.25 * b + 0.1 * c,
                "lambda": c + 0.3 * np.sin(a),
            }
        if self.family == "coupled_oscillator":
            x, v, q = state
            return {
                "field": x + 0.25 * q,
                "momentum": v,
                "glow": x * x + 0.4 * q,
            }
        if self.family == "dual_fluid":
            A, B, C = state
            c_active = float(params.get("c_active", 0.0))
            # Composite sensor readings — the agent must decompose these
            # pressure = A (direct)
            # turbulence = (A - B)^2  — requires knowing BOTH A and B
            # thermal_radiation = B * exp(-A) + C_contribution  — twist: C contaminates this
            # magnetic_flux = 0.1 * A * B  — weak coupling, easy to miss
            return {
                "pressure":           float(A),
                "turbulence":         float((A - B) ** 2),
                "thermal_radiation":  float(B * np.exp(-abs(A)) + c_active * 0.3 * C),
                "magnetic_flux":      float(0.1 * A * B),
            }
        ion, shear, echo = state
        return {
            "ion": ion + 0.2 * shear,
            "shear": shear - 0.1 * echo,
            "echo": echo + params["mix"] * np.sin(ion),
        }


def _get_flag(record: SensorRecord | dict, key: str) -> bool:
    return bool(getattr(record, key) if isinstance(record, SensorRecord) else record.get(key, False))


def _get_time(record: SensorRecord | dict) -> float:
    return float(getattr(record, "t") if isinstance(record, SensorRecord) else record["t"])


def _get_sensors(record: SensorRecord | dict) -> dict[str, float]:
    sensors = getattr(record, "sensors") if isinstance(record, SensorRecord) else record["sensors"]
    return {str(name): float(value) for name, value in sensors.items()}
