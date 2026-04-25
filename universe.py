"""Procedural alien universes with hidden latent ODE/SDE dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

try:  # Package import when installed, top-level import during local scripts/tests.
    from .config import UniverseConfig
except ImportError:  # pragma: no cover
    from config import UniverseConfig


TRAIN_FAMILIES = ("thermal_split", "coupled_oscillator")
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
        raise ValueError(f"Unknown universe family: {self.family}")

    def generate(self, n_records: int) -> list[SensorRecord]:
        state = self._initial_state()
        params = dict(self._params)
        records: list[SensorRecord] = []
        drift_every = max(1, self.config.drift_interval * self.config.future_window)

        for index in range(n_records):
            drift = index > 0 and index % drift_every == 0
            if drift:
                params = self._drift_params(params)

            state = self._step_state(state, params)
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
                    t=round(index * self.config.dt, 6),
                    sensors=sensors,
                    anomaly=bool(anomaly),
                    drift=bool(drift),
                )
            )
        return records

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
        return self.rng.normal([0.4, -0.1, 0.2], [0.08, 0.08, 0.08])

    def _initial_params(self) -> dict[str, float]:
        if self.family == "thermal_split":
            return {"coupling": 0.06, "leak": 0.11, "phase": 0.08}
        if self.family == "coupled_oscillator":
            return {"omega": 1.12, "damping": 0.10, "charge": 0.16}
        return {"drive": 0.18, "decay": 0.09, "mix": 0.12}

    def _drift_params(self, params: dict[str, float]) -> dict[str, float]:
        shifted = {}
        for key, value in params.items():
            shifted[key] = float(value * (1.0 + self.rng.normal(0.12, 0.04)))
        return shifted

    def _step_state(self, state: np.ndarray, params: dict[str, float]) -> np.ndarray:
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
