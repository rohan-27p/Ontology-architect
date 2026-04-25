"""Environment-owned scoring for submitted theories."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

try:
    from .config import RewardConfig
    from .sandbox import SandboxResult
    from .universe import SensorRecord, window_has_anomaly, window_has_drift
except ImportError:  # pragma: no cover
    from config import RewardConfig
    from sandbox import SandboxResult
    from universe import SensorRecord, window_has_anomaly, window_has_drift


@dataclass(frozen=True)
class RewardBreakdown:
    reward: float
    log_likelihood: float
    mdl_penalty: float
    anomaly_bonus: float
    drift_bonus: float
    execution_ok: bool
    missed_anomaly: bool
    false_paradigm_shift: bool
    details: dict[str, float | int | bool | str]


def score_theory(
    result: SandboxResult,
    future: list[SensorRecord],
    theory_module: str,
    config: RewardConfig,
    paradigm_shift_claim: bool = False,
) -> RewardBreakdown:
    if len(theory_module) > config.max_theory_chars:
        return RewardBreakdown(
            reward=config.execution_error_penalty,
            log_likelihood=float("-inf"),
            mdl_penalty=config.mdl_lambda * len(theory_module),
            anomaly_bonus=0.0,
            drift_bonus=0.0,
            execution_ok=False,
            missed_anomaly=window_has_anomaly(future),
            false_paradigm_shift=False,
            details={"error": "theory module exceeded max_theory_chars"},
        )
    if not result.ok:
        return RewardBreakdown(
            reward=config.execution_error_penalty,
            log_likelihood=float("-inf"),
            mdl_penalty=config.mdl_lambda * len(theory_module),
            anomaly_bonus=0.0,
            drift_bonus=0.0,
            execution_ok=False,
            missed_anomaly=window_has_anomaly(future),
            false_paradigm_shift=False,
            details={"error": result.error},
        )

    log_likelihood = gaussian_log_likelihood(result.predictions, future, config.prediction_sigma)
    mdl_penalty = config.mdl_lambda * len(theory_module)
    anomaly_bonus, missed_anomaly = anomaly_score(result.predictions, future, config.anomaly_bonus)
    drift_bonus, false_paradigm_shift = drift_score(
        result,
        future,
        paradigm_shift_claim,
        config.drift_bonus,
        config.false_paradigm_shift_penalty,
    )
    reward = log_likelihood - mdl_penalty + anomaly_bonus + drift_bonus
    return RewardBreakdown(
        reward=float(reward),
        log_likelihood=float(log_likelihood),
        mdl_penalty=float(mdl_penalty),
        anomaly_bonus=float(anomaly_bonus),
        drift_bonus=float(drift_bonus),
        execution_ok=True,
        missed_anomaly=missed_anomaly,
        false_paradigm_shift=false_paradigm_shift,
        details={
            "future_records": len(future),
            "prediction_records": len(result.predictions) if isinstance(result.predictions, list) else 0,
            "reported_log_prob": result.reported_log_prob,
            "theory_chars": len(theory_module),
            "runtime_ms": result.runtime_ms,
        },
    )


def gaussian_log_likelihood(predictions: Any, future: list[SensorRecord], sigma: float) -> float:
    prediction_rows = _as_prediction_rows(predictions)
    if not prediction_rows:
        return -1e6

    sigma = max(1e-6, sigma)
    constant = -math.log(sigma * math.sqrt(2.0 * math.pi))
    log_likelihood = 0.0
    count = 0
    for index, record in enumerate(future):
        row = prediction_rows[min(index, len(prediction_rows) - 1)]
        sensors = row["sensors"]
        for name, observed in record.sensors.items():
            predicted = float(sensors.get(name, 0.0))
            error = observed - predicted
            log_likelihood += constant - 0.5 * (error / sigma) ** 2
            count += 1
    if count == 0:
        return -1e6
    return max(-1e6, log_likelihood / count)


def anomaly_score(predictions: Any, future: list[SensorRecord], bonus: float) -> tuple[float, bool]:
    if not window_has_anomaly(future):
        return 0.0, False
    prediction_rows = _as_prediction_rows(predictions)
    detected = False
    for index, record in enumerate(future):
        if not record.anomaly:
            continue
        if not prediction_rows:
            break
        row = prediction_rows[min(index, len(prediction_rows) - 1)]
        detected = detected or float(row.get("anomaly_prob", 0.0)) >= 0.5
    return (bonus if detected else 0.0), not detected


def drift_score(
    result: SandboxResult,
    future: list[SensorRecord],
    paradigm_shift_claim: bool,
    bonus: float,
    false_claim_penalty: float,
) -> tuple[float, bool]:
    drifted = window_has_drift(future)
    claimed = bool(paradigm_shift_claim or result.drift_detected)
    if drifted and claimed:
        return bonus, False
    if not drifted and claimed:
        return -false_claim_penalty, True
    return 0.0, False


def _as_prediction_rows(predictions: Any) -> list[dict[str, Any]]:
    if not isinstance(predictions, list):
        return []
    rows = []
    for item in predictions:
        if not isinstance(item, dict):
            rows.append({"sensors": {}, "anomaly_prob": 0.0})
            continue
        sensors = item.get("sensors")
        if isinstance(sensors, dict):
            numeric_sensors = {
                str(name): float(value)
                for name, value in sensors.items()
                if isinstance(value, int | float)
            }
        else:
            numeric_sensors = {
                str(name): float(value)
                for name, value in item.items()
                if name not in {"anomaly_prob", "anomaly_probability"} and isinstance(value, int | float)
            }
        rows.append(
            {
                "sensors": numeric_sensors,
                "anomaly_prob": float(item.get("anomaly_prob", item.get("anomaly_probability", 0.0))),
            }
        )
    return rows
