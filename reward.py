"""Environment-owned scoring for submitted theories."""

from __future__ import annotations

import ast
from collections import Counter
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
    weak_paradigm_shift: bool
    details: dict[str, Any]


@dataclass(frozen=True)
class ComplexityBreakdown:
    mode: str
    score: float
    ast_nodes: int
    import_count: int
    class_count: int
    function_count: int
    parameter_count: int
    branch_count: int
    self_state_assignments: int
    parse_ok: bool
    fallback_chars: int = 0

    def to_dict(self) -> dict[str, float | int | bool | str]:
        return {
            "mdl_complexity_mode": self.mode,
            "mdl_complexity": self.score,
            "ast_nodes": self.ast_nodes,
            "import_count": self.import_count,
            "class_count": self.class_count,
            "function_count": self.function_count,
            "parameter_count": self.parameter_count,
            "branch_count": self.branch_count,
            "self_state_assignments": self.self_state_assignments,
            "complexity_parse_ok": self.parse_ok,
            "fallback_chars": self.fallback_chars,
        }


def score_theory(
    result: SandboxResult,
    future: list[SensorRecord],
    theory_module: str,
    config: RewardConfig,
    paradigm_shift_claim: bool = False,
    previous_theory_module: str | None = None,
) -> RewardBreakdown:
    complexity = theory_complexity(theory_module, config.mdl_complexity_mode)
    mdl_penalty = config.mdl_lambda * complexity.score
    if len(theory_module) > config.max_theory_chars:
        return RewardBreakdown(
            reward=config.execution_error_penalty,
            log_likelihood=float("-inf"),
            mdl_penalty=mdl_penalty,
            anomaly_bonus=0.0,
            drift_bonus=0.0,
            execution_ok=False,
            missed_anomaly=window_has_anomaly(future),
            false_paradigm_shift=False,
            weak_paradigm_shift=False,
            details={
                "error": "theory module exceeded max_theory_chars",
                "theory_chars": len(theory_module),
                **complexity.to_dict(),
            },
        )
    if not result.ok:
        return RewardBreakdown(
            reward=config.execution_error_penalty,
            log_likelihood=float("-inf"),
            mdl_penalty=mdl_penalty,
            anomaly_bonus=0.0,
            drift_bonus=0.0,
            execution_ok=False,
            missed_anomaly=window_has_anomaly(future),
            false_paradigm_shift=False,
            weak_paradigm_shift=False,
            details={
                "error": result.error,
                "theory_chars": len(theory_module),
                **complexity.to_dict(),
            },
        )

    log_likelihood = gaussian_log_likelihood(result.predictions, future, config.prediction_sigma)
    anomaly_bonus, missed_anomaly = anomaly_score(result.predictions, future, config.anomaly_bonus)
    structural_delta = theory_structural_distance(theory_module, previous_theory_module)
    drift_bonus, false_paradigm_shift, weak_paradigm_shift = drift_score(
        result,
        future,
        paradigm_shift_claim,
        config.drift_bonus,
        config.false_paradigm_shift_penalty,
        structural_delta,
        config.paradigm_shift_min_structural_delta,
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
        weak_paradigm_shift=weak_paradigm_shift,
        details={
            "future_records": len(future),
            "prediction_records": len(result.predictions) if isinstance(result.predictions, list) else 0,
            "reported_log_prob": result.reported_log_prob,
            "theory_chars": len(theory_module),
            "runtime_ms": result.runtime_ms,
            "structural_delta": structural_delta,
            **complexity.to_dict(),
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
    structural_delta: float = 1.0,
    min_structural_delta: float = 0.0,
) -> tuple[float, bool, bool]:
    drifted = window_has_drift(future)
    claimed = bool(paradigm_shift_claim or result.drift_detected)
    if drifted and claimed:
        if min_structural_delta <= 0.0:
            return bonus, False, False
        change_scale = min(1.0, max(0.0, structural_delta) / min_structural_delta)
        return bonus * change_scale, False, change_scale < 1.0
    if not drifted and claimed:
        return -false_claim_penalty, True, False
    return 0.0, False, False


def theory_complexity(theory_module: str, mode: str = "ast") -> ComplexityBreakdown:
    if mode == "chars":
        return ComplexityBreakdown(
            mode=mode,
            score=float(len(theory_module)),
            ast_nodes=0,
            import_count=0,
            class_count=0,
            function_count=0,
            parameter_count=0,
            branch_count=0,
            self_state_assignments=0,
            parse_ok=True,
            fallback_chars=len(theory_module),
        )
    try:
        tree = ast.parse(theory_module)
    except SyntaxError:
        fallback_score = max(1.0, len(theory_module) / 80.0)
        return ComplexityBreakdown(
            mode=mode,
            score=float(fallback_score),
            ast_nodes=0,
            import_count=0,
            class_count=0,
            function_count=0,
            parameter_count=0,
            branch_count=0,
            self_state_assignments=0,
            parse_ok=False,
            fallback_chars=len(theory_module),
        )

    nodes = list(ast.walk(tree))
    import_count = sum(isinstance(node, ast.Import | ast.ImportFrom) for node in nodes)
    class_count = sum(isinstance(node, ast.ClassDef) for node in nodes)
    function_count = sum(isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) for node in nodes)
    branch_count = sum(isinstance(node, ast.If | ast.For | ast.While | ast.Try | ast.With | ast.BoolOp) for node in nodes)
    arg_count = sum(len(node.args.args) + len(node.args.kwonlyargs) for node in nodes if isinstance(node, ast.FunctionDef))
    numeric_constants = sum(
        isinstance(node, ast.Constant) and isinstance(node.value, int | float)
        for node in nodes
    )
    self_state_assignments = sum(_self_assignment_count(node) for node in nodes)
    parameter_count = int(arg_count + numeric_constants + self_state_assignments)
    score = (
        len(nodes)
        + 2.0 * import_count
        + 3.0 * class_count
        + 2.0 * function_count
        + 1.5 * parameter_count
        + 2.0 * branch_count
    )
    return ComplexityBreakdown(
        mode=mode,
        score=float(score),
        ast_nodes=len(nodes),
        import_count=int(import_count),
        class_count=int(class_count),
        function_count=int(function_count),
        parameter_count=parameter_count,
        branch_count=int(branch_count),
        self_state_assignments=int(self_state_assignments),
        parse_ok=True,
    )


def theory_structural_distance(theory_module: str, previous_theory_module: str | None) -> float:
    if not previous_theory_module:
        return 1.0
    current = _ast_feature_counts(theory_module)
    previous = _ast_feature_counts(previous_theory_module)
    if not current and not previous:
        return 0.0 if theory_module == previous_theory_module else 1.0
    keys = set(current) | set(previous)
    numerator = sum(abs(current.get(key, 0) - previous.get(key, 0)) for key in keys)
    denominator = sum(max(current.get(key, 0), previous.get(key, 0)) for key in keys)
    if denominator <= 0:
        return 0.0
    return float(max(0.0, min(1.0, numerator / denominator)))


def prediction_diagnostics(
    predictions: Any,
    future: list[SensorRecord],
    sigma: float,
    divergence_sigma: float = 2.0,
) -> dict[str, Any]:
    prediction_rows = _as_prediction_rows(predictions)
    threshold = max(1e-9, abs(float(sigma)) * float(divergence_sigma))
    per_sensor_errors: dict[str, list[dict[str, float]]] = {}
    first_divergence: dict[str, Any] | None = None

    for index, record in enumerate(future):
        row = prediction_rows[min(index, len(prediction_rows) - 1)] if prediction_rows else {"sensors": {}}
        sensors = row["sensors"]
        for name, observed in record.sensors.items():
            predicted = float(sensors.get(name, 0.0))
            error = float(observed - predicted)
            entry = {
                "t": float(record.t),
                "observed": float(observed),
                "predicted": predicted,
                "error": error,
                "abs_error": abs(error),
            }
            per_sensor_errors.setdefault(str(name), []).append(entry)
            if first_divergence is None and abs(error) > threshold:
                first_divergence = {
                    "t": float(record.t),
                    "sensor": str(name),
                    "abs_error": abs(error),
                    "threshold": threshold,
                }

    per_sensor = {}
    worst_sensor = None
    worst_mae = -1.0
    for name, errors in per_sensor_errors.items():
        if not errors:
            continue
        abs_errors = [item["abs_error"] for item in errors]
        signed_errors = [item["error"] for item in errors]
        mae = sum(abs_errors) / len(abs_errors)
        rmse = math.sqrt(sum(value * value for value in signed_errors) / len(signed_errors))
        divergences = [item for item in errors if item["abs_error"] > threshold]
        per_sensor[name] = {
            "mae": mae,
            "rmse": rmse,
            "bias": sum(signed_errors) / len(signed_errors),
            "max_abs_error": max(abs_errors),
            "first_divergence_t": divergences[0]["t"] if divergences else None,
        }
        if mae > worst_mae:
            worst_sensor = name
            worst_mae = mae

    return {
        "divergence_threshold": threshold,
        "divergence_sigma": divergence_sigma,
        "per_sensor": per_sensor,
        "worst_sensor": worst_sensor,
        "first_divergence": first_divergence,
        "prediction_records": len(prediction_rows),
        "future_records": len(future),
        "missing_prediction_rows": max(0, len(future) - len(prediction_rows)),
    }


def _self_assignment_count(node: ast.AST) -> int:
    targets: list[ast.AST]
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    elif isinstance(node, ast.AnnAssign):
        targets = [node.target]
    elif isinstance(node, ast.AugAssign):
        targets = [node.target]
    else:
        return 0
    return sum(
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id == "self"
        for target in targets
    )


def _ast_feature_counts(theory_module: str) -> Counter[str]:
    try:
        tree = ast.parse(theory_module)
    except SyntaxError:
        return Counter()
    counts: Counter[str] = Counter()
    for node in ast.walk(tree):
        counts[f"node:{type(node).__name__}"] += 1
        if isinstance(node, ast.ClassDef):
            counts[f"class:{node.name}"] += 1
        elif isinstance(node, ast.FunctionDef):
            counts[f"function:{node.name}"] += 1
        elif isinstance(node, ast.Call):
            counts[f"call:{_call_name(node.func)}"] += 1
        elif isinstance(node, ast.Attribute):
            counts[f"attribute:{node.attr}"] += 1
        elif isinstance(node, ast.BinOp):
            counts[f"binop:{type(node.op).__name__}"] += 1
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                counts[f"compare:{type(op).__name__}"] += 1
    return counts


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return type(node).__name__


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
