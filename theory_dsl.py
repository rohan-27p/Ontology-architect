"""Structured theory DSL rendering for low-invalidity exploration."""

from __future__ import annotations

import ast
import json
from typing import Any


DSL_MARKERS = {"theory_dsl", "ontology_theory"}
MAX_STATE_VARS = 12
MAX_SENSOR_OUTPUTS = 16
MAX_EXPR_DEPTH = 8
MAX_EXPR_NODES = 80


class TheoryDSLValidationError(ValueError):
    """Raised when a structured theory spec is malformed."""


def parse_theory_dsl(text: str) -> dict[str, Any] | None:
    """Return a structured theory spec if text is a DSL object, else None."""

    stripped = text.strip()
    if not stripped or not stripped.startswith("{"):
        return None
    try:
        candidate = json.loads(stripped)
    except json.JSONDecodeError:
        try:
            candidate = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return None
    if not isinstance(candidate, dict):
        return None
    kind = str(candidate.get("type", candidate.get("kind", ""))).lower()
    if candidate.get("dsl_version") == 1 or kind in DSL_MARKERS:
        return candidate
    return None


def is_theory_dsl(text: str) -> bool:
    return parse_theory_dsl(text) is not None


def render_theory_module(text: str) -> str:
    spec = parse_theory_dsl(text)
    if spec is None:
        return text
    normalized = normalize_theory_spec(spec)
    return _EXECUTOR_TEMPLATE.replace("__THEORY_SPEC__", repr(normalized))


def normalize_theory_spec(spec: dict[str, Any]) -> dict[str, Any]:
    if int(spec.get("dsl_version", 1)) != 1:
        raise TheoryDSLValidationError("Only theory DSL version 1 is supported")

    state = _string_list(spec.get("state") or spec.get("latents") or [])
    dynamics = _dict(spec.get("dynamics", {}), "dynamics")
    observations = _dict(spec.get("observations", spec.get("sensors", {})), "observations")
    init = _dict(spec.get("init", spec.get("initial_state", {})), "init")
    fit = _dict(spec.get("fit", {}), "fit")
    integrator = _dict(spec.get("integrator", {}), "integrator")

    if not state:
        state = sorted(set(dynamics) | set(_expression_variables(dynamics)) | set(_expression_variables(observations)) | set(init))
    if not state:
        raise TheoryDSLValidationError("DSL must define at least one state variable")
    if len(state) > MAX_STATE_VARS:
        raise TheoryDSLValidationError(f"DSL state exceeds {MAX_STATE_VARS} variables")
    if len(observations) > MAX_SENSOR_OUTPUTS:
        raise TheoryDSLValidationError(f"DSL observations exceed {MAX_SENSOR_OUTPUTS} outputs")

    state_set = set(state)
    for name, expr in dynamics.items():
        if str(name) not in state_set:
            raise TheoryDSLValidationError(f"Dynamic variable '{name}' is not in state")
        _validate_expression(expr, state_set)
    for expr in observations.values():
        _validate_expression(expr, state_set)

    normalized = {
        "dsl_version": 1,
        "name": str(spec.get("name", "structured theory"))[:120],
        "state": state,
        "init": init,
        "dynamics": dynamics,
        "observations": observations,
        "fit": {
            "lookback": _bounded_int(fit.get("lookback", 6), 2, 32),
            "trend_weight": _bounded_float(fit.get("trend_weight", 0.0), -4.0, 4.0),
            "drift_threshold": _bounded_float(fit.get("drift_threshold", 0.35), 0.0, 10.0),
            "anomaly_threshold": _bounded_float(fit.get("anomaly_threshold", 0.45), 0.0, 10.0),
        },
        "integrator": {
            "dt": _bounded_float(integrator.get("dt", spec.get("dt", 1.0)), 1e-4, 5.0),
            "substeps": _bounded_int(integrator.get("substeps", 1), 1, 8),
        },
        "noise": _bounded_float(spec.get("noise", spec.get("sigma", 0.12)), 1e-6, 10.0),
    }
    return normalized


def dsl_complexity(text: str) -> float | None:
    spec = parse_theory_dsl(text)
    if spec is None:
        return None
    normalized = normalize_theory_spec(spec)
    expression_nodes = 0
    for expr in normalized["dynamics"].values():
        expression_nodes += _expression_node_count(expr)
    for expr in normalized["observations"].values():
        expression_nodes += _expression_node_count(expr)
    return float(
        len(normalized["state"]) * 3
        + len(normalized["dynamics"]) * 4
        + len(normalized["observations"]) * 3
        + expression_nodes
    )


def dsl_signature(text: str) -> dict[str, int] | None:
    spec = parse_theory_dsl(text)
    if spec is None:
        return None
    normalized = normalize_theory_spec(spec)
    counts: dict[str, int] = {}
    for name in normalized["state"]:
        counts[f"state:{name}"] = counts.get(f"state:{name}", 0) + 1
    for section in ("dynamics", "observations"):
        for name, expr in normalized[section].items():
            counts[f"{section}:{name}"] = counts.get(f"{section}:{name}", 0) + 1
            _count_expression_signature(expr, counts)
    return counts


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    names = []
    for item in value:
        name = str(item)
        if name.isidentifier() and not name.startswith("_"):
            names.append(name)
    return names


def _dict(value: Any, label: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TheoryDSLValidationError(f"DSL field '{label}' must be an object")
    return {str(key): item for key, item in value.items()}


def _validate_expression(expr: Any, variables: set[str], depth: int = 0) -> None:
    if depth > MAX_EXPR_DEPTH:
        raise TheoryDSLValidationError("DSL expression is too deep")
    if _expression_node_count(expr) > MAX_EXPR_NODES:
        raise TheoryDSLValidationError("DSL expression is too large")
    if isinstance(expr, int | float):
        return
    if isinstance(expr, str):
        if expr not in variables:
            raise TheoryDSLValidationError(f"Unknown variable '{expr}' in DSL expression")
        return
    if not isinstance(expr, dict):
        raise TheoryDSLValidationError("DSL expressions must be numbers, variables, or expression objects")
    if "var" in expr:
        var = str(expr["var"])
        if var not in variables:
            raise TheoryDSLValidationError(f"Unknown variable '{var}' in DSL expression")
    if "linear" in expr:
        _validate_linear(expr["linear"], variables)
    if "terms" in expr:
        _validate_linear(expr, variables)
    for key in ("add", "mul"):
        if key in expr:
            if not isinstance(expr[key], list):
                raise TheoryDSLValidationError(f"DSL '{key}' expression must be a list")
            for item in expr[key]:
                _validate_expression(item, variables, depth + 1)
    for key in ("sin", "cos", "tanh", "neg", "exp", "abs", "sqrt"):
        if key in expr:
            _validate_expression(expr[key], variables, depth + 1)
    if "pow" in expr:
        if not isinstance(expr["pow"], list) or len(expr["pow"]) != 2:
            raise TheoryDSLValidationError("DSL 'pow' expression must be [base, exponent]")
        for item in expr["pow"]:
            _validate_expression(item, variables, depth + 1)


def _validate_linear(expr: Any, variables: set[str]) -> None:
    if not isinstance(expr, dict):
        raise TheoryDSLValidationError("DSL linear expression must be an object")
    terms = expr.get("terms", expr)
    if not isinstance(terms, dict):
        raise TheoryDSLValidationError("DSL linear terms must be an object")
    for name, coef in terms.items():
        if name == "bias":
            continue
        if str(name) not in variables:
            raise TheoryDSLValidationError(f"Unknown variable '{name}' in DSL linear expression")
        _bounded_float(coef, -1e6, 1e6)
    if "bias" in expr:
        _bounded_float(expr["bias"], -1e6, 1e6)


def _expression_variables(value: Any) -> set[str]:
    variables: set[str] = set()
    if isinstance(value, str):
        variables.add(value)
    elif isinstance(value, dict):
        if "var" in value:
            variables.add(str(value["var"]))
        for item in value.values():
            variables.update(_expression_variables(item))
    elif isinstance(value, list):
        for item in value:
            variables.update(_expression_variables(item))
    return variables


def _expression_node_count(expr: Any) -> int:
    if isinstance(expr, int | float | str):
        return 1
    if isinstance(expr, list):
        return 1 + sum(_expression_node_count(item) for item in expr)
    if isinstance(expr, dict):
        return 1 + sum(_expression_node_count(item) for item in expr.values())
    return 1


def _count_expression_signature(expr: Any, counts: dict[str, int]) -> None:
    if isinstance(expr, int | float):
        counts["expr:const"] = counts.get("expr:const", 0) + 1
    elif isinstance(expr, str):
        counts[f"expr:var:{expr}"] = counts.get(f"expr:var:{expr}", 0) + 1
    elif isinstance(expr, list):
        for item in expr:
            _count_expression_signature(item, counts)
    elif isinstance(expr, dict):
        for key, value in expr.items():
            counts[f"expr:{key}"] = counts.get(f"expr:{key}", 0) + 1
            _count_expression_signature(value, counts)


def _bounded_int(value: Any, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise TheoryDSLValidationError(f"Expected integer value, got {value!r}") from exc
    return max(minimum, min(maximum, parsed))


def _bounded_float(value: Any, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise TheoryDSLValidationError(f"Expected numeric value, got {value!r}") from exc
    return max(minimum, min(maximum, parsed))


_EXECUTOR_TEMPLATE = r'''
import math

THEORY_SPEC = __THEORY_SPEC__


class Theory:
    def __init__(self):
        self.spec = THEORY_SPEC
        self.state = {}
        self.last = {}
        self.sensor_names = []
        self.trend = {}
        self.residual_scale = 0.0
        self.last_predictions = []

    def fit(self, history):
        self.sensor_names = list(history[-1]["sensors"]) if history else []
        self.last = dict(history[-1]["sensors"]) if history else {}
        lookback = int(self.spec["fit"]["lookback"])
        recent = history[-lookback:] if history else []
        self.trend = self._fit_trend(recent)
        self.residual_scale = self._fit_residual_scale(recent)
        self.state = self._initial_state()
        return {
            "dsl_version": self.spec["dsl_version"],
            "state": list(self.spec["state"]),
            "records": len(history),
            "residual_scale": self.residual_scale,
        }

    def predict(self, window):
        steps = int(window["steps"])
        sensor_names = list(window.get("sensor_names") or self.sensor_names)
        state = dict(self.state)
        rows = []
        for _ in range(steps):
            state = self._integrate(state)
            sensors = self._project(state, sensor_names)
            rows.append({"sensors": sensors, "anomaly_prob": self._anomaly_probability()})
        self.last_predictions = rows
        return rows

    def log_prob(self, observations):
        if not observations:
            return 0.0
        preds = self.last_predictions
        if len(preds) < len(observations):
            preds = self.predict({"steps": len(observations), "sensor_names": list(observations[0]["sensors"])})
        sigma = max(float(self.spec["noise"]), 1e-6)
        total = 0.0
        count = 0
        for index, row in enumerate(observations):
            predicted = preds[min(index, len(preds) - 1)]["sensors"]
            for name, observed in row["sensors"].items():
                error = float(observed) - float(predicted.get(name, 0.0))
                total += -0.5 * (error / sigma) ** 2
                count += 1
        return total / max(count, 1)

    def detect_drift(self, history):
        if len(history) < 4:
            return False
        lookback = min(int(self.spec["fit"]["lookback"]), len(history) // 2)
        old = history[-2 * lookback : -lookback]
        recent = history[-lookback:]
        old_delta = self._window_delta(old)
        recent_delta = self._window_delta(recent)
        threshold = float(self.spec["fit"]["drift_threshold"])
        return abs(recent_delta - old_delta) > threshold

    def describe(self):
        return "Structured Theory DSL v1: " + str(self.spec.get("name", "unnamed"))

    def _initial_state(self):
        state = {}
        init = self.spec.get("init", {})
        for name in self.spec["state"]:
            rule = init.get(name, name)
            if isinstance(rule, dict):
                sensor = str(rule.get("sensor", name))
                scale = float(rule.get("scale", 1.0))
                bias = float(rule.get("bias", 0.0))
                state[name] = scale * float(self.last.get(sensor, 0.0)) + bias
            elif isinstance(rule, (int, float)):
                state[name] = float(rule)
            else:
                state[name] = float(self.last.get(str(rule), self.last.get(name, 0.0)))
        return state

    def _fit_trend(self, rows):
        if len(rows) < 2:
            return {name: 0.0 for name in self.sensor_names}
        first = rows[0]["sensors"]
        last = rows[-1]["sensors"]
        span = max(len(rows) - 1, 1)
        return {name: (float(last.get(name, 0.0)) - float(first.get(name, 0.0))) / span for name in last}

    def _fit_residual_scale(self, rows):
        if len(rows) < 2:
            return 0.0
        errors = []
        for before, after in zip(rows[:-1], rows[1:]):
            for name, value in after["sensors"].items():
                expected = float(before["sensors"].get(name, value)) + self.trend.get(name, 0.0)
                errors.append(abs(float(value) - expected))
        return sum(errors) / max(len(errors), 1)

    def _window_delta(self, rows):
        if len(rows) < 2:
            return 0.0
        total = 0.0
        count = 0
        for before, after in zip(rows[:-1], rows[1:]):
            for name, value in after["sensors"].items():
                total += abs(float(value) - float(before["sensors"].get(name, value)))
                count += 1
        return total / max(count, 1)

    def _integrate(self, state):
        dt = float(self.spec["integrator"]["dt"])
        substeps = int(self.spec["integrator"]["substeps"])
        step_dt = dt / max(substeps, 1)
        for _ in range(max(substeps, 1)):
            deriv = self._derivative(state)
            state = {name: self._clip(value + step_dt * deriv.get(name, 0.0)) for name, value in state.items()}
        return state

    def _derivative(self, state):
        dynamics = self.spec.get("dynamics", {})
        trend_weight = float(self.spec["fit"]["trend_weight"])
        deriv = {}
        for name in self.spec["state"]:
            value = self._eval(dynamics.get(name, 0.0), state)
            if name in self.trend:
                value += trend_weight * self.trend.get(name, 0.0)
            deriv[name] = self._clip(value)
        return deriv

    def _project(self, state, sensor_names):
        observations = self.spec.get("observations", {})
        sensors = {}
        for name in sensor_names:
            if name in observations:
                sensors[name] = self._clip(self._eval(observations[name], state))
            elif name in state:
                sensors[name] = self._clip(state[name])
            else:
                sensors[name] = self._clip(float(self.last.get(name, 0.0)) + self.trend.get(name, 0.0))
        return sensors

    def _anomaly_probability(self):
        threshold = max(float(self.spec["fit"]["anomaly_threshold"]), 1e-6)
        return 0.65 if self.residual_scale > threshold else 0.08

    def _eval(self, expr, state):
        if isinstance(expr, (int, float)):
            return float(expr)
        if isinstance(expr, str):
            return float(state.get(expr, 0.0))
        if not isinstance(expr, dict):
            return 0.0
        if "const" in expr:
            return float(expr["const"])
        if "var" in expr:
            return float(state.get(str(expr["var"]), 0.0))
        if "linear" in expr:
            return self._linear(expr["linear"], state)
        if "terms" in expr:
            return self._linear(expr, state)
        if "add" in expr:
            return sum(self._eval(item, state) for item in expr["add"])
        if "mul" in expr:
            value = 1.0
            for item in expr["mul"]:
                value *= self._eval(item, state)
            return value
        if "sin" in expr:
            return math.sin(self._eval(expr["sin"], state))
        if "cos" in expr:
            return math.cos(self._eval(expr["cos"], state))
        if "tanh" in expr:
            return math.tanh(self._eval(expr["tanh"], state))
        if "neg" in expr:
            return -self._eval(expr["neg"], state)
        if "exp" in expr:
            v = self._eval(expr["exp"], state)
            return math.exp(max(-20.0, min(20.0, v)))
        if "abs" in expr:
            return abs(self._eval(expr["abs"], state))
        if "sqrt" in expr:
            return math.sqrt(max(0.0, self._eval(expr["sqrt"], state)))
        if "pow" in expr:
            base = self._eval(expr["pow"][0], state)
            exp = self._eval(expr["pow"][1], state)
            return self._clip(math.pow(abs(base) + 1e-12, max(-10.0, min(10.0, exp))))
        return 0.0

    def _linear(self, expr, state):
        terms = expr.get("terms", expr)
        value = float(expr.get("bias", terms.get("bias", 0.0)))
        for name, coef in terms.items():
            if name == "bias":
                continue
            value += float(coef) * float(state.get(str(name), 0.0))
        return value

    def _clip(self, value):
        if value != value:
            return 0.0
        return max(-1e6, min(1e6, float(value)))
'''
