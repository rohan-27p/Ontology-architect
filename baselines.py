"""Baseline and teacher theory modules used by smoke tests and benchmarks."""

from __future__ import annotations


STATIC_THEORY = r'''
import math


class Theory:
    def __init__(self):
        self.last = {}
        self.sensor_names = []

    def fit(self, history):
        if history:
            self.last = dict(history[-1]["sensors"])
            self.sensor_names = list(self.last)
        return {"records": len(history)}

    def predict(self, window):
        steps = int(window["steps"])
        return [{"sensors": dict(self.last), "anomaly_prob": 0.05} for _ in range(steps)]

    def log_prob(self, observations):
        return -float(len(observations))

    def describe(self):
        return "Static persistence model: predicts each sensor remains at its last observed value."
'''


LINEAR_TREND_THEORY = r'''
import math


class Theory:
    def __init__(self):
        self.last = {}
        self.slope = {}

    def fit(self, history):
        if not history:
            return {"records": 0}
        self.last = dict(history[-1]["sensors"])
        if len(history) < 3:
            self.slope = {name: 0.0 for name in self.last}
            return {"records": len(history)}
        prev = history[-3]["sensors"]
        self.slope = {name: (self.last[name] - prev.get(name, self.last[name])) / 2.0 for name in self.last}
        return {"records": len(history)}

    def predict(self, window):
        rows = []
        for step in range(1, int(window["steps"]) + 1):
            sensors = {name: value + self.slope.get(name, 0.0) * step for name, value in self.last.items()}
            rows.append({"sensors": sensors, "anomaly_prob": 0.08})
        return rows

    def log_prob(self, observations):
        return -0.5 * float(len(observations))

    def detect_drift(self, history):
        if len(history) < 8:
            return False
        names = list(history[-1]["sensors"])
        old = 0.0
        recent = 0.0
        for name in names:
            old += abs(history[-5]["sensors"][name] - history[-8]["sensors"][name])
            recent += abs(history[-1]["sensors"][name] - history[-4]["sensors"][name])
        return recent > 2.0 * max(old, 1e-6)

    def describe(self):
        return "Linear trend model with a residual-change drift heuristic."
'''


ANOMALY_AWARE_TEACHER = r'''
import math
import statistics


class Theory:
    def __init__(self):
        self.last = {}
        self.slope = {}
        self.volatility = 0.0

    def fit(self, history):
        if not history:
            return {"records": 0}
        self.last = dict(history[-1]["sensors"])
        names = list(self.last)
        if len(history) < 5:
            self.slope = {name: 0.0 for name in names}
            self.volatility = 0.0
            return {"records": len(history)}
        self.slope = {}
        deltas = []
        for name in names:
            series = [row["sensors"][name] for row in history[-8:]]
            step_deltas = [b - a for a, b in zip(series[:-1], series[1:])]
            self.slope[name] = statistics.mean(step_deltas)
            deltas.extend(abs(value) for value in step_deltas)
        self.volatility = statistics.mean(deltas) if deltas else 0.0
        return {"records": len(history), "volatility": self.volatility}

    def predict(self, window):
        rows = []
        anomaly_prob = 0.65 if self.volatility > 0.35 else 0.10
        for step in range(1, int(window["steps"]) + 1):
            sensors = {}
            for name, value in self.last.items():
                curvature = 0.02 * math.sin(step)
                sensors[name] = value + self.slope.get(name, 0.0) * step + curvature
            rows.append({"sensors": sensors, "anomaly_prob": anomaly_prob})
        return rows

    def log_prob(self, observations):
        return -0.25 * float(len(observations))

    def detect_drift(self, history):
        return self.volatility > 0.45

    def describe(self):
        return "Teacher model: local trend, weak curvature, anomaly and drift heuristics."
'''


BASELINES = {
    "static": STATIC_THEORY,
    "linear": LINEAR_TREND_THEORY,
    "oracle": ANOMALY_AWARE_TEACHER,
    "teacher": ANOMALY_AWARE_TEACHER,
}


def get_baseline(name: str) -> str:
    try:
        return BASELINES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown baseline '{name}'. Choose from {sorted(BASELINES)}") from exc
