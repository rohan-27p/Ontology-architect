"""Core Ontology Architect OpenEnv environment."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import State
except ImportError:  # pragma: no cover - older OpenEnv compatibility.
    from openenv_core.env_server.interfaces import Environment
    from openenv_core.env_server.types import State

try:
    from ontology_architect.config import ExperimentConfig
    from ontology_architect.models import OntologyArchitectAction, OntologyArchitectObservation
    from ontology_architect.reward import RewardBreakdown, prediction_diagnostics, score_theory
    from ontology_architect.sandbox import SandboxResult, TheorySandbox
    from ontology_architect.universe import ProceduralAlienUniverse, format_sensor_log, window_has_anomaly, window_has_drift
except ImportError:  # pragma: no cover - local server.app execution from repository root.
    from config import ExperimentConfig
    from models import OntologyArchitectAction, OntologyArchitectObservation
    from reward import RewardBreakdown, prediction_diagnostics, score_theory
    from sandbox import SandboxResult, TheorySandbox
    from universe import ProceduralAlienUniverse, format_sensor_log, window_has_anomaly, window_has_drift


class OntologyArchitectEnvironment(Environment):
    """Environment where actions are Python theories for hidden sensor dynamics."""

    def __init__(self, config: ExperimentConfig | None = None):
        self.config = config or ExperimentConfig()
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count = 0
        self._records = []
        self._cursor = 0
        self._universe = self._build_universe(self.config.universe.seed)
        self._sandbox = TheorySandbox(self.config.sandbox)
        self._last_execution_output = "No theory has been submitted yet."
        self._peer_reviews: list[str] = []
        self._theory_lineage: list[dict] = []
        self._last_theory_module: str | None = None
        self._last_metrics: dict = {}

    def reset(self) -> OntologyArchitectObservation:
        self._state = State(episode_id=str(uuid4()), step_count=0)
        self._reset_count += 1
        episode_seed = self.config.universe.seed + self._reset_count * 997
        self._universe = self._build_universe(episode_seed)

        total_records = (
            self.config.universe.observation_window
            + (self.config.universe.max_steps + 1) * self.config.universe.future_window
            + 1
        )
        self._records = self._universe.generate(total_records)
        self._cursor = self.config.universe.observation_window
        self._last_execution_output = "No theory has been submitted yet."
        self._peer_reviews = [
            "No hypotheses have been falsified yet. Prefer compact theories that predict future windows."
        ]
        self._theory_lineage = []
        self._last_theory_module = None
        self._last_metrics = {}
        return self._make_observation(done=False, reward=0.0)

    def step(self, action: OntologyArchitectAction) -> OntologyArchitectObservation:  # type: ignore[override]
        self._state.step_count += 1
        history = self._records[: self._cursor]
        future = self._records[self._cursor : self._cursor + self.config.universe.future_window]

        if not future:
            return self._make_observation(done=True, reward=0.0)

        sandbox_result = self._sandbox.execute(
            action.theory_module,
            history,
            future,
            self._universe.sensor_names,
        )
        reward = score_theory(
            sandbox_result,
            future,
            action.theory_module,
            self.config.reward,
            paradigm_shift_claim=action.paradigm_shift_claim,
            previous_theory_module=self._last_theory_module,
        )
        diagnostics = prediction_diagnostics(
            sandbox_result.predictions,
            future,
            self.config.reward.prediction_sigma,
            self.config.feedback.divergence_sigma,
        )
        self._last_metrics = self._metrics_dict(reward, sandbox_result, diagnostics)
        self._last_execution_output = self._format_execution_output(reward, sandbox_result, action.revision_note)
        review = self._review(reward, sandbox_result, future, action, diagnostics)
        self._peer_reviews.append(review)
        self._peer_reviews = self._peer_reviews[-max(1, self.config.feedback.peer_review_window) :]
        self._record_theory_lineage(action, reward, sandbox_result, review, diagnostics)
        self._last_theory_module = action.theory_module

        self._cursor += self.config.universe.future_window
        done = self._state.step_count >= self.config.universe.max_steps
        return self._make_observation(done=done, reward=reward.reward)

    @property
    def state(self) -> State:
        return self._state

    def _build_universe(self, seed: int) -> ProceduralAlienUniverse:
        universe_config = replace(self.config.universe, seed=seed)
        return ProceduralAlienUniverse(universe_config)

    def _make_observation(self, done: bool, reward: float) -> OntologyArchitectObservation:
        start = max(0, self._cursor - self.config.universe.observation_window)
        visible_records = self._records[start : self._cursor]
        sensor_log = format_sensor_log(visible_records)
        peer_review = "\n".join(f"- {line}" for line in self._peer_reviews)
        theory_lineage = self._format_theory_lineage()
        metadata = {
            "episode_id": self._state.episode_id,
            "step": self._state.step_count,
            "history_records": self._cursor,
            "future_horizon": self.config.universe.future_window,
            "sensor_names": list(self._universe.sensor_names),
            "allowed_imports": list(self.config.sandbox.allowed_imports),
            "theory_api": "class Theory with fit(history), predict(window), log_prob(observations)",
            "last_metrics": self._last_metrics,
            "feedback": {
                "peer_review_window": self.config.feedback.peer_review_window,
                "lineage_window": self.config.feedback.lineage_window,
                "divergence_sigma": self.config.feedback.divergence_sigma,
            },
            "theory_lineage": list(self._theory_lineage),
        }
        text = "\n\n".join(
            [
                "# ONTOLOGY ARCHITECT OBSERVATION",
                "## RAW SENSOR LOG\n" + sensor_log,
                "## LAST THEORY EXECUTION\n" + self._last_execution_output,
                "## PEER REVIEW\n" + peer_review,
                "## THEORY LINEAGE\n" + theory_lineage,
                "## METADATA\n"
                + "\n".join(
                    [
                        f"episode_id={metadata['episode_id']}",
                        f"step={metadata['step']}",
                        f"history_records={metadata['history_records']}",
                        f"future_horizon={metadata['future_horizon']}",
                        f"sensor_names={','.join(metadata['sensor_names'])}",
                        f"allowed_imports={','.join(metadata['allowed_imports'])}",
                        f"theory_api={metadata['theory_api']}",
                    ]
                ),
            ]
        )
        return OntologyArchitectObservation(
            text=text,
            sensor_log=sensor_log,
            execution_output=self._last_execution_output,
            peer_review=peer_review,
            metadata=metadata,
            done=done,
            reward=reward,
        )

    @staticmethod
    def _metrics_dict(reward: RewardBreakdown, result: SandboxResult, diagnostics: dict) -> dict:
        return {
            "reward": reward.reward,
            "log_likelihood": reward.log_likelihood,
            "mdl_penalty": reward.mdl_penalty,
            "anomaly_bonus": reward.anomaly_bonus,
            "drift_bonus": reward.drift_bonus,
            "execution_ok": reward.execution_ok,
            "missed_anomaly": reward.missed_anomaly,
            "false_paradigm_shift": reward.false_paradigm_shift,
            "weak_paradigm_shift": reward.weak_paradigm_shift,
            "runtime_ms": result.runtime_ms,
            "details": reward.details,
            "prediction_diagnostics": diagnostics,
        }

    @staticmethod
    def _format_execution_output(
        reward: RewardBreakdown,
        result: SandboxResult,
        revision_note: str,
    ) -> str:
        if not result.ok:
            return f"Execution failed: {result.error}\n{result.traceback}".strip()
        lines = [
            f"reward={reward.reward:.5f}",
            f"log_likelihood={reward.log_likelihood:.5f}",
            f"mdl_penalty={reward.mdl_penalty:.5f}",
            f"anomaly_bonus={reward.anomaly_bonus:.5f}",
            f"drift_adaptation_bonus={reward.drift_bonus:.5f}",
            f"reported_log_prob={result.reported_log_prob:.5f}",
            f"runtime_ms={result.runtime_ms:.3f}",
        ]
        if reward.weak_paradigm_shift:
            lines.append("paradigm_shift_review=claimed drift adaptation but structural change was weak")
        if result.description:
            lines.append(f"description={result.description}")
        if revision_note:
            lines.append(f"revision_note={revision_note[:500]}")
        if result.stdout:
            lines.append("stdout=" + result.stdout[-1000:])
        return "\n".join(lines)

    def _review(
        self,
        reward: RewardBreakdown,
        result: SandboxResult,
        future: list,
        action: OntologyArchitectAction,
        diagnostics: dict,
    ) -> str:
        if not result.ok:
            return f"Execution review: rejected theory because {result.error}"

        parts = []
        if reward.false_paradigm_shift:
            parts.append("Paradigm review: claim was not supported by the next hidden evaluation window.")
        elif reward.weak_paradigm_shift:
            parts.append(
                "Paradigm review: drift was present, but the submitted theory was only weakly different from its parent."
            )
        elif reward.missed_anomaly and window_has_anomaly(future):
            parts.append("Anomaly review: a rare sensor excursion occurred and the theory assigned it low probability.")
        elif reward.log_likelihood < -20.0:
            parts.append(
                "Prediction review: future residuals were very large; the ontology may be missing a latent variable."
            )
        elif reward.log_likelihood < -5.0:
            parts.append("Prediction review: residuals remain structured; a compact reparameterization may help.")
        else:
            parts.append("Review: theory survived this window; continue testing whether its ontology stays compact.")

        worst_sensor = diagnostics.get("worst_sensor")
        per_sensor = diagnostics.get("per_sensor", {})
        if worst_sensor and worst_sensor in per_sensor:
            stats = per_sensor[worst_sensor]
            parts.append(
                "Sensor error: "
                f"worst={worst_sensor} mae={stats['mae']:.5f} "
                f"rmse={stats['rmse']:.5f} max_abs={stats['max_abs_error']:.5f}"
            )
        first_divergence = diagnostics.get("first_divergence")
        if first_divergence:
            parts.append(
                "Divergence: "
                f"t={first_divergence['t']:.3f} sensor={first_divergence['sensor']} "
                f"abs_error={first_divergence['abs_error']:.5f} "
                f"threshold={first_divergence['threshold']:.5f}"
            )

        falsified = self._falsified_hypotheses(reward, result, future, action, diagnostics)
        if falsified:
            parts.append("Falsified hypotheses: " + json.dumps(falsified, sort_keys=True))
        return " | ".join(parts)

    def _falsified_hypotheses(
        self,
        reward: RewardBreakdown,
        result: SandboxResult,
        future: list,
        action: OntologyArchitectAction,
        diagnostics: dict,
    ) -> list[dict]:
        hypotheses = []
        first_divergence = diagnostics.get("first_divergence")
        if first_divergence:
            hypotheses.append(
                {
                    "type": "prediction_divergence",
                    "claim": "future residuals stay inside the configured Gaussian error band",
                    "t": first_divergence["t"],
                    "sensor": first_divergence["sensor"],
                    "abs_error": first_divergence["abs_error"],
                    "threshold": first_divergence["threshold"],
                }
            )
        if reward.missed_anomaly and window_has_anomaly(future):
            hypotheses.append(
                {
                    "type": "missed_anomaly",
                    "claim": "anomaly probability covers rare sensor excursions",
                }
            )
        if reward.log_likelihood < -20.0:
            hypotheses.append(
                {
                    "type": "latent_variable_gap",
                    "claim": "current ontology explains the next window without an additional latent state",
                    "log_likelihood": reward.log_likelihood,
                }
            )
        if reward.false_paradigm_shift:
            hypotheses.append(
                {
                    "type": "false_paradigm_shift",
                    "claim": "submitted paradigm shift corresponds to actual non-stationarity",
                }
            )
        if reward.weak_paradigm_shift:
            hypotheses.append(
                {
                    "type": "weak_paradigm_shift",
                    "claim": "submitted paradigm shift is structurally different from its parent theory",
                    "structural_delta": reward.details.get("structural_delta", 0.0),
                    "required_delta": self.config.reward.paradigm_shift_min_structural_delta,
                }
            )
        if window_has_drift(future) and not (action.paradigm_shift_claim or result.drift_detected):
            hypotheses.append(
                {
                    "type": "missed_drift",
                    "claim": "current ontology remains stationary across this future window",
                }
            )
        return hypotheses

    def _record_theory_lineage(
        self,
        action: OntologyArchitectAction,
        reward: RewardBreakdown,
        result: SandboxResult,
        review: str,
        diagnostics: dict,
    ) -> None:
        parent_id = self._theory_lineage[-1]["theory_id"] if self._theory_lineage else None
        theory_id = hashlib.sha256(action.theory_module.encode("utf-8")).hexdigest()[:12]
        record = {
            "step": self._state.step_count,
            "theory_id": theory_id,
            "parent_id": parent_id,
            "reward": reward.reward,
            "log_likelihood": reward.log_likelihood,
            "mdl_penalty": reward.mdl_penalty,
            "mdl_complexity": reward.details.get("mdl_complexity"),
            "structural_delta": reward.details.get("structural_delta", 1.0),
            "paradigm_shift_claim": action.paradigm_shift_claim,
            "drift_detected": result.drift_detected,
            "worst_sensor": diagnostics.get("worst_sensor"),
            "revision_note": action.revision_note[:160],
            "review": review[:500],
        }
        self._theory_lineage.append(record)
        self._theory_lineage = self._theory_lineage[-max(1, self.config.feedback.lineage_window) :]

    def _format_theory_lineage(self) -> str:
        if not self._theory_lineage:
            return "No submitted theories in this episode yet."
        lines = []
        for item in self._theory_lineage:
            parent = item["parent_id"] or "root"
            lines.append(
                f"- step={item['step']} id={item['theory_id']} parent={parent} "
                f"reward={item['reward']:.5f} log_likelihood={item['log_likelihood']:.5f} "
                f"mdl_complexity={item['mdl_complexity']} structural_delta={item['structural_delta']:.3f} "
                f"paradigm_shift_claim={item['paradigm_shift_claim']} worst_sensor={item['worst_sensor']}"
            )
        return "\n".join(lines)
