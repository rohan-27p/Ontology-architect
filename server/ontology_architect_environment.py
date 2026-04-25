"""Core Ontology Architect OpenEnv environment."""

from __future__ import annotations

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
    from ontology_architect.reward import RewardBreakdown, score_theory
    from ontology_architect.sandbox import SandboxResult, TheorySandbox
    from ontology_architect.universe import ProceduralAlienUniverse, format_sensor_log, window_has_anomaly
except ImportError:  # pragma: no cover - local server.app execution from repository root.
    from config import ExperimentConfig
    from models import OntologyArchitectAction, OntologyArchitectObservation
    from reward import RewardBreakdown, score_theory
    from sandbox import SandboxResult, TheorySandbox
    from universe import ProceduralAlienUniverse, format_sensor_log, window_has_anomaly


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
        )
        self._last_metrics = self._metrics_dict(reward, sandbox_result)
        self._last_execution_output = self._format_execution_output(reward, sandbox_result, action.revision_note)
        self._peer_reviews.append(self._review(reward, sandbox_result, future))
        self._peer_reviews = self._peer_reviews[-8:]

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
        metadata = {
            "episode_id": self._state.episode_id,
            "step": self._state.step_count,
            "history_records": self._cursor,
            "future_horizon": self.config.universe.future_window,
            "sensor_names": list(self._universe.sensor_names),
            "allowed_imports": list(self.config.sandbox.allowed_imports),
            "theory_api": "class Theory with fit(history), predict(window), log_prob(observations)",
            "last_metrics": self._last_metrics,
        }
        text = "\n\n".join(
            [
                "# ONTOLOGY ARCHITECT OBSERVATION",
                "## RAW SENSOR LOG\n" + sensor_log,
                "## LAST THEORY EXECUTION\n" + self._last_execution_output,
                "## PEER REVIEW\n" + peer_review,
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
    def _metrics_dict(reward: RewardBreakdown, result: SandboxResult) -> dict:
        return {
            "reward": reward.reward,
            "log_likelihood": reward.log_likelihood,
            "mdl_penalty": reward.mdl_penalty,
            "anomaly_bonus": reward.anomaly_bonus,
            "drift_bonus": reward.drift_bonus,
            "execution_ok": reward.execution_ok,
            "missed_anomaly": reward.missed_anomaly,
            "false_paradigm_shift": reward.false_paradigm_shift,
            "runtime_ms": result.runtime_ms,
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
        if result.description:
            lines.append(f"description={result.description}")
        if revision_note:
            lines.append(f"revision_note={revision_note[:500]}")
        if result.stdout:
            lines.append("stdout=" + result.stdout[-1000:])
        return "\n".join(lines)

    @staticmethod
    def _review(reward: RewardBreakdown, result: SandboxResult, future: list) -> str:
        if not result.ok:
            return f"Execution review: rejected theory because {result.error}"
        if reward.false_paradigm_shift:
            return "Review: paradigm-shift claim was not supported by the next hidden evaluation window."
        if reward.missed_anomaly and window_has_anomaly(future):
            return "Anomaly review: a rare sensor excursion occurred and the theory assigned it low probability."
        if reward.log_likelihood < -20.0:
            return "Prediction review: future residuals were very large; the ontology may be missing a latent variable."
        if reward.log_likelihood < -5.0:
            return "Prediction review: residuals remain structured; a compact reparameterization may help."
        return "Review: theory survived this window; continue testing whether its ontology stays compact."
