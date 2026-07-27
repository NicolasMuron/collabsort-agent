"""
Unit tests for the Advantage Racing Diffusion (ARD) deliberator.
"""

import matplotlib.pyplot as plt
import numpy as np

from collabsort_agent.decision import DecisionConfig
from collabsort_agent.decision.ard import ARD
from collabsort_agent.decision.decision_rule import WinAllRule
from collabsort_agent.learning import ActionValueEstimator, LearningConfig
from collabsort_agent.metacognition import Hyperparameters, MetaConfig
from collabsort_agent.metacognition.confidence import BayesianConfidence
from collabsort_agent.metacognition.controller import MetaController
from collabsort_agent.metacognition.monitoring import MetaMonitoring


class EstimatorStub(ActionValueEstimator):
    """Estimator stub returning fixed Q-values"""

    def __init__(self, action_values: np.ndarray) -> None:
        super().__init__(config=LearningConfig(), n_actions=len(action_values))

        self._action_values = action_values

    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        return self._action_values

    def update_action_values(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ):
        pass

    def save_state(self, dir: str) -> None:
        pass

    def load_state(self, dir: str) -> None:
        pass


class MetaStub(MetaController):
    """Metacognition stub"""

    def __init__(
        self, decision_cfg: DecisionConfig, hyperparameters: Hyperparameters
    ) -> None:
        super().__init__(
            # No update of hyperparameters
            config=MetaConfig(alpha_rate=0.0, theta_rate=0.0),
            learning_cfg=LearningConfig(),
            decision_cfg=decision_cfg,
            hyperparameters=hyperparameters,
        )
        self.confidence: float = 0.0
        self.rt: float = 0.0

    def update_hyperparameters(self, confidence: float, reaction_time: float) -> None:
        # Store received values for latter assertion
        self.confidence = confidence
        self.rt = reaction_time


class TestARD:
    def test_chosen_action(self, show_plots: bool = False) -> None:
        state = np.zeros(5, dtype=np.float32)
        q_matrix = [[0.0, 1.0, 0.0], [0.5, 1.0, 0.2], [0.05, 0.2, 0.1]]

        for q_values in q_matrix:
            q_values = np.array(q_values)
            ard, _ = self._make_ard(action_values=q_values, config=DecisionConfig())
            action = ard.choose_action(state=state, training_step=0)

            if show_plots:
                # Plot all accumulators
                lines = plt.plot(ard.accumulators.evidence_history.T)

                # Plot decision threashold
                plt.hlines(
                    ard.hyperparameters.theta,
                    xmin=plt.xlim()[0],
                    xmax=plt.xlim()[1],
                    linestyles="dotted",
                )
                plt.annotate(
                    "Decision threshold",
                    xy=(plt.xlim()[1] / 4, ard.hyperparameters.theta * 1.02),
                )

                # Plot decision time
                plt.vlines(
                    len(ard.accumulators.evidence_history[0]) - 1,
                    ymin=plt.ylim()[0],
                    ymax=plt.ylim()[1],
                    linestyles="dotted",
                )

                # Add labels for each accumulator
                line_i = 0
                for i, j in ard.accumulators.action_pairs:
                    lines[line_i].set_label(f"({i},{j})")
                    line_i += 1

                # Change line style for accumulators (0,*) and (2,*).
                # (Should be improved from n_actions != 3)
                for line in lines[:2]:
                    line.set_linestyle("dashed")
                for line in lines[4:]:
                    line.set_linestyle("dashdot")

                plt.xlabel("Time steps (decision)")
                plt.ylabel("Evidence")
                plt.legend()
                plt.title(f"Accumulator race ({len(q_values)} actions)")

                plt.show()

            # Given the q_values, the following assertion should apply in all cases
            assert action == np.argmax(q_values)

    def test_drift_rates(self) -> None:
        state = np.zeros(5, dtype=np.float32)
        q_values = np.array([0.5, 1.0, 0.2])

        ard, _ = self._make_ard(action_values=q_values, config=DecisionConfig())
        ard.choose_action(state=state, training_step=0)

        drift_rates = ard._compute_drift_rates_dict(q_values=q_values)
        for i, j in ard.accumulators.action_pairs:
            # v(i,j) = w_d*(Q_i - Q_j) + w_s*(Q_i + Q_j) + V0
            expected_drift_rate = (
                ard.config.w_d * (q_values[i] - q_values[j])
                + ard.config.w_s * (q_values[i] + q_values[j])
                + ard.config.V_0
            )
            assert abs(drift_rates[i, j] - expected_drift_rate) < 1e-6

    def _make_ard(
        self, action_values: np.ndarray, config: DecisionConfig
    ) -> tuple[ARD, MetaStub]:
        rng = np.random.default_rng(42)

        learning_cfg = LearningConfig()
        hyperparameters = Hyperparameters(
            decision_cfg=config, learning_cfg=learning_cfg
        )
        meta_monitoring = MetaMonitoring(
            confidence_method=BayesianConfidence(
                decision_cfg=config, hyperparameters=hyperparameters
            )
        )
        meta_stub = MetaStub(decision_cfg=config, hyperparameters=hyperparameters)
        ard = ARD(
            config=config,
            estimator=EstimatorStub(action_values=action_values),
            decision_rule=WinAllRule(rng=rng),
            hyperparameters=hyperparameters,
            meta_monitoring=meta_monitoring,
            meta_ctrl=meta_stub,
            rng=rng,
        )
        return ard, meta_stub


if __name__ == "__main__":
    # Standalone execution
    TestARD().test_chosen_action(show_plots=True)
