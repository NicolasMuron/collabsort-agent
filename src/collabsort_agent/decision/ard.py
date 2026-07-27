"""
Advantage Racing Diffusion definitions.
"""

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from collabsort_agent.decision import DecisionConfig, Deliberator
from collabsort_agent.decision.accumulators import Accumulators
from collabsort_agent.decision.decision_rule import DecisionRule
from collabsort_agent.learning import ActionValueEstimator
from collabsort_agent.metacognition import Hyperparameters
from collabsort_agent.metacognition.controller import MetaController
from collabsort_agent.metacognition.monitoring import MetaMonitoring


class ARD(Deliberator):
    """
    Advantage Racing Diffusion algorithm for decision making.

    Translates Q-values into an action selection with associated confidence
    via evidence accumulation. Each action has n-1 "advantage" accumulators.

    Inspired by Miletic2021 https://elifesciences.org/articles/63055
    """

    def __init__(
        self,
        config: DecisionConfig,
        estimator: ActionValueEstimator,
        decision_rule: DecisionRule,
        hyperparameters: Hyperparameters,
        meta_monitoring: MetaMonitoring,
        meta_ctrl: MetaController,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(config=config, estimator=estimator, rng=rng)

        self.decision_rule = decision_rule
        self.hyperparameters = hyperparameters
        self.meta_monitoring = meta_monitoring
        self.meta_ctrl = meta_ctrl

        # Init accumulators
        self.accumulators = Accumulators(n_actions=estimator.n_actions)

    def choose_action(
        self,
        state: np.ndarray,
        training_step: int,
    ) -> int:
        """Choose the action to perform"""

        action_values = self.estimator.get_action_values(state=state)

        n_actions = len(action_values)
        if n_actions == 1:
            return 0  # Only one possible action

        # Reset evidence and history
        self.accumulators.evidence = self.accumulators.empty_evidence()
        self.accumulators.evidence_history = self.accumulators.empty_evidence()

        # Compute drift rates for all accumulators
        drift_rates = self._compute_drift_rates(action_values)
        self.accumulators.drift_rates = drift_rates

        chosen_action = -1
        rt = float(self.config.max_steps)

        # Evidence accumulation loop
        for t in range(1, self.config.max_steps + 1):
            # Compute accumulation noise
            noise = self.rng.normal(
                loc=self.config.noise_mean,
                scale=self.config.noise_std,
                size=self.accumulators.n_accumulators,
            )

            # Accumulate evidence
            self.accumulators.evidence += drift_rates * self.config.dt + noise

            # Absorb lower bound
            np.clip(
                self.accumulators.evidence,
                a_min=0.0,
                a_max=None,  # self.meta_ctrl.theta,
                out=self.accumulators.evidence,
            )

            # Add new evidence to history
            self.accumulators.evidence_history = np.c_[
                self.accumulators.evidence_history, self.accumulators.evidence
            ]

            winning_actions = self.decision_rule.get_winning_actions(
                n_actions=n_actions,
                evidence=self.accumulators.evidence,
                theta=self.hyperparameters.theta,
                adv_accs=self.accumulators.adv_accs,
            )

            if winning_actions:
                if len(winning_actions) > 1:
                    # More than one action have seen all their advantage accumulators cross the threshold.
                    # Select action whose slowest accumulator has highest value.
                    min_winners_evidence = self.accumulators.min_evidence(
                        actions=winning_actions
                    )
                    chosen_action = np.argmax(min_winners_evidence).item()

                elif len(winning_actions) == 1:
                    # Only one action has seen all its advantage accumulators cross the threshold
                    chosen_action = winning_actions[0]

                rt = float(t)
                break

        min_actions_evidence = self.accumulators.min_evidence(
            actions=list(range(n_actions))
        )

        # Fallback: no action chosen within max_steps.
        # Select action whose slowest accumulator has highest value.
        if chosen_action == -1:
            chosen_action = np.argmax(min_actions_evidence).item()

        # Compute second best action (non-winning action whose slowest accumulator has highest value)
        min_actions_evidence[chosen_action] = 0.0
        runnerup_action = np.argmax(min_actions_evidence).item()

        # Compute decision confidence and adjust hyperparameters
        confidence = self.meta_monitoring.compute_decision_confidence(
            chosen_action=chosen_action,
            runnerup_action=runnerup_action,
            reaction_time=rt,
            accumulators=self.accumulators,
        )
        self.meta_ctrl.update_hyperparameters(confidence=confidence, reaction_time=rt)

        return chosen_action

    def _compute_drift_rates(self, q_values: np.ndarray) -> np.ndarray:
        """
        Return the drift rates for all accumulators. Shape: (n_accumulators,).

        v(i,j) = w_d*(Q_i - Q_j) + w_s*(Q_i + Q_j) + V0
        """

        pairs = np.array(self.accumulators.action_pairs)
        i_idx = pairs[:, 0]
        j_idx = pairs[:, 1]
        return (
            self.config.w_d * (q_values[i_idx] - q_values[j_idx])
            + self.config.w_s * (q_values[i_idx] + q_values[j_idx])
            + self.config.V_0
        )

    def _compute_drift_rates_dict(
        self, q_values: np.ndarray
    ) -> dict[tuple[int, int], float]:
        """Return {(i,j): drift_rate}. Used for debugging."""

        v = self._compute_drift_rates(q_values)
        return {
            pair: float(v[k]) for k, pair in enumerate(self.accumulators.action_pairs)
        }

    def log_episode(self, logger: SummaryWriter, episode: int) -> None:
        """Log information after an episode"""

        logger.add_scalar(
            tag="decision/accumulation_threshold",
            scalar_value=self.hyperparameters.theta,
            global_step=episode,
        )
        self.meta_ctrl.log_episode(logger=logger, episode=episode)

    def save_state(self, dir: str) -> None:
        # TODO save state for ARD
        pass

    def load_state(self, dir: str) -> None:
        # TODO load state for ARD
        pass
