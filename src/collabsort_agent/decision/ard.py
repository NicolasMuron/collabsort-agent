"""
Advantage Racing Diffusion definitions.
"""

import itertools
import math

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from collabsort_agent.decision import Config as DecisionConfig
from collabsort_agent.decision import Deliberator
from collabsort_agent.decision.decision_rule import DecisionRule
from collabsort_agent.learning import ActionValueEstimator
from collabsort_agent.metacognition import MetaController


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
        meta_ctrl: MetaController,
        rng: np.random.Generator,
    ) -> None:
        super().__init__(config=config, estimator=estimator, rng=rng)

        self.decision_rule = decision_rule
        self.meta_ctrl = meta_ctrl

        # Ordered list of (i,j) pairs of actions for 0 <= i,j < n_actions and i != j.
        # Length = n_actions(n_actions - 1) = n_accumulators.
        # Example for n_actions = 3: (0,1),(0,2),(1,0),(1,2),(2,0),(2,1).
        self._action_pairs: list[tuple[int, int]] = list(
            itertools.permutations(range(estimator.n_actions), 2)
        )

        # Dictionary of advantage ("pro-action") accumulators indexes for each action.
        # Example for n_actions = 3: {0:[0,1], 1:[2,3], 2:[4,5]}.
        # These indexes are used to access individual accumulators in self._evidence.
        self._adv_accs: dict[int, list[int]] = {
            i: [] for i in range(estimator.n_actions)
        }
        for k, (i, _) in enumerate(self._action_pairs):
            self._adv_accs[i].append(k)

        # Cumulated evidence for each accumulator during a decision. Shape: (n_accumulators,)
        self._evidence = self._reset_evidence()

        # Drift rates for the accumulators of the current/last decision. Shape: (n_accumulators,)
        # Stored so that _compute_confidence can access the drift rate of any
        # specific accumulator (needed for the Bayesian confidence estimate).
        self._drift_rates = self._reset_evidence()

        # History matrix of evidence values at each time step of a decision. Used for plotting/debugging.
        # Each line stores all successive evidence values for one accumulator.
        # Shape: (n_accumulators, n_decision_steps).
        # Initialized as a 1D array. Each decision step will add a new column.
        self._evidence_history = self._reset_evidence()

    @property
    def n_accumulators(self) -> int:
        """Return the number of accumulators = n_actions(n_actions - 1)"""

        return len(self._action_pairs)

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
        self._evidence = self._reset_evidence()
        self._evidence_history = self._reset_evidence()

        # Compute drift rates for all accumulators
        drift_rates = self._compute_drift_rates(action_values)
        self._drift_rates = drift_rates

        chosen_action = -1
        rt = float(self.config.max_steps)

        # Evidence accumulation loop
        for t in range(1, self.config.max_steps + 1):
            # Compute accumulation noise
            noise = self.rng.normal(
                loc=self.config.noise_mean,
                scale=self.config.noise_std,
                size=self.n_accumulators,
            )

            # Accumulate evidence
            self._evidence += drift_rates * self.config.dt + noise

            # Absorb lower bound
            np.clip(
                self._evidence,
                a_min=0.0,
                a_max=None,  # self.meta_ctrl.theta,
                out=self._evidence,
            )

            # Add new evidence to history
            self._evidence_history = np.c_[self._evidence_history, self._evidence]

            winning_actions = self.decision_rule.get_winning_actions(
                n_actions=n_actions,
                evidence=self._evidence,
                theta=self.meta_ctrl.theta,
                adv_accs=self._adv_accs,
            )

            if winning_actions:
                if len(winning_actions) > 1:
                    # More than one action have seen all their advantage accumulators cross the threshold.
                    # Select action whose slowest accumulator has highest value.
                    min_winners_evidence = self._compute_min_evidence(
                        actions=winning_actions
                    )
                    chosen_action = np.argmax(min_winners_evidence).item()

                elif len(winning_actions) == 1:
                    # Only one action has seen all its advantage accumulators cross the threshold
                    chosen_action = winning_actions[0]

                rt = float(t)
                break

        min_actions_evidence = self._compute_min_evidence(
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
        confidence = self._compute_confidence(
            chosen_action=chosen_action,
            runnerup_action=runnerup_action,
            reaction_time=rt,
        )
        self.meta_ctrl.update_hyperparameters(confidence=confidence, reaction_time=rt)

        return chosen_action

    def _reset_evidence(self) -> np.ndarray:
        """Reset evidence."""

        return np.zeros((self.n_accumulators,), dtype=float)

    def _compute_min_evidence(self, actions: list[int]) -> list[float]:
        """Return the minimum value of all advantage accumulators for a list of actions."""

        return [np.min(self._evidence[self._adv_accs[action]]) for action in actions]

    def _argmin_accumulator(self, action: int) -> int:
        """
        Return the global index (into self._evidence / self._drift_rates) of
        the "slowest" advantage accumulator for a given action, i.e. the one
        with the lowest evidence value. This is the accumulator that
        determines when/whether the action wins the race (win-all rule).
        """

        accs = self._adv_accs[action]
        local_argmin = int(np.argmin(self._evidence[accs]))
        return accs[local_argmin]

    def _compute_confidence(
        self, chosen_action: int, runnerup_action: int, reaction_time: float
    ) -> float:
        """Compute decision confidence, using the method set in config.confidence_method."""

        if self.config.confidence_method == "bayesian":
            return self._compute_confidence_bayesian(
                chosen_action=chosen_action,
                runnerup_action=runnerup_action,
                reaction_time=reaction_time,
            )
        return self._compute_confidence_gap(
            chosen_action=chosen_action, runnerup_action=runnerup_action
        )

    def _compute_confidence_gap(
        self, chosen_action: int, runnerup_action: int
    ) -> float:
        """
        Legacy confidence measure: normalized distance between slowest
        accumulators of chosen and runner-up actions. Purely geometric,
        not on a fixed/interpretable scale (depends on noise level and
        accumulation duration).
        """

        min_evidence = self._compute_min_evidence(
            actions=[chosen_action, runnerup_action]
        )
        return (min_evidence[0] - min_evidence[1]) / (self.meta_ctrl.theta + 1e-6)

    def _compute_confidence_bayesian(
        self, chosen_action: int, runnerup_action: int, reaction_time: float
    ) -> float:
        """
        Bayesian (signal-detection) confidence: the posterior probability
        that the chosen action's slowest accumulator truly has a higher
        drift rate than the runner-up's slowest accumulator, given the
        accumulation noise model. Inspired by Kepecs2008
        https://doi.org/10.1038/nature07200

        Each accumulator x_k follows dx_k = v_k*dt + noise, noise ~ N(0, noise_std^2)
        per simulation step (see choose_action). After `reaction_time` steps:
            x_k ~ N(v_k * reaction_time * dt, reaction_time * noise_std^2)
        The difference between the chosen and runner-up slowest accumulators
        is then approximately Gaussian, and confidence is the posterior
        probability that this difference has positive mean, i.e. that the
        winning accumulator's drift genuinely exceeded the runner-up's:

            c = Phi( (v_i - v_j) * t* / (sigma * sqrt(2*t*)) )

        with t* = reaction_time * dt and sigma^2 = noise_std^2 / dt (the
        per-unit-time noise variance implied by the discretization above).
        """

        chosen_idx = self._argmin_accumulator(chosen_action)
        runnerup_idx = self._argmin_accumulator(runnerup_action)

        v_diff = self._drift_rates[chosen_idx] - self._drift_rates[runnerup_idx]

        # Elapsed simulated time and standard deviation of the accumulator
        # difference, matching the discretization used in choose_action
        # (noise variance accrues per step, drift accrues per unit of dt).
        elapsed_time = reaction_time * self.config.dt
        std_diff = self.config.noise_std * math.sqrt(2.0 * reaction_time)

        z = (v_diff * elapsed_time) / (std_diff + 1e-12)
        return self._normal_cdf(z)

    @staticmethod
    def _normal_cdf(z: float) -> float:
        """Standard normal CDF, computed via the error function (no scipy dependency)."""

        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    def _compute_drift_rates(self, q_values: np.ndarray) -> np.ndarray:
        """
        Return the drift rates for all accumulators. Shape: (n_accumulators,).

        v(i,j) = w_d*(Q_i - Q_j) + w_s*(Q_i + Q_j) + V0
        """

        pairs = np.array(self._action_pairs)
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
        return {pair: float(v[k]) for k, pair in enumerate(self._action_pairs)}

    def log_episode(self, logger: SummaryWriter, episode: int) -> None:
        """Log information after an episode"""

        logger.add_scalar(
            tag="decision/accumulation_threshold",
            scalar_value=self.meta_ctrl.theta,
            global_step=episode,
        )
        self.meta_ctrl.log_episode(logger=logger, episode=episode)

    def save_state(self, dir: str) -> None:
        # TODO save state for ARD
        pass

    def load_state(self, dir: str) -> None:
        # TODO load state for ARD
        pass
