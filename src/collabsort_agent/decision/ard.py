"""
Advantage Racing Diffusion definitions.
"""

import itertools

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from collabsort_agent.decision import Config as DecisionConfig
from collabsort_agent.decision import Deliberator
from collabsort_agent.learning import ActionValueEstimator
from collabsort_agent.metacognition import MetaController


class ARD(Deliberator):
    """Advantage Racing Diffusion algorithm."""

    def __init__(
        self,
        config: DecisionConfig,
        estimator: ActionValueEstimator,
        meta_ctrl: MetaController,
    ) -> None:
        super().__init__(config=config, estimator=estimator)

        self.meta_ctrl = meta_ctrl

        self._n_actions: int = 0
        self._pairs: list[tuple[int, int]] = []
        self._pro_idx: dict[int, list[int]] = {}

    def choose_action(
        self,
        state: np.ndarray,
        training_step: int,
        rng: np.random.Generator,
    ) -> int:
        """Choose the action to perform"""

        action_values = self.estimator.get_action_values(state=state)

        n = len(action_values)
        if n != self._n_actions:
            self._build_pair_index(n)

        if n == 1:
            return 0  # Only one possible action

        # Mean drift rates, shape (n_pairs,)
        v = self._drift_rates(action_values)
        n_pairs = len(v)
        noise_std = self.config.noise_std * np.sqrt(
            self.config.dt
        )  # if stochastic else 0.0

        evidence = np.zeros(n_pairs, dtype=float)
        action = -1
        rt = float(self.config.max_steps)

        for t in range(1, self.config.max_steps + 1):
            # if stochastic:
            evidence += v * self.config.dt + rng.normal(0.0, noise_std, size=n_pairs)
            # else:
            #     evidence += v * self.config.dt
            np.clip(evidence, 0.0, None, out=evidence)  # absorbing lower bound

            # Win-all check: action i wins when ALL its pro-i accumulators ≥ θ
            crossed = evidence >= self.meta_ctrl.theta
            winners = [i for i in range(n) if all(crossed[k] for k in self._pro_idx[i])]

            if winners:
                rt = float(t)
                action = int(rng.choice(winners))  # if stochastic else winners[0]
                # action = winners[0]
                break

        # Fallback: no action completed within max_steps — pick least incomplete
        if action == -1:
            completion = np.array(
                [
                    np.mean(evidence[self._pro_idx[i]] >= self.meta_ctrl.theta)
                    for i in range(n)
                ],
                dtype=float,
            )
            action = (
                int(rng.choice(np.where(completion == completion.max())[0]))
                # if stochastic
                # else int(np.argmax(completion))
            )

        # Confidence: 1 − runner-up completion ratio at decision time
        completion = np.array(
            [
                np.mean(evidence[self._pro_idx[i]] >= self.meta_ctrl.theta)
                for i in range(n)
            ],
            dtype=float,
        )
        sorted_comp = np.sort(completion)[::-1]
        runner_up = float(sorted_comp[1]) if n > 1 else 0.0
        confidence = float(np.clip(1.0 - runner_up, 0.0, 1.0))

        # self.meta_ctrl.update_hyperparameters(confidence=confidence, reaction_time=rt)

        return action

    def _build_pair_index(self, n_actions: int) -> None:
        """Pre-compute ordered pair indices; called once per new n_actions."""

        self._n_actions = n_actions
        self._pairs = list(itertools.permutations(range(n_actions), 2))
        self._pro_idx = {i: [] for i in range(n_actions)}
        for k, (i, _j) in enumerate(self._pairs):
            self._pro_idx[i].append(k)

    def _drift_rates(self, q: np.ndarray) -> np.ndarray:
        """
        v(i,j) = w_d*(Q_i - Q_j) + w_s*(Q_i + Q_j) + V0

        Returns shape (n_pairs,) array.
        """

        pairs_arr = np.array(self._pairs)  # (n_pairs, 2)
        i_idx = pairs_arr[:, 0]
        j_idx = pairs_arr[:, 1]
        return (
            self.config.w_d * (q[i_idx] - q[j_idx])
            + self.config.w_s * (q[i_idx] + q[j_idx])
            + self.config.V_0
        )

    def log_episode(self, logger: SummaryWriter, episode: int) -> None:
        """Log information after an episode"""

        logger.add_scalar(
            tag="decision/accumulation_threshold",
            scalar_value=self.meta_ctrl.theta,
            global_step=episode,
        )
        self.meta_ctrl.log_episode(logger=logger, episode=episode)

    def save_state(self, dir: str) -> None:
        """Save the deliberator state to disk"""

        # TODO save state for ARD

    def load_state(self, dir: str) -> None:
        """Load a previously saved deliberator state from disk"""

        # TODO load state for ARD
