"""
Definitions for Q-Learning algorithm.
"""

from collections import defaultdict

import numpy as np

from collabsort_agent.learning import ActionValueEstimator
from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.metacognition import MetaController


class Qlearning(ActionValueEstimator):
    """Q-Learning algorithm."""

    def __init__(
        self, config: LearningConfig, n_actions: int, meta_ctrl: MetaController
    ) -> None:
        super().__init__(config=config, n_actions=n_actions)

        self.config = config
        self.n_actions = n_actions
        self.meta_ctrl = meta_ctrl

        self._table: dict[tuple, np.ndarray] = defaultdict(
            lambda: np.full(n_actions, config.q_start, dtype=float)
        )

    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        """Return the action values for all actions"""

        return self._table[self._make_key(state)].copy()

    def update_action_values(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ):
        """Update action values after an action was taken"""

        q_values = self.get_action_values(state)
        self.mean_q_values.append(np.mean(q_values))

        q_current = float(q_values[action])
        q_next_max = float(self.get_action_values(next_state).max())

        # δ = r + γ · max_a' Q(s', a') − Q(s, a)
        q_target = reward if done else reward + self.config.gamma * q_next_max
        td_error = q_target - q_current
        self.losses.append(td_error)

        key = self._make_key(state)
        self._table[key][action] += self.meta_ctrl.learning_rate * td_error

    def _make_key(self, state: np.ndarray) -> tuple:
        """Convert a state vector to a hashable dictionary key."""

        return tuple(state.ravel().astype(int))

    def save_state(self, dir: str) -> None:
        """Save the estimator state to disk"""

        # TODO save Q-Learning state

    def load_state(self, dir: str) -> None:
        """Load a previously saved estimator state from disk"""

        # TODO Load Q-Learning state
