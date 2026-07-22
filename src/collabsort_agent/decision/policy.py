"""
Policy deliberator: sample actions from a stochastic policy provided by the estimator.
"""

from typing import Any

import numpy as np
from collabsort_agent.decision import Deliberator


class Policy(Deliberator):
    """Deliberator that samples actions from a probability distribution.

    Expects `estimator.get_action_values(state)` to return a probability vector
    over discrete actions (sums to 1). Uses the provided RNG for reproducible
    sampling.
    """

    def choose_action(self, state: Any, training_step: int) -> int:
        probs = self.estimator.get_action_values(state=state)
        # Ensure numpy array
        probs = np.asarray(probs, dtype=float)

        # If the estimator returned raw (possibly negative/unbounded) scores
        # (e.g. Q-values from a DQN), convert them to a probability vector via
        # a stable softmax. Otherwise, if already non-negative and sums to 1,
        # keep as-is.
        if not np.all(np.isfinite(probs)):
            probs = np.ones_like(probs, dtype=float) / len(probs)
        elif np.any(probs < 0) or probs.sum() <= 0:
            # stable softmax
            shifted = probs - np.max(probs)
            exp = np.exp(shifted)
            probs = exp / np.sum(exp)
        else:
            probs = probs / probs.sum()

        return int(self.rng.choice(len(probs), p=probs))

    def log_episode(self, logger, episode: int) -> None:
        pass

    def save_state(self, dir: str) -> None:
        pass

    def load_state(self, dir: str) -> None:
        pass
