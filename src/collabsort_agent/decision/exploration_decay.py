"""
Exploration/exploitation ratio management.
"""

from abc import ABC, abstractmethod

import numpy as np

from collabsort_agent.decision import Config as DecisionConfig


class ExplorationDecay(ABC):
    """Abstract base class for exploration decay algorithms"""

    def __init__(self, config: DecisionConfig, total_steps: int) -> None:
        self.config = config

        # Number of steps during which exploration probability is decayed
        self.decay_steps: int = int(total_steps * self.config.decay_span)

    def get_epsilon(self, training_step: int) -> float:
        """Return the exploration probability epsilon"""

        epsilon_decayed = self._decay_epsilon(training_step=training_step)

        return max(epsilon_decayed, self.config.epsilon_min)

    def reset(self, total_steps: int) -> None:
        """Restart the decay from the beginning for a new training phase."""
        self.decay_steps = max(1, int(total_steps * self.config.decay_span))
        self._reset_state()

    @abstractmethod
    def _reset_state(self) -> None:
        """Recompute any cached state required by the decay schedule."""

    @abstractmethod
    def _decay_epsilon(self, training_step: int) -> float:
        """Compute the decayed value of exploration probability"""


class LinearExplorationDecay(ExplorationDecay):
    """Linear exploration decay"""

    def __init__(self, config: DecisionConfig, total_steps: int) -> None:
        super().__init__(config=config, total_steps=total_steps)
        self._reset_state()

    def _reset_state(self) -> None:
        # Pre-compute decay slope (ε_min - ε_start) / decay_steps
        self._decay_slope: float = (
            self.config.epsilon_min - self.config.epsilon_start
        ) / self.decay_steps

    def _decay_epsilon(self, training_step: int) -> float:
        # Decay epsilon linearly: ε = ε_start + t * decay_slope
        return self.config.epsilon_start + self._decay_slope * training_step


class ExponentialExplorationDecay(ExplorationDecay):
    """Exponential exploration decay"""

    def __init__(self, config: DecisionConfig, total_steps: int) -> None:
        super().__init__(config=config, total_steps=total_steps)
        self._reset_state()

    def _reset_state(self) -> None:
        # Maximal difference between current and minimum values of epsilon for stopping decay
        epsilon_delta = 0.01

        # Pre-compute the decay rate λ
        self._decay_rate: float = (
            -np.log(
                epsilon_delta / (self.config.epsilon_start - self.config.epsilon_min)
            )
            / self.decay_steps
        )

    def _decay_epsilon(self, training_step: int) -> float:
        # Decay epsilon exponentially: ε = ε_min + (ε_start - ε_min) * exp(-λt)
        return self.config.epsilon_min + (
            self.config.epsilon_start - self.config.epsilon_min
        ) * np.exp(-self._decay_rate * training_step)
