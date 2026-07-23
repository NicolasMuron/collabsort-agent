"""
Common definitions for decision-making algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from collabsort_agent.learning import ActionValueEstimator


@dataclass
class Config:
    """Decision configuration"""

    # Decision algorithm to use
    algorithm: Literal["eps", "ard", "gre", "pol"] = "eps"

    # ---------- Exploration decay ----------

    # Starting exploration probability
    epsilon_start: float = 1

    # Minimum exploration probability at the end of decay
    epsilon_min: float = 0.05

    # Exploration probability decay algorithm
    exploration_decay: Literal["lin", "exp"] = "lin"

    # Percentage of training time during which exploration probability is decayed
    decay_span: float = 0.8

    # If enabled, reset the exploration decay at the start of each curriculum phase.
    reset_exploration_per_phase: bool = False

    # ---------- Advantage Racing Diffusion ----------

    # Rule for ending evidence accumulation and choosing an action
    decision_rule: Literal["win-all"] = "win-all"

    # Initial value for decision threshold (adjusted via metacognition)
    theta_start: float = 1.0

    # Minimum decision threshold
    theta_min: float = 0.2

    # Maximum decision threshold
    theta_max: float = 3.0

    # Weight of the advantage (Q_i - Q_j) term
    w_d: float = 1.0

    # Weight of the sum (Q_i + Q_j) term
    w_s: float = 0.1

    # Urgency / baseline drift added to every accumulator
    V_0: float = 0.1

    # Mean of accumulation noise
    noise_mean: float = 0.0

    # Standard deviation of accumulation noise (denoted s in Miletic2021 paper)
    noise_std: float = 0.03

    # Safety cap on the inner accumulation loop
    max_steps: int = 100

    # Euler-Maruyama timestep
    dt: float = 0.01


class Deliberator(ABC):
    """Base class for decision-making algorithms."""

    def __init__(
        self, config: Config, estimator: ActionValueEstimator, rng: np.random.Generator
    ) -> None:
        self.config = config
        self.estimator = estimator
        self.rng = rng

    @abstractmethod
    def choose_action(
        self,
        state: np.ndarray,
        training_step: int,
    ) -> int:
        """Choose the action to perform"""

    def reset_for_phase(self, phase_steps: int) -> None:
        """Reset any phase-dependent exploration state at the start of a new phase."""
        return None

    @abstractmethod
    def log_episode(self, logger: SummaryWriter, episode: int) -> None:
        """Log information after an episode"""

    @abstractmethod
    def save_state(self, dir: str) -> None:
        """Save the deliberator state to disk"""

    @abstractmethod
    def load_state(self, dir: str) -> None:
        """Load a previously saved deliberator state from disk"""

    @property
    def state_filename(self) -> str:
        """Return the file name for saving/loading deliberator state"""

        return "deliberator.pth"
