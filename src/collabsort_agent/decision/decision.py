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

    # Deision algorithm to use
    algorithm: Literal["eps"] = "eps"

    # Starting exploration probability
    epsilon_start: float = 1

    # Minimum exploration probability at the end of decay
    epsilon_min: float = 0.05

    # Exploration probability decay algorithm
    exploration_decay: Literal["lin", "exp"] = "lin"

    # Percentage of training time during which exploration probability is decayed
    exploration_decay_span: float = 0.5


class Deliberator(ABC):
    """Base class for decision-making algorithms."""

    def __init__(self, config: Config, estimator: ActionValueEstimator) -> None:
        self.config = config
        self.estimator = estimator

    @abstractmethod
    def choose_action(self, state: np.ndarray, training_step: int | None) -> int:
        """Choose the action to perform"""

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
