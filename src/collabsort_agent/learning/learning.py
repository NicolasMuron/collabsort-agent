"""
Common definitions for learning algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from torch.utils.tensorboard.writer import SummaryWriter


@dataclass
class Config:
    """Learning configuration"""

    # Learning algorithm to use
    algorithm: Literal["dqn"] = "dqn"

    # Discount factor for Temporal-Difference algorithms
    gamma: float = 0.99

    # Learning rate for gradient descent
    lr: float = 1e-3

    # Batch size for sampling from replay buffer
    batch_size: int = 64

    # Size of the DQN replay buffer
    replay_buffer_size: int = 10000

    # Interval in learning steps to copy online weights to target network.
    target_network_sync_freq: int = 500


class ActionValueEstimator(ABC):
    """Base class for action value estimators."""

    def __init__(
        self,
        config: Config,
        n_actions: int,
    ) -> None:
        self.config = config
        self.n_actions = n_actions

    @abstractmethod
    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        """Return the action values for all actions"""

    @abstractmethod
    def update_action_values(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ):
        """Update action values after an action was taken"""

    @abstractmethod
    def log_episode(self, logger: SummaryWriter, episode: int) -> None:
        """Log information after an episode"""

    @abstractmethod
    def save_state(self, dir: str) -> None:
        """Save the estimator state to disk"""

    @abstractmethod
    def load_state(self, dir: str) -> None:
        """Load a previously saved estimator state from disk"""

    @property
    def state_filename(self) -> str:
        """Return the file name for saving/loading estimator state"""

        return "estimator.pth"
