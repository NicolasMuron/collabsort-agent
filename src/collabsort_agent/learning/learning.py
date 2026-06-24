"""
Common definitions for learning algorithms.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from statistics import mean
from typing import Literal

import numpy as np
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Config:
    """Learning configuration"""

    # Learning algorithm to use
    algorithm: Literal["ql", "dqn", "dueling_dqn", "ddqn", "dd_dqn", "per", "n_step", "noisy"] = "dqn"

    # Discount factor for Temporal-Difference algorithms
    gamma: float = 0.99

    # Learning rate for gradient descent
    lr: float = 1e-3

    # Initial value for variable learning rate (adjusted via metacognition)
    alpha_start: float = 0.1

    # Minimum variable learning rate
    alpha_min: float = 0.01

    # Maximum variable learning rate
    alpha_max: float = 0.5

    # Batch size for sampling from replay buffer
    batch_size: int = 256

    # Size of the DQN replay buffer
    replay_buffer_size: int = 100000

    # Number of steps for n-step returns (1 = standard DQN)
    n_step: int = 1

    # Interval in learning steps to copy online weights to target network.
    target_network_sync_freq: int = 500

    # Initial Q-Value
    q_start: float = 0
    
    # Number of training episodes
    n_episodes: int = 300

    # Maximal number of steps in an episode
    n_steps_episode: int = 1000

    # Number of training episodes
    n_episodes: int = 300

    # Maximal number of steps in an episode
    n_steps_episode: int = 1000


class ActionValueEstimator(ABC):
    """Base class for action value estimators."""

    def __init__(
        self,
        config: Config,
        n_actions: int,
    ) -> None:
        self.config = config
        self.n_actions = n_actions

        # Recorded loss values (used for logging)
        self.losses: list[float] = []

        # Average Q-values (used for logging)
        self.mean_q_values: list[float] = []

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

    def log_episode(self, logger: SummaryWriter, episode: int) -> None:
        logger.add_scalar(
            tag="learning/mean_td_error",
            scalar_value=mean(self.losses),
            global_step=episode,
        )
        logger.add_scalar(
            tag="learning/mean_q_value",
            scalar_value=mean(self.mean_q_values),
            global_step=episode,
        )     

        # Reset episode data
        self.losses.clear()
        self.mean_q_values.clear()

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
