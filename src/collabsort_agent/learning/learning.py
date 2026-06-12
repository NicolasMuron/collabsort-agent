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
    algorithm: Literal["ql", "dqn", "dueling_dqn", "double_dqn", "dd_dqn"] = "dqn"

    # Discount factor for Temporal-Difference algorithms
    gamma: float = 0.95

    # Learning rate for gradient descent
    lr: float = 1e-3

    # Initial value for variable learning rate (adjusted via metacognition)
    alpha_start: float = 0.1

    # Minimum variable learning rate
    alpha_min: float = 0.01

    # Maximum variable learning rate
    alpha_max: float = 0.5

    # Batch size for sampling from replay buffer
    batch_size: int = 128

    # Size of the DQN replay buffer
    replay_buffer_size: int = 50000

    # Interval in learning steps to copy online weights to target network.
    target_network_sync_freq: int = 2000

    # Initial Q-Value
    q_start: float = 0


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
        # --- TENSORBOARD : AVANTAGES QUAND V EST FAIBLE ---
        # On vérifie si l'action 0 a enregistré des données ce tour-ci
        if hasattr(self, 'low_v_advantages'):
            if len(self.low_v_gaps) > 0 and len(self.high_v_gaps) > 0:
                mean_low = sum(self.low_v_gaps) / len(self.low_v_gaps)
                mean_high = sum(self.high_v_gaps) / len(self.high_v_gaps)
                
                logger.add_scalars(
                    "Action_gap_V",
                    {
                        "Low_V": mean_low,
                        "High_V": mean_high
                    },
                    episode
                )

        # Reset episode data
        self.losses.clear()
        self.mean_q_values.clear()
        if hasattr(self, 'low_v_advantages'):
            self.low_v_gaps.clear()
            self.high_v_gaps.clear()

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
