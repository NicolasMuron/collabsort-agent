"""
N-step learning algorithm
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.dd_dqn import DoubleDuelingDQN

class NStepLearning(
    DoubleDuelingDQN
):  # Inherit from DoubleDuelingDQN to reuse its methods and attributes
    def __init__(
        self,
        config: LearningConfig,
        n_actions: int,
        n_step: int = 3,  # Default to 3-step learning
    ):
        super().__init__(config, n_actions)
        self.n_step = n_step
        self.n_step_buffer = []  # Buffer to store the last n steps

    def _store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the N-step buffer and update the replay buffer."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        if len(self.n_step_buffer) < self.n_step:
            return  # Wait until we have enough transitions

        # Calculate the N-step return
        R = sum(
            [self.n_step_buffer[i][2] * (self.config.gamma ** i) for i in range(self.n_step)]
        )
        state_n, action_n, _, next_state_n, done_n = self.n_step_buffer[0]
        self.replay_buffer.add(state_n, action_n, R, next_state_n, done_n)

        # Remove the oldest transition
        self.n_step_buffer.pop(0)