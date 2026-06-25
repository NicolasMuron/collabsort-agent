"""
Rainbow DQN algorithm combining Double DQN, Dueling, Noisy Nets, Prioritized Replay, and N-step returns.
"""

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.dueling_dqn import Dueling_Network
from collabsort_agent.learning.noisy_dqn import NoisyLinear
from collabsort_agent.learning.per import PER


class RainbowDQN(PER):
    """Rainbow DQN implementation with prioritized replay, n-step returns, noisy dueling network, and double DQN target logic."""

    def __init__(
        self,
        config: LearningConfig,
        n_actions: int,
        state_size: int,
        n_step: int = 1,
    ):
        self.n_step = n_step
        self.n_step_buffer: list[tuple[Any, ...]] = []
        super().__init__(config=config, n_actions=n_actions, state_size=state_size)

        # Recreate optimizer for the noisy dueling network parameters.
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.per_lr)

    def build_network(self) -> nn.Module:
        """Build a Noisy Dueling Q-network for Rainbow using shared dueling architecture."""
        return Dueling_Network(
            state_size=self.state_size,
            action_size=self.n_actions,
            linear_layer=NoisyLinear,
        )

    def _store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        """Store transitions using N-step buffering and prioritized replay."""
        self.n_step_buffer.append((state, action, reward, next_state, done))

        if done:
            while self.n_step_buffer:
                self._process_n_step()
            return

        if len(self.n_step_buffer) < self.n_step:
            return

        self._process_n_step()

    def _process_n_step(self) -> None:
        state_0, action_0, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_state_n, done_n = self.n_step_buffer[-1]
        actual_n = len(self.n_step_buffer)

        reward_n = sum(
            self.n_step_buffer[i][2] * (self.config.gamma**i) for i in range(actual_n)
        )

        if self.tree.n_entries == 0:
            max_priority = 1.0
        else:
            start_idx = self.tree.capacity - 1
            end_idx = start_idx + self.tree.n_entries
            max_priority = float(np.max(self.tree.tree[start_idx:end_idx]))
            if max_priority == 0.0:
                max_priority = 1.0

        self.tree.add(
            max_priority,
            (state_0, action_0, reward_n, next_state_n, done_n, actual_n),
        )

        self.n_step_buffer.pop(0)

    def _get_next_q_values(self, next_states: torch.Tensor) -> torch.Tensor:
        """Double DQN target computation."""
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=-1, keepdim=True)
            return self.target_network(next_states).gather(-1, next_actions).squeeze(-1)

    def _compute_q_values_and_targets(
        self, tensors: tuple
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate current Q values and N-step Q targets for Rainbow."""
        states, actions, rewards, next_states, dones, actual_ns = tensors

        q_values = self.q_network(states).gather(1, actions).squeeze(1)
        self.mean_q_values.append(torch.mean(q_values).item())

        with torch.no_grad():
            q_next = self._get_next_q_values(next_states)
            q_target = rewards + self.config.gamma**actual_ns * q_next * (1 - dones)

        return q_values, q_target

    def update_action_values(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ):
        super().update_action_values(state, action, reward, next_state, done)

        reset_fn = getattr(self.q_network, "reset_noise", None)
        if callable(reset_fn):
            reset_fn()

        if done:
            reset_target = getattr(self.target_network, "reset_noise", None)
            if callable(reset_target):
                reset_target()

    def _optimize_network(self, loss: torch.Tensor) -> None:
        super()._optimize_network(loss)

        reset_fn = getattr(self.q_network, "reset_noise", None)
        if callable(reset_fn):
            reset_fn()
