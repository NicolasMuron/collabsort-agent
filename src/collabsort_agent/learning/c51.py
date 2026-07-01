"""
Categorical DQN (C51) algorithm.

This implementation follows the distributional RL formulation where the value
function is represented as a categorical distribution over a fixed support of
atoms. It reuses the shared DQN training loop and only overrides the parts that
are specific to the categorical distributional target.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.dqn import DQN, QNetwork


class C51Network(nn.Module):
    """Distributional network outputting logits over atoms for each action."""

    def __init__(
        self, input_size: int, output_size: int, hidden_sizes: tuple = (100, 100)
    ) -> None:
        super().__init__()
        self.backbone = QNetwork(
            input_size=input_size,
            output_size=hidden_sizes[-1],
            hidden_sizes=hidden_sizes[:-1],
        )
        self.head = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.head(features)


class C51(DQN):
    """Categorical DQN (C51) estimator."""

    def __init__(self, config: LearningConfig, n_actions: int, state_size: int) -> None:
        self.n_atoms = getattr(config, "n_atoms", 51)
        self.v_min = getattr(config, "v_min", -10.0)
        self.v_max = getattr(config, "v_max", 10.0)

        super().__init__(config=config, n_actions=n_actions, state_size=state_size)

        self.support: torch.Tensor = torch.linspace(
            self.v_min,
            self.v_max,
            self.n_atoms,
            dtype=torch.float32,
            device=self.device,
        )
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        self.loss_fn = nn.KLDivLoss(reduction="batchmean")
        self.optimizer = optim.Adam(
            params=self.q_network.parameters(), lr=self.config.lr
        )
        self.target_network.eval()

    def build_network(self) -> nn.Module:
        return C51Network(
            input_size=self.state_size,
            output_size=self.n_actions * self.n_atoms,
        )

    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        state_tensor = (
            torch.as_tensor(state, dtype=torch.float32)
            .unsqueeze(0)
            .to(self.device, non_blocking=True)
        )
        with torch.no_grad():
            logits = self.q_network(state_tensor)
        logits = logits.view(1, self.n_actions, self.n_atoms)
        probs = torch.softmax(logits, dim=-1)
        expected_values = (probs * self.support).sum(dim=-1)
        return expected_values[0].cpu().numpy()

    def _compute_q_values_and_targets(
        self, tensors: tuple
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(tensors) == 6:
            states, actions, rewards, next_states, dones, _ = tensors
        else:
            states, actions, rewards, next_states, dones = tensors

        current_logits = self.q_network(states).view(-1, self.n_actions, self.n_atoms)
        current_dist = torch.log_softmax(current_logits, dim=-1)
        current_action_logits = current_dist.gather(
            1,
            actions.unsqueeze(-1).expand(-1, 1, self.n_atoms),
        ).squeeze(1)

        expected_values = (torch.softmax(current_logits, dim=-1) * self.support).sum(
            dim=-1
        )
        chosen_q = expected_values.gather(1, actions.squeeze(1).unsqueeze(1)).squeeze(1)
        self.mean_q_values.append(chosen_q.mean().item())

        with torch.no_grad():
            next_logits = self.target_network(next_states).view(
                -1, self.n_actions, self.n_atoms
            )
            next_probs = torch.softmax(next_logits, dim=-1)
            next_q = (next_probs * self.support).sum(dim=-1)
            next_actions = next_q.argmax(dim=1)
            next_action_probs = next_probs.gather(
                1,
                next_actions.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, self.n_atoms),
            ).squeeze(1)

            target_probs = torch.zeros(
                (rewards.size(0), self.n_atoms),
                device=next_action_probs.device,
                dtype=next_action_probs.dtype,
            )
            projected = rewards.unsqueeze(
                1
            ) + self.config.gamma * self.support.unsqueeze(0) * (
                1.0 - dones.unsqueeze(1)
            )
            projected = projected.clamp(self.v_min, self.v_max)
            lower = torch.floor((projected - self.v_min) / self.delta_z).long()
            lower = torch.clamp(lower, 0, self.n_atoms - 1)
            upper = torch.clamp(lower + 1, 0, self.n_atoms - 1)
            fractional = (
                (projected - self.v_min) / self.delta_z
                - torch.floor((projected - self.v_min) / self.delta_z)
            ).to(next_action_probs.dtype)

            target_probs.scatter_add_(1, lower, (1.0 - fractional) * next_action_probs)
            target_probs.scatter_add_(1, upper, fractional * next_action_probs)

        return current_action_logits, target_probs
