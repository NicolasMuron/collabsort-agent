"""
Noisy Networks for Exploration extension for DQN.
"""

import math
import torch
import torch.nn as nn
import numpy as np

from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.dqn import DQN


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer.
    Generates its own internal noise for autonomous exploration.
    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Trainable parameters (Mean mu and Standard Deviation sigma)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # PyTorch buffers to store random noise matrices (non-trainable)
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Parameter initialization according to Fortunato et al. (2017)."""
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Generates factorized Gaussian noise."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self) -> None:
        """Resample noise tensors for both weights and biases."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        # On extrait l'attribut brut en garantissant au typeur qu'il s'agit d'un Tensor
        w_eps = getattr(self, "weight_epsilon")
        b_eps = getattr(self, "bias_epsilon")

        if isinstance(w_eps, torch.Tensor) and isinstance(b_eps, torch.Tensor):
            w_eps.copy_(torch.outer(epsilon_out, epsilon_in))
            b_eps.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass sampling weights on the fly if training."""
        if self.training:
            w_eps = getattr(self, "weight_epsilon")
            b_eps = getattr(self, "bias_epsilon")

            assert isinstance(w_eps, torch.Tensor)
            assert isinstance(b_eps, torch.Tensor)

            weight = self.weight_mu + self.weight_sigma * w_eps
            bias = self.bias_mu + self.bias_sigma * b_eps
            return torch.nn.functional.linear(x, weight, bias)
        else:
            return torch.nn.functional.linear(x, self.weight_mu, self.bias_mu)


class NoisyQNetwork(nn.Module):
    """Q-Network architecture using NoisyLinear layers."""

    def __init__(
        self, input_size: int, output_size: int, hidden_sizes: tuple = (100, 100)
    ):
        super().__init__()
        # The first layer remains standard to extract baseline state features
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])

        # Subsequent layers are noisy to propagate exploration in depth
        self.noisy1 = NoisyLinear(hidden_sizes[0], hidden_sizes[1])
        self.noisy2 = NoisyLinear(hidden_sizes[1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.noisy1(x))
        return self.noisy2(x)

    def reset_noise(self):
        """Triggers noisy layers to sample fresh noise matrices."""
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


class NoisyDQN(DQN):
    """DQN Agent utilizing Noisy Networks instead of Epsilon-Greedy exploration."""

    def __init__(self, config: LearningConfig, n_actions: int, state_size: int):
        # 1. Call parent constructor to initialize baseline structures (buffer, device...)
        super().__init__(config=config, n_actions=n_actions, state_size=state_size)

        # 2. Properly instantiate noisy networks on the correct device
        self.q_network = NoisyQNetwork(state_size, n_actions).to(self.device)
        self.target_network = NoisyQNetwork(state_size, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 3. CRUCIAL: Recreate the optimizer to bind to the actual parameters of NoisyQNetwork
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(), lr=self.config.lr
        )

    def update_action_values(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        """Update action values and reset exploration noise at episode step boundaries."""
        super().update_action_values(state, action, reward, next_state, done)

        reset_q = getattr(self.q_network, "reset_noise", None)
        if callable(reset_q):
            reset_q()

        if done:
            reset_target = getattr(self.target_network, "reset_noise", None)
            if callable(reset_target):
                reset_target()

    def _optimize_network(self, loss: torch.Tensor) -> None:
        """Perform a gradient descent step and reset noisy network layers."""
        super()._optimize_network(loss)

        # On récupère la méthode dynamiquement pour contourner l'union type Tensor | Module
        reset_fn = getattr(self.q_network, "reset_noise", None)
        if callable(reset_fn):
            reset_fn()
