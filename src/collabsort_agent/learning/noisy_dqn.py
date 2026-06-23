"""
Noisy Networks for Exploration extension for DQN.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collabsort_agent.learning.dqn import DQN


class NoisyLinear(nn.Module):
    """
    Noisy Linear Layer.
    Generates its own internal noise for autonomous exploration.
    """
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
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

    def reset_noise(self):
        """Resamples new random noise matrices for the current step."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Outer product for weight noise, vector for bias noise
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # In training mode, apply the learned exploration noise
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # In evaluation/test mode, the network becomes deterministic (no noise)
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)


class NoisyQNetwork(nn.Module):
    """Q-Network architecture using NoisyLinear layers."""
    def __init__(self, input_size: int, output_size: int, hidden_sizes: tuple = (100, 100)):
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
    def __init__(self, config, state_size, n_actions):
        # 1. Call parent constructor to initialize baseline structures (buffer, device...)
        super().__init__(config, state_size, n_actions)
        
        # 2. Properly instantiate noisy networks on the correct device
        self.q_network = NoisyQNetwork(state_size, n_actions).to(self.device)
        self.target_network = NoisyQNetwork(state_size, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # 3. CRUCIAL: Recreate the optimizer to bind to the actual parameters of NoisyQNetwork
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.config.lr)

    def update_action_values(self, state, action, reward, next_state, done) -> None:
        """Override the interaction loop to force noise mutations at every single timestep."""
        # 1. Apply standard DQN storage and optimization triggers
        super().update_action_values(state, action, reward, next_state, done)
        
        # 2. NoisyNet rule: Resample noise after EVERY action taken in the environment
        self.q_network.reset_noise()

    def _optimize_network(self, loss) -> None:
        # Delegate backpropagation and optimization steps to baseline DQN
        super()._optimize_network(loss)
        
        # Resample noise after gradient steps to break temporal batch correlations
        self.q_network.reset_noise()
        self.target_network.reset_noise()