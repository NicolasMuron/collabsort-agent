"""
Dueling DQN algorithm
"""

import torch
import torch.nn as nn
import torch.optim as optim

from .dqn import DQN, get_device

class Dueling_Network(nn.Module):
    """Dueling DQN architecture."""
    def __init__(self, state_size: int, action_size: int) -> None:
        super(Dueling_Network, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, action_size)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine value and advantages to get Q-values: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values


class DuelingDQN(DQN):  
    """Dueling DQN algorithm implementation."""
    
    def __init__(self, config, n_actions: int, state_size: int) -> None:
        super().__init__(config=config, n_actions=n_actions, state_size=state_size)
        
       # Create Dueling-Network for estimating action values from dueling architecture
        self.q_network = Dueling_Network(state_size=state_size, action_size=n_actions).to(self.device)
        self.target_network = Dueling_Network(state_size=state_size, action_size=n_actions).to(self.device)
        
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()
        
        self.optimizer = optim.Adam(params=self.q_network.parameters(), lr=self.config.lr)