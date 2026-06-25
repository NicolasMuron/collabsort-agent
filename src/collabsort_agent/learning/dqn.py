"""
Deep Q-Learning algorithm.
"""

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collabsort_agent.learning import ActionValueEstimator
from collabsort_agent.learning import Config as LearningConfig


def get_device() -> torch.device:
    """Return accelerated device if available, or fall back to CPU"""

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


class QNetwork(nn.Module):
    """Neural network for Q-value estimation over all actions"""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: tuple = (100, 100),
    ) -> None:
        super().__init__()

        # Create network layers
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(in_features=prev_size, out_features=hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
        layers.append(nn.Linear(in_features=prev_size, out_features=output_size))

        self.net = nn.Sequential(*layers)

    def forward(self, x) -> torch.Tensor:
        return self.net(x)


class UniformReplayBuffer:
    """Classic replay buffer with uniform sampling (FIFO)."""

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        minibatch = random.sample(self.buffer, batch_size)
        states = np.array([t[0] for t in minibatch], dtype=np.float32)
        actions = np.array([t[1] for t in minibatch], dtype=np.int64)
        rewards = np.array([t[2] for t in minibatch], dtype=np.float32)
        next_states = np.array([t[3] for t in minibatch], dtype=np.float32)
        dones = np.array([t[4] for t in minibatch], dtype=np.float32)
        # Return None for index and weights because DQN doesn't need them
        return (states, actions, rewards, next_states, dones), None, None

    def __len__(self):
        return len(self.buffer)


class DQN(ActionValueEstimator):
    """Deep Q-Learning algorithm for estimating action values."""

    def __init__(self, config: LearningConfig, n_actions: int, state_size: int) -> None:
        super().__init__(config=config, n_actions=n_actions)

        self.device = get_device()
        self.state_size = state_size

        # Create Q-network for estimating action values
        self.q_network = self.build_network().to(self.device)
        # Create target network with fixed parameters (stabilizes training)
        self.target_network = self.build_network().to(self.device)

        # Use SmoothL1Loss (Huber) rather than MSELoss.
        # DQN targets can have large variance; Huber loss is less sensitive to
        # outlier rewards (acts like MAE for large errors, MSE for small ones).
        self.loss_fn = nn.SmoothL1Loss()

        self.optimizer = optim.Adam(
            params=self.q_network.parameters(), lr=self.config.lr
        )

        # Target network must not accumulate gradients
        self.target_network.eval()

        # Create replay buffer for training the Q-network
        self.replay_buffer = UniformReplayBuffer(config.replay_buffer_size)

        # Step counter used to decide when to sync the target network
        self.learning_step: int = 0

    def build_network(self) -> nn.Module:
        """Default Network (Vanilla / Double)."""
        return QNetwork(input_size=self.state_size, output_size=self.n_actions)

    def build_network(self) -> nn.Module:
        """Default Network (Vanilla / Double)."""
        return QNetwork(input_size=self.state_size, output_size=self.n_actions)

    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        # Convert NumPy array to PyTorch tensor
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # Compute Q-Values for current state
        with torch.no_grad():
            q_values = self.q_network(state_tensor)

        # Convert PyTorch tensor to NumPy array
        return q_values[0].detach().cpu().numpy()

    def update_action_values(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ):
        self._store_transition(
            state=state, action=action, reward=reward, next_state=next_state, done=done
        )
        self._learn()

    def _store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        """Store transition between states for future learning."""

        # Store transition in replay buffer
        self.replay_buffer.add(state, action, reward, next_state, done)

    def _get_next_q_values(self, next_states: torch.Tensor) -> torch.Tensor:
        """Compute the max Q-values for the next states using the target network (Vanilla DQN)."""
        with torch.no_grad():
            return self.target_network(next_states).max(1)[0]

    def _prepare_tensors(self, batch: list) -> tuple:
        """Prepare and convert a batch of transitions (5 or 6 elements) into PyTorch tensors."""
        # Sample a batch of past experiences from replay buffer
        unzipped = list(zip(*batch, strict=True))
        states, actions, rewards, next_states, dones = unzipped[:5]

        # Obtain PyTorch tensors from NumPy arrays.
        # torch.from_numpy avoids allocating new memory
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(
            1
        )
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(
            self.device
        )
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Clamp actions to valid range
        actions = torch.clamp(actions, 0, self.n_actions - 1)

        # If the batch contains the `actual_n` from the n-step learning.
        if len(unzipped) == 6:
            actual_ns = torch.tensor(
                unzipped[5], dtype=torch.float32, device=self.device
            )
            return states, actions, rewards, next_states, dones, actual_ns

        return states, actions, rewards, next_states, dones

    def _optimize_network(self, loss: torch.Tensor) -> None:
        """Perform the backpropagation step and updates the weights."""
        self.losses.append(loss.item())

        # Update Q-network parameters through a gradient descent step.
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent exploding gradients.
        # max_norm=10 is a common conservative bound for DQN.
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

    def _handle_target_sync(self) -> None:
        """Periodically sync the target network with the online network."""
        self.learning_step += 1
        if self.learning_step % self.config.target_network_sync_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def _compute_q_values_and_targets(
        self, tensors: tuple
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Calculate the current Q values and Q targets (broken down into DQN and PER)."""
        if len(tensors) == 6:
            states, actions, rewards, next_states, dones, _ = tensors
        else:
            states, actions, rewards, next_states, dones = tensors

        # Compute action values for the current states
        q_values = self.q_network(states).gather(1, actions).squeeze(1)
        self.mean_q_values.append(torch.mean(q_values).item())

        # Using target_network (not q_network) to compute Q-targets.
        # Using q_network here would defeat the purpose of the target network: the
        # same network would be used both to generate targets and to be updated,
        # creating a moving-target problem that destabilises training.
        with torch.no_grad():
            q_next = self._get_next_q_values(next_states)
            # Q_target = r + gamma * max_a' Q_target(s', a') * (1 - done)
            q_target = rewards + self.config.gamma * q_next * (1 - dones)

        return q_values, q_target

    def _learn(self) -> None:
        """Update the Q-network parameters."""

        if len(self.replay_buffer) < self.config.batch_size:
            return

        # Sample a batch of past experiences from replay buffer
        (states, actions, rewards, next_states, dones), _, _ = (
            self.replay_buffer.sample(self.config.batch_size)
        )

        # Sample a batch of past experiences from replay buffer
        batch = list(zip(states, actions, rewards, next_states, dones))

        # Prepare and convert a batch of transitions (5 or 6 elements) into PyTorch tensors.
        tensors = self._prepare_tensors(batch)

        # Calculate the current Q values and Q targets (broken down into DQN and PER).
        q_values, q_target = self._compute_q_values_and_targets(tensors)

        loss = self.loss_fn(q_values, q_target)

        # Perform the backpropagation step and updates the weights.
        self._optimize_network(loss)

        # Periodically sync the target network with the online network.
        self._handle_target_sync()

    def save_state(self, dir: str) -> None:
        Path(dir).mkdir(parents=True, exist_ok=True)
        file_path = f"{dir}/{self.state_filename}"
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            file_path,
        )

    def load_state(self, dir: str) -> None:
        file_path = f"{dir}/{self.state_filename}"
        checkpoint = torch.load(file_path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Target network is only used for inference during target computation.
        self.target_network.eval()
