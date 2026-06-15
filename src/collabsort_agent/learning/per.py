"""
PER algorithm
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
    
class dueling_Network(nn.Module):
    """Dueling DQN architecture"""

    def __init__(self, state_size: int, action_size: int) -> None:
        super(dueling_Network, self).__init__()
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
            nn.Linear(50, 1)  # Output is a single value for the state
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, action_size)  # Output is an advantage for each action
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine value and advantages to get Q-values
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
    
class SumTree:
    data_pointer = 0

    # Here we initialize the tree with all nodes = 0, and initialize the data with all values = 0
    def __init__(self, capacity):
        # Number of leaf nodes (final nodes) that contains experiences
        self.capacity = capacity

        # Generate the tree with all nodes values = 0
        # To understand this calculation (2 * capacity - 1) look at the schema below
        # Remember we are in a binary node (each node has max 2 children) so 2x size of leaf (capacity) - 1 (root node)
        # Parent nodes = capacity - 1
        # Leaf nodes = capacity
        self.tree = np.zeros(2 * capacity - 1)

        # Contains the experiences (so the size of data is capacity)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    # Here we define function that will add our priority score in the sumtree leaf and add the experience in data:
    def add(self, priority, data):
        # Look at what index we want to put the experience
        tree_index = self.data_pointer + self.capacity - 1

        # Update data frame
        self.data[self.data_pointer] = data

        # Update the leaf
        self.update (tree_index, priority)

        # Add 1 to data_pointer
        self.data_pointer += 1

        if self.data_pointer >= self.capacity:  # If we're above the capacity, we go back to first index (we overwrite)
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # Update the leaf priority score and propagate the change through tree
    def update(self, tree_index, priority):
        # Change = new priority score - former priority score
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority

        # then propagate the change through tree
        # this method is faster than the recursive loop in the reference code
        self._propagate(tree_index, change)

    def _retrieve(self, idx, s):
        left_child_index = 2 * idx + 1
        right_child_index = left_child_index + 1
        if left_child_index >= len(self.tree):
            return idx
        if s <= self.tree[left_child_index]:
            return self._retrieve(left_child_index, s)
        else:
            return self._retrieve(right_child_index, s - self.tree[left_child_index])

    def get_leaf(self, s):
        leaf_index = self._retrieve(0, s)

        data_index = leaf_index - self.capacity + 1

        return (leaf_index, self.tree[leaf_index], self.data[data_index])

    def total_priority(self):
        return self.tree[0] # Returns the root node
    

class PER(ActionValueEstimator):
    """Prioritized Experience Replay built on top of Dueling DD-DQN."""

    def __init__(self, config: LearningConfig, n_actions: int, state_size: int) -> None:
        super().__init__(config=config, n_actions=n_actions)

        self.device = get_device()

        # Réseaux — même architecture que DD_DQN
        self.q_network = dueling_Network(state_size=state_size, action_size=n_actions).to(self.device)
        self.target_network = dueling_Network(state_size=state_size, action_size=n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.loss_fn = nn.SmoothL1Loss(reduction="none") 
        self.optimizer = optim.Adam(params=self.q_network.parameters(), lr=self.config.lr)

        # PER — remplace le deque de DD_DQN
        self.tree = SumTree(config.replay_buffer_size)

        # Hyperparamètres PER
        self.per_epsilon = 0.001        # évite priorité nulle
        self.per_alpha = 0.6            # exposant de prioritisation
        self.per_beta = 0.4             # IS weight initial
        self.per_beta_increment = 0.001 # β → 1 progressivement

        self.learning_step: int = 0

    def _get_priority(self, error: float) -> float:
        return (abs(error) + self.per_epsilon) ** self.per_alpha

    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values[0].detach().cpu().numpy()

    def update_action_values(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ):
        self._store_transition(state=state, action=action, reward=reward, next_state=next_state, done=done)
        self._learn()

    def _store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        """Stocke la transition avec priorité maximale (sera réévaluée au premier sample)."""
        max_priority = self.tree.tree.max() if self.tree.n_entries > 0 else 1.0
        self.tree.add(max_priority, (state, action, reward, next_state, done))

    def _sample(self):
        """Sample un batch depuis le SumTree avec stratification."""
        minibatch, idxs, priorities = [], [], []
        segment = self.tree.total_priority() / self.config.batch_size

        # β augmente progressivement vers 1.0
        self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

        for i in range(self.config.batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            idx, p, data = self.tree.get_leaf(value)
            priorities.append(p)
            minibatch.append(data)
            idxs.append(idx)

        # Calcul IS weights — corrige le biais d'échantillonnage
        sampling_probs = np.array(priorities) / self.tree.total_priority()
        is_weights = np.power(self.tree.n_entries * sampling_probs, -self.per_beta)
        is_weights /= is_weights.max()  # normalisation

        return minibatch, idxs, is_weights

    def _learn(self) -> None:
        """Update du réseau — même logique DD_DQN + pondération IS weights."""

        if self.tree.n_entries < self.config.batch_size:
            return

        batch, idxs, is_weights = self._sample()
        states, actions, rewards, next_states, dones = zip(*batch, strict=True)

        # Tenseurs — même pattern que DD_DQN
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        is_weights_t = torch.tensor(is_weights, dtype=torch.float32, device=self.device)

        actions = torch.clamp(actions, 0, self.n_actions - 1)

        # Q-values courantes
        q_values = self.q_network(states).gather(1, actions).squeeze(1)
        self.mean_q_values.append(torch.mean(q_values).item())

        with torch.no_grad():
            # Double DQN target — identique à DD_DQN
            best_next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            q_next = self.target_network(next_states).gather(1, best_next_actions).squeeze(1)
            q_target = rewards + self.config.gamma * q_next * (1 - dones)

        # Loss pondérée par IS weights — seule différence vs DD_DQN
        elementwise_loss = self.loss_fn(q_values, q_target)  # shape [batch]
        loss = (is_weights_t * elementwise_loss).mean()
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Mise à jour des priorités dans le SumTree
        td_errors = (q_target - q_values).abs().detach().cpu().numpy()
        for idx, error in zip(idxs, td_errors):
            self.tree.update(idx, self._get_priority(error))

        # Sync target network — même logique que DD_DQN
        self.learning_step += 1
        if self.learning_step % self.config.target_network_sync_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def save_state(self, dir: str) -> None:
        Path(dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            f"{dir}/{self.state_filename}",
        )

    def load_state(self, dir: str) -> None:
        checkpoint = torch.load(f"{dir}/{self.state_filename}", map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.target_network.eval()