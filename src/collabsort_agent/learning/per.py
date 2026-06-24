"""
PER (Prioritized Experience Replay) algorithm.
"""

from typing import Any
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.dqn import DQN


class SumTree:
    """A binary tree structure for efficiently storing priorities."""

    data_pointer = 0

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def add(self, priority: float, data: tuple):
        """Add a transition with its initial priority."""
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def __len__(self):
        return self.n_entries

    def _propagate(self, idx: int, change: float):
        if idx <= 0:
            return
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, tree_index: int, priority: float):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        self._propagate(tree_index, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left_child_index = 2 * idx + 1
        right_child_index = left_child_index + 1

        # If we've reached the bottom of the tree (the leaves)
        if left_child_index >= len(self.tree):
            return idx

        if s <= self.tree[left_child_index]:
            return self._retrieve(left_child_index, s)
        else:
            # Security: Make sure you don't try to access a value greater than what the tree contains
            remaining_val = s - self.tree[left_child_index]
            return self._retrieve(right_child_index, remaining_val)

    def get_leaf(self, s: float) -> tuple[int, float, Any]:
        leaf_index = self._retrieve(0, s)
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def total_priority(self) -> float:
        return self.tree[0]


class PER(DQN):
    """Prioritized Experience Replay built on top of DQN."""

    def __init__(self, config: LearningConfig, n_actions: int, state_size: int) -> None:
        # Initialise DQN
        super().__init__(config=config, n_actions=n_actions, state_size=state_size)

        # Override loss_fn to avoid immediate average reduction (requires element-wise IS weights)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

        self.tree = SumTree(config.replay_buffer_size)
        self.replay_buffer = self.tree

        # PER hyperparameters (values aligned with Schaul et al. 2016, “proportional” variant)
        self.per_epsilon = 0.001  # Avoids zero priority
        self.per_alpha = 0.6  # Prioritization exponent
        self.per_beta = 0.4  # Importance Sampling weight initial
        self.ratio = 1.0  # Reaches 1 at the last step
        self.n_steps = (
            self.config.n_episodes * self.config.n_steps_episode * self.ratio
        )  # Total number of steps for progressive β increase
        self.per_beta_increment = (
            1 - self.per_beta
        ) / self.n_steps  # Progressive increase towards 1.0

        # Reward/TD-error clipping range, like in the original paper (numerical stability)
        self.per_clip_value = 1.0

        # The paper reduces the step-size by a factor of 4 compared to the baseline,
        # as prioritization increases the typical magnitude of gradients.
        # We recreate the optimizer with a learning rate specific to PER (without touching self.config.lr,
        # which remains the reference for other algorithms).
        self.per_lr = self.config.lr / 4
        self.optimizer = optim.Adam(params=self.q_network.parameters(), lr=self.per_lr)

    def _get_priority(self, error: float) -> float:
        return (abs(error) + self.per_epsilon) ** self.per_alpha

    def _store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        """Override the storage method to use SumTree with the maximum priority."""
        # Calculating the maximum dynamic priority among the inserted elements
        if self.tree.n_entries == 0:
            max_priority = 1.0
        else:
            start_idx = self.tree.capacity - 1
            end_idx = start_idx + self.tree.n_entries
            max_priority = np.max(self.tree.tree[start_idx:end_idx])
            if max_priority == 0:
                max_priority = 1.0

        self.tree.add(max_priority, (state, action, reward, next_state, done))

    def _sample(self) -> tuple[list, list, np.ndarray]:
        """Sample a batch from the SumTree with stratification and calculate the IS weights."""
        minibatch, idxs, priorities = [], [], []

        total_p = self.tree.total_priority()
        if total_p == 0:
            total_p = 1.0

        segment = total_p / self.config.batch_size

        # β gradually increases toward 1.0
        self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

        for i in range(self.config.batch_size):
            a = segment * i
            b = segment * (i + 1)

            value = np.random.uniform(a, min(b, total_p - 1e-5))

            idx, p, data = self.tree.get_leaf(value)

            # If the retrieved data is an integer (the 0 from initialization) or is invalid
            attempts = 0
            while (not isinstance(data, (tuple, list))) and attempts < 10:
                # We resample to a purely valid random value (lower in the tree)
                value = np.random.uniform(0, max(1e-5, total_p - 1e-2))
                idx, p, data = self.tree.get_leaf(value)
                attempts += 1

            # Avoid a strictly zero priority, which would cause Importance Sampling to crash
            p = max(p, self.per_epsilon)

            priorities.append(p)
            minibatch.append(data)
            idxs.append(idx)

        # Calculating IS weights
        sampling_probs = np.array(priorities) / total_p

        # Use self.tree.n_entries to reflect the ACTUAL number of stored elements
        is_weights = np.power(self.tree.n_entries * sampling_probs, -self.per_beta)

        # Security to avoid division by zero during normalization
        max_weight = is_weights.max()
        if max_weight == 0:
            max_weight = 1.0
        is_weights /= max_weight  # Normalisation

        return minibatch, idxs, is_weights

    def _learn(self) -> None:
        if self.tree.n_entries < self.config.batch_size:
            return

        # PER-Specific Sampling
        batch, idxs, is_weights = self._sample()

        # We'll let DQN handle the preparation of the standard tensors
        tensors = self._prepare_tensors(batch)
        is_weights_t = torch.tensor(is_weights, dtype=torch.float32, device=self.device)

        # Calculation of the element-wise loss weighted by the IS weights
        q_values, q_target = self._compute_q_values_and_targets(tensors)
        elementwise_loss = self.loss_fn(q_values, q_target)
        loss = (is_weights_t * elementwise_loss).mean()

        # Legacy Standard Optimization
        self._optimize_network(loss)

        # PER Feature: Tree Update
        td_errors = (q_target - q_values).abs().detach().cpu().numpy()
        for idx, error in zip(idxs, td_errors):
            self.tree.update(idx, self._get_priority(error))

        # Legacy Synchronization
        self._handle_target_sync()
