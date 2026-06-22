"""
N-step learning algorithm
"""
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.dqn import DQN


class NStepLearning(DQN):
    """N-step learning algorithm implementation"""
    
    def __init__(
        self,
        config: LearningConfig,
        n_actions: int,
        state_size: int,
        n_step: int = 1,  # Default value is 1 (like the classic DQN)
    ):
        super().__init__(config=config, n_actions=n_actions, state_size=state_size)
        self.n_step = n_step
        self.n_step_buffer = []  # Temporary local buffer

    def _store_transition(self, state, action, reward, next_state, done=False):
        """Store a transition in the N-step buffer and update the replay buffer global."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # At the end of the episode, we clear the entire remaining local buffer
        if done:
            while self.n_step_buffer:
                self._process_n_step()
            return

        if len(self.n_step_buffer) < self.n_step:
            return  # We're waiting until we've completed enough stages

        self._process_n_step()

    def _process_n_step(self):
        """Calculate the N-step return and push the 6-element tuple into the global buffer."""
        state_0, action_0, _, _, _ = self.n_step_buffer[0]
        _, _, _, next_state_n, done_n = self.n_step_buffer[-1]
        actual_n = len(self.n_step_buffer)

        # Calculation of the Cumulative Discounted Return (N-step discounted return)
        R = sum(
            [self.n_step_buffer[i][2] * (self.config.gamma ** i) for i in range(actual_n)]
        )
        
        # Store the data directly in the UniformReplayBuffer's deque.
        self.replay_buffer.buffer.append((state_0, action_0, R, next_state_n, done_n, actual_n))

        # We drag the window
        self.n_step_buffer.pop(0)

    def _learn(self) -> None:
        """Update the Q-network using the N-step Bellman targets."""
        if len(self.replay_buffer) < self.config.batch_size:
            return

        # Manual sampling from the deque to retrieve our 6-element tuples
        batch = random.sample(self.replay_buffer.buffer, self.config.batch_size)
        states, actions, rewards, next_states, dones, actual_ns = zip(*batch, strict=True)

        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        actual_ns = torch.tensor(actual_ns, dtype=torch.float32, device=self.device)

        actions = torch.clamp(actions, 0, self.n_actions - 1)

        q_values = self.q_network(states).gather(1, actions).squeeze(1)
        self.mean_q_values.append(torch.mean(q_values).item())

        with torch.no_grad():
            q_next = self._get_next_q_values(next_states)
            # Application of a dynamic gamma (gamma^actual_n) specific to each transition
            gamma_corrected = self.config.gamma ** actual_ns
            q_target = rewards + gamma_corrected * q_next * (1 - dones)

        loss = self.loss_fn(q_values, q_target)
        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learning_step += 1
        if self.learning_step % self.config.target_network_sync_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())