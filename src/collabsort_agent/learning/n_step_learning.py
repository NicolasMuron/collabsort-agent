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


class NStepLearning(DoubleDuelingDQN):  # Inherit from DoubleDuelingDQN to reuse its methods and attributes
    
    def __init__(
        self,
        config: LearningConfig,
        n_actions: int,
        state_size: int,
        n_step: int = 3,  # Default to 3-step learning
    ):
        # Initialize DoubleDuelingDQN parent class
        super().__init__(config=config, n_actions=n_actions, state_size=state_size)
        self.n_step = n_step
        self.n_step_buffer = []  # Buffer to store the last n steps

    def _store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the N-step buffer and update the replay buffer."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # If the episode ends, flush the remaining transitions from the temporary buffer
        if done:
            while self.n_step_buffer:
                self._process_n_step()
            return

        if len(self.n_step_buffer) < self.n_step:
            return  # Wait until we have enough transitions

        self._process_n_step()

    def _process_n_step(self):
        """Calculate the N-step return and append the sequence to the global replay buffer."""
        # The starting state and action are taken from the oldest transition in the buffer
        state_0, action_0, _, _, _ = self.n_step_buffer[0]
        
        # The final state and done flag must come from the latest transition in the buffer
        _, _, _, next_state_n, done_n = self.n_step_buffer[-1]

        # Calculate the N-step discounted return
        R = sum(
            [self.n_step_buffer[i][2] * (self.config.gamma ** i) for i in range(len(self.n_step_buffer))]
        )
        
        # Append to standard replay_buffer deque using the expected tuple structure
        self.replay_buffer.append((state_0, action_0, R, next_state_n, done_n))

        # Remove the oldest transition to slide the window forward
        self.n_step_buffer.pop(0)

    def _learn(self) -> None:
        """Update the Q-network parameters using N-step Bellman targets."""
        if len(self.replay_buffer) < self.config.batch_size:
            return

        # Sample a batch of past experiences from replay buffer
        import random
        batch = random.sample(self.replay_buffer, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch, strict=True)

        # Convert lists to PyTorch tensors
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)

        actions = torch.clamp(actions, 0, self.n_actions - 1)

        # Compute action values for the current states
        q_values = self.q_network(states).gather(1, actions).squeeze(1)
        self.mean_q_values.append(torch.mean(q_values).item())

        # Compute the next state action-values according to the Double DQN strategy
        with torch.no_grad():
            q_next = self._get_next_q_values(next_states)
            # Critical adjustment: Use gamma corrected to the power of n_step for the future value bootstraps
            q_target = rewards + (self.config.gamma ** self.n_step) * q_next * (1 - dones)

        loss = self.loss_fn(q_values, q_target)
        self.losses.append(loss.item())

        # Update Q-network parameters through a gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Periodically sync the target network with the online network
        self.learning_step += 1
        if self.learning_step % self.config.target_network_sync_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())