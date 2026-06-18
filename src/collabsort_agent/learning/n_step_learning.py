"""
N-step learning algorithm
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.dqn import DQN


class NStepLearning(DQN):  # Inherit from DQN to reuse its methods and attributes
    
    def __init__(
        self,
        config: LearningConfig,
        n_actions: int,
        state_size: int,
        n_step: int = 7,  # Default to 3-step learning
    ):
        super().__init__(config=config, n_actions=n_actions, state_size=state_size)
        self.n_step = n_step
        self.n_step_buffer = []  # Buffer to store the last n steps

    def _store_transition(self, state, action, reward, next_state, done):
        """Store a transition in the N-step buffer and update the replay buffer."""
        self.n_step_buffer.append((state, action, reward, next_state, done))
        
        # If the episode ends, flush all remaining transitions correctly
        if done:
            while self.n_step_buffer:
                self._process_n_step()
            return

        if len(self.n_step_buffer) < self.n_step:
            return  # Wait until we have enough transitions

        self._process_n_step()

    def _process_n_step(self):
        """Calculate the N-step return and append the sequence to the global replay buffer."""
        # The starting state and action are taken from the oldest transition
        state_0, action_0, _, _, _ = self.n_step_buffer[0]
        
        # Le dernier état et le 'done' proviennent du DERNIER élément actuel du buffer
        _, _, _, next_state_n, done_n = self.n_step_buffer[-1]

        # L'exposant réel dépend de la taille actuelle du buffer (très important pour la fin d'épisode)
        actual_n = len(self.n_step_buffer)

        # Calculate the N-step discounted return
        R = sum(
            [self.n_step_buffer[i][2] * (self.config.gamma ** i) for i in range(actual_n)]
        )
        
        # RECOUVREMENT : On ajoute 'actual_n' dans le replay buffer pour adapter gamma pendant le learning
        self.replay_buffer.append((state_0, action_0, R, next_state_n, done_n, actual_n))

        # Remove the oldest transition to slide the window forward
        self.n_step_buffer.pop(0)

    def _learn(self) -> None:
        """Update the Q-network parameters using N-step Bellman targets."""
        if len(self.replay_buffer) < self.config.batch_size:
            return

        # Sample a batch of past experiences (now including actual_n)
        import random
        batch = random.sample(self.replay_buffer, self.config.batch_size)
        states, actions, rewards, next_states, dones, actual_ns = zip(*batch, strict=True)

        # Convert lists to PyTorch tensors
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # Tensor pour les n-steps spécifiques à chaque transition du batch
        actual_ns = torch.tensor(actual_ns, dtype=torch.float32, device=self.device)

        actions = torch.clamp(actions, 0, self.n_actions - 1)

        # Compute action values for the current states
        q_values = self.q_network(states).gather(1, actions).squeeze(1)
        self.mean_q_values.append(torch.mean(q_values).item())

        # Compute the next state action-values according to the Double DQN strategy
        with torch.no_grad():
            q_next = self._get_next_q_values(next_states)
            
            # CORRECTION : On applique un gamma dynamique propre à chaque échantillon du batch
            gamma_corrected = self.config.gamma ** actual_ns
            q_target = rewards + gamma_corrected * q_next * (1 - dones)

        loss = self.loss_fn(q_values, q_target)
        self.losses.append(loss.item())

        # Update Q-network parameters
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Periodically sync the target network
        self.learning_step += 1
        if self.learning_step % self.config.target_network_sync_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())