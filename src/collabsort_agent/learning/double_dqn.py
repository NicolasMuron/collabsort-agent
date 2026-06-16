"""
Double Deep Q-Learning algorithm.
"""

import torch
import torch.nn as nn

from .dqn import DQN

class DDQN(DQN):
    """Double Deep Q-Learning algorithm for estimating action values."""
    
    def _learn(self) -> None:
        """Update weights using Double DQN target calculation rule"""
        if len(self.replay_buffer) < self.config.batch_size:
            return

        # Sample a batch of past experiences from replay buffer
        states, actions, rewards, next_states, dones = self._sample_batch()

        # Compute action values for the current states
        q_values = self.q_network(states)
        state_action_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # --- Double DQN target (differs from vanilla DQN) ---
            #
            # Vanilla DQN:
            #   best_next_action = argmax Q_target(s', a')   ← target net selects
            #   q_next           = Q_target(s', best_next)   ← target net evaluates
            #
            # Double DQN:
            #   best_next_action = argmax Q_online(s', a')   ← online net selects
            #   q_next           = Q_target(s', best_next)   ← target net evaluates
            #
            # Decoupling selection from evaluation removes the maximisation bias.
            
            # Step 1 — online network selects the best next action
            next_state_actions = self.q_network(next_states).max(1)[1].unsqueeze(1)
            
            # Step 2 — target network evaluates that action
            next_state_values = self.target_network(next_states).gather(1, next_state_actions).squeeze(1)
            
             # Q_target = r + gamma * Q_target(s', argmax_a Q_online(s', a)) * (1 - done)
            expected_state_action_values = rewards + (self.config.gamma * next_state_values * (1 - dones))

        loss = self.loss_fn(state_action_values, expected_state_action_values)

        # Enregistrement des métriques pour TensorBoard
        self.losses.append(loss.item())
        self.mean_q_values.append(state_action_values.mean().item())

        # Update Q-network parameters through a gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients.
        # max_norm=10 is a common conservative bound for DQN.
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        
        self.optimizer.step()

        # Periodically sync the target network with the online network.
        self.learning_step += 1
        if self.learning_step % self.config.target_network_sync_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())