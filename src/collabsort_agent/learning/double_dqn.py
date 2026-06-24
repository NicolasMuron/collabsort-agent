"""
Double Deep Q-Learning algorithm
"""

import torch
from collabsort_agent.learning.dqn import DQN


class DoubleDQN(DQN):
    """Double DQN algorithm implementation"""

    def _get_next_q_values(self, next_states: torch.Tensor) -> torch.Tensor:
        """Double DQN calculation rule: the online network selects, the target network evaluates"""
        with torch.no_grad():
            # a* = argmax_a Q_online(s', a)
            next_state_actions = self.q_network(next_states).argmax(
                dim=-1, keepdim=True
            )
            # Q_target(s', a*)
            return (
                self.target_network(next_states)
                .gather(-1, next_state_actions)
                .squeeze(-1)
            )
