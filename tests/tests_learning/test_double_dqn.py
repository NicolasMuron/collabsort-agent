"""
Unit tests for the Double DQN algorithm.
"""
import numpy as np
import torch
from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.double_dqn import DoubleDQN
from tests.tests_learning.test_dqn import TestDQN 

class TestDoubleDQN(TestDQN):
    
    def _make_dqn(
        self,
        n_actions: int = 4,
        state_size: int = 5,
        batch_size: int = 4,
        replay_buffer_size: int = 100,
        target_sync_freq: int = 500,
    ) -> DoubleDQN:
        config = LearningConfig(
            batch_size=batch_size,
            replay_buffer_size=replay_buffer_size,
            target_network_sync_freq=target_sync_freq,
        )
        return DoubleDQN(config=config, n_actions=n_actions, state_size=state_size)

    def test_double_dqn_decoupling_logic(self) -> None:
        """Verify that Double DQN actually uses the _get_next_q_values method."""
        state = np.zeros(5, dtype=np.float32)
        agent = self._make_dqn(state_size=len(state))
        
        # Create a test tensor
        next_states = torch.zeros((2, 5), dtype=torch.float32).to(agent.device)
        
        # Verify that the call to the modified method doesn't crash and returns the correct dimension
        with torch.no_grad():
            next_q_values = agent._get_next_q_values(next_states)
        
        assert next_q_values.shape == (2,)
        
    def test_double_dqn_mathematical_decoupling(self) -> None:
        """
        Comprehensively verifies that the action is selected by the q_network 
        and evaluated by the target_network (Double DQN logic).
        """
        n_actions = 3
        agent = self._make_dqn(n_actions=n_actions, state_size=5)
        
        # Create a dummy next state tensor
        next_states = torch.zeros((1, 5), dtype=torch.float32).to(agent.device)

        # We temporarily replace the networks with mock functions
        # to control their responses down to the last detail.
        
        # The Online network believes that the best stock is Stock 2 (score 15.0)
        agent.q_network = lambda state: torch.tensor([[5.0, 10.0, 15.0]], device=agent.device)
        
        # The Target network assigns different values. 
        # For Action 2 (chosen by the online player), it is 2.0.
        # For Action 1, it is 99.0 (the target's overall maximum, which the Vanilla DQN would mistakenly choose!)
        agent.target_network = lambda state: torch.tensor([[1.0, 99.0, 2.0]], device=agent.device)

        with torch.no_grad():
            next_q_values = agent._get_next_q_values(next_states)

        # --- VALIDATION ---
        # If this were a Vanilla DQN: it would take the maximum value from the Target Network -> 99.0
        # Since this is a Double DQN: 
        #   - The online player chooses action 2 (because 15.0 is the maximum of [5, 10, 15])
        #   - The target evaluates action 2 -> the returned value MUST be 2.0
        assert torch.allclose(next_q_values, torch.tensor([2.0], device=agent.device))