import numpy as np
import pytest
import torch
from tests.test_learning.test_dqn import TestDQN
from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.n_step_learning import NStepLearning


class TestNStepLearning(TestDQN):  # Inherit from TestDQN to test DQN compliance
    
    def _make_dqn(self, **kwargs) -> NStepLearning:
        """Factory method override to inject NStepLearning instead of standard DQN."""
        config = LearningConfig(
            batch_size=kwargs.get('batch_size', 32),
            replay_buffer_size=kwargs.get('replay_buffer_size', 1000),
            target_network_sync_freq=kwargs.get('target_sync_freq', 100),
            gamma=0.99
        )
        
        # Default to 1 so that inherited tests (standard DQN behavior) pass.
        # But if the test explicitly requests another value, we apply it.
        n_step = kwargs.get('n_step', 1)
        
        agent = NStepLearning(
            config=config, 
            n_actions=kwargs.get('n_actions', 4), 
            state_size=kwargs.get('state_size', 5),
            n_step=n_step
        )
        agent.losses = []
        agent.mean_q_values = []
        return agent

    # -------------------------------------------------------------------------
    # N-STEP LEARNING SPECIFIC TESTS
    # -------------------------------------------------------------------------

    def test_initialization(self):
        """Verify N-step specific initialization attributes."""
        agent = self._make_dqn(n_step=5)
        assert agent.n_step == 5
        assert isinstance(agent.n_step_buffer, list)

    def test_n_step_replay_buffer_delay_with_n5(self) -> None:
        """Verify the global replay buffer update delay specifically for N=5."""
        state = np.zeros(5, dtype=np.float32)
        
        # Force n_step=5 here
        dqn = self._make_dqn(replay_buffer_size=10, n_step=5)

        # The first 4 transitions fill the local buffer, nothing goes to the global buffer
        for _ in range(4):
            dqn._store_transition(state=state, action=0, reward=1.0, next_state=state)
        assert len(dqn.replay_buffer) == 0
        assert len(dqn.n_step_buffer) == 4

        # The 5th transition triggers the first insertion into the global replay buffer
        dqn._store_transition(state=state, action=0, reward=1.0, next_state=state)
        assert len(dqn.replay_buffer) == 1

    def test_store_transition_fills_buffer_with_n3(self):
        """Verify the exact mathematical calculation of discounted rewards for N=3."""
        agent = self._make_dqn(n_step=3)
        state = np.zeros(5, dtype=np.float32)

        agent._store_transition(state, 0, 1.0, state, False)
        agent._store_transition(state, 1, 2.0, state, False)
        
        # 3rd step -> triggers N-step discounted return calculation
        agent._store_transition(state, 2, 3.0, state, False)
        assert len(agent.replay_buffer) == 1

        stored = agent.replay_buffer.buffer[0]
        expected_R = 1.0 + (0.99 * 2.0) + (0.99**2 * 3.0)
        assert pytest.approx(stored[2]) == expected_R

    def test_store_transition_done_flushes_buffer(self):
        """Verify that done=True forces an immediate flush even if N=5."""
        agent = self._make_dqn(n_step=5)
        state = np.zeros(5, dtype=np.float32)

        agent._store_transition(state, 0, 1.0, state, False)
        agent._store_transition(state, 0, 1.0, state, False)
        
        # Episode finished, the buffer must flush all remaining transitions immediately
        agent._store_transition(state, 0, 1.0, state, True)
        assert len(agent.n_step_buffer) == 0
        assert len(agent.replay_buffer) == 3