"""
Unit tests for the Double Dueling DQN (DD-DQN) algorithm.
"""
from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.dd_dqn import DoubleDuelingDQN

from tests.tests_learning.test_double_dqn import TestDoubleDQN
from tests.tests_learning.test_dueling_dqn import TestDuelingDQN

# Multiple inheritance: TestDoubleDuelingDQN inherits all tests from BOTH parents
class TestDoubleDuelingDQN(TestDoubleDQN, TestDuelingDQN):
    
    def _make_dqn(
        self,
        n_actions: int = 4,
        state_size: int = 5,
        batch_size: int = 4,
        replay_buffer_size: int = 100,
        target_sync_freq: int = 500,
    ) -> DoubleDuelingDQN:
        config = LearningConfig(
            batch_size=batch_size,
            replay_buffer_size=replay_buffer_size,
            target_network_sync_freq=target_sync_freq,
        )
        return DoubleDuelingDQN(config=config, n_actions=n_actions, state_size=state_size)