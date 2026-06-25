"""
Unit tests for the Rainbow DQN algorithm.
"""

import numpy as np
from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.rainbow_dqn import RainbowDQN


def test_rainbow_dqn_initialization() -> None:
    config = LearningConfig(
        batch_size=4,
        replay_buffer_size=10,
        target_network_sync_freq=10,
        n_step=3,
    )
    agent = RainbowDQN(config=config, n_actions=4, state_size=5, n_step=3)

    assert agent.n_step == 3
    assert hasattr(agent, "q_network")
    assert hasattr(agent, "target_network")
    assert agent.q_network is not None
    assert agent.target_network is not None


def test_rainbow_dqn_action_values_shape() -> None:
    config = LearningConfig(
        batch_size=4,
        replay_buffer_size=10,
        target_network_sync_freq=10,
        n_step=3,
    )
    agent = RainbowDQN(config=config, n_actions=4, state_size=5, n_step=3)

    state = np.zeros(5, dtype=np.float32)
    action_values = agent.get_action_values(state=state)

    assert action_values.shape == (4,)
