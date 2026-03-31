"""
Unit tests for perception.
"""

import gymnasium as gym
from gym_collabsort.config import Config as EnvConfig

from collabsort_agent.perception import Config as PerceptionConfig
from collabsort_agent.perception import Perceiver


def test_perceiver() -> None:
    """Test the Perceiver class"""

    env_config = EnvConfig(n_objects=1)

    # Initialize environment
    env = gym.make("CollabSort-v0", config=EnvConfig(n_objects=1))
    sample_obs, _ = env.reset()

    # Create perceiver
    perceiver = Perceiver(
        config=PerceptionConfig(),
        treadmill_rows=[env_config.upper_treadmill_row, env_config.lower_treadmill_row],
    )

    sensory_state = perceiver.get_sensory_state(obs=sample_obs)

    # Check that sensory state is a vector with the expected number of features:
    # - 3 for the agent (coords + presence of a picked object)
    # - 2 for the robot (coords)
    # - 3 for each perceived position (presence, color and shape of the object)
    assert sensory_state.ndim == 1
    assert len(sensory_state) == 3 + 2 + (
        len(perceiver.treadmill_rows) * perceiver.config.n_perceived_cols * 3
    )
