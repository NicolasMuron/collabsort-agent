"""
Unit tests for perception.
"""

import gymnasium as gym
import numpy as np
from gym_collabsort.config import Config as EnvConfig

from collabsort_agent.perception import Config as PerceptionConfig
from collabsort_agent.perception import Perceiver


def make_perceiver(
    n_perceived_cols: int = 3, n_objects: int = 1
) -> tuple[Perceiver, EnvConfig]:
    """Helper function to create a Percevier object."""

    env_config = EnvConfig(n_objects=n_objects)
    perceiver = Perceiver(
        config=PerceptionConfig(n_perceived_cols=n_perceived_cols),
        treadmill_rows=env_config.treadmill_rows,
    )
    return perceiver, env_config


def sample_obs(env_config: EnvConfig) -> dict:
    """Helper function to sample an observation from the environment."""

    env = gym.make("CollabSort-v0", config=env_config)
    obs, _ = env.reset()
    env.close()
    return obs


def test_perceiver_state_size() -> None:
    for n_perceived_cols in (1, 3, 6):
        # Create perceiver and sample observation
        perceiver, env_config = make_perceiver(n_perceived_cols=n_perceived_cols)
        obs = sample_obs(env_config=env_config)

        sensory_state = perceiver.get_sensory_state(obs=obs)

        # Check that sensory state is a vector with the expected number of features:
        # - 3 for the agent (coords + presence of a picked object)
        # - 2 for the robot (coords)
        # - 3 for each perceived position (presence, color and shape of the object)
        assert sensory_state.ndim == 1
        expected_len = (
            3
            + 2
            + (len(perceiver.treadmill_rows) * perceiver.config.n_perceived_cols * 3)
        )
        assert len(sensory_state) == expected_len


def test_perceiver_state_content() -> None:
    # Create perceiver and sample observation
    perceiver, env_config = make_perceiver()
    obs = sample_obs(env_config=env_config)

    # Check agent coordinates
    sensory_state = perceiver.get_sensory_state(obs=obs)
    agent_row, agent_col = obs["self"]["coords"]
    assert sensory_state[0] == agent_row
    assert sensory_state[1] == agent_col

    # Check picked object flag
    assert sensory_state[2] == obs["self"]["picked_object"]

    # Check robot coordinates
    robot_row, robot_col = obs["robot"]
    assert sensory_state[3] == robot_row
    assert sensory_state[4] == robot_col

    # Check object presence flag.
    # Object slots start at index 5, every 3rd value is the presence flag
    presence_indices = range(5, len(sensory_state), 3)
    for i in presence_indices:
        assert sensory_state[i] in (0.0, 1.0), (
            f"Presence flag at index {i} should be 0 or 1"
        )

    # Check state consistency for same observation
    s1 = perceiver.get_sensory_state(obs=obs)
    s2 = perceiver.get_sensory_state(obs=obs)
    np.testing.assert_array_equal(s1, s2)
