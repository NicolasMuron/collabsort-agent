"""
Unit tests for perception.
"""

import gymnasium as gym
import numpy as np
from gym_collabsort.config import Config as EnvConfig

from collabsort_agent.perception import Config as PerceptionConfig
from collabsort_agent.perception import Perceiver


def make_perceiver(
    n_perceived_cols: int = 3, n_objects: int = 1, cone_perception: bool = False
) -> tuple[Perceiver, EnvConfig]:
    """Helper function to create a Perceiver object."""

    env_config = EnvConfig(n_objects=n_objects)
    perceiver = Perceiver(
        config=PerceptionConfig(
            n_perceived_cols=n_perceived_cols, cone_perception=cone_perception
        ),
        treadmill_rows=env_config.treadmill_rows,
        upper_treadmill_row=env_config.upper_treadmill_row,
        middle_treadmill_row=env_config.middle_treadmill_row,
    )
    return perceiver, env_config


def sample_obs(env_config: EnvConfig) -> dict:
    """Helper function to sample an observation from the environment."""

    env = gym.make("CollabSort-v0", config=env_config)
    obs, _ = env.reset()
    env.close()
    return obs


def test_perceiver_state_size() -> None:
    for cone_mode in (False, True):
        for n_perceived_cols in (1, 2, 3, 4, 5, 6):
            perceiver, env_config = make_perceiver(
                n_perceived_cols=n_perceived_cols, cone_perception=cone_mode
            )
            obs = sample_obs(env_config=env_config)

            sensory_state = perceiver.get_sensory_state(obs=obs)

            assert sensory_state.ndim == 1

            # Calculate expected length of sensory state
            total_cols = 0
            for row in perceiver.treadmill_rows:
                if cone_mode:
                    if row == env_config.upper_treadmill_row:
                        total_cols += n_perceived_cols + 2
                    elif row == env_config.middle_treadmill_row:
                        total_cols += n_perceived_cols + 1
                    else:
                        total_cols += n_perceived_cols
                else:
                    total_cols += n_perceived_cols

            expected_len = 3 + 2 + (total_cols * 3)
            assert len(sensory_state) == expected_len


def test_perceiver_state_content() -> None:
    # Test the content of the sensory state
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
    presence_indices = range(5, len(sensory_state), 3)
    for i in presence_indices:
        assert sensory_state[i] in (0.0, 1.0), (
            f"Presence flag at index {i} should be 0 or 1"
        )

    # Check state consistency for same observation
    s1 = perceiver.get_sensory_state(obs=obs)
    s2 = perceiver.get_sensory_state(obs=obs)
    np.testing.assert_array_equal(s1, s2)
