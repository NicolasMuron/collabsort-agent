"""
Code for demoing a previously trained agent.
"""

from pathlib import Path

import gymnasium as gym
import tyro
from gym_collabsort.config import Action, RenderMode

from collabsort_agent.config import load_cfg
from collabsort_agent.train import create_agent


def demo(train_dir: str) -> None:
    """Demonstrates a previously trained agent"""

    if not Path(train_dir).is_dir():
        raise Exception(f"Invalid path '{train_dir}'")

    # Load config used for training
    config = load_cfg(dir=train_dir)

    # Switch configuration to demo mode
    config.env.render_mode = RenderMode.HUMAN

    # Initialize environment
    env = gym.make("CollabSort-v0", config=config.env)

    # Create agent and load its state from disk
    agent = create_agent(
        config=config, sample_obs=env.observation_space.sample(), rng=env.np_random
    )
    agent.load_state(dir=train_dir)

    # Reset environment
    obs, _ = env.reset()
    ep_over: bool = False

    # Episode loop
    while not ep_over:
        # Agent chooses an action
        action: Action = agent.act(obs=obs, training_step=0)

        # Take action and observe result
        next_obs, _, terminated, truncated, info = env.step(action=action)

        # Move to next state
        obs = next_obs
        ep_over = terminated or truncated

    env.close()


if __name__ == "__main__":  # pragma: no cover
    # Run demo with command line args
    tyro.cli(demo)
