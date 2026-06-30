"""
Train an agent using Curriculum Learning.
"""

import copy
import json
import time
from dataclasses import dataclass

import gymnasium as gym
import tyro
from gym_collabsort.config import Action, Config as EnvConfig, RobotStrategy
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from collabsort_agent.config import Config, save_cfg
from collabsort_agent.train import EpisodeMetrics, create_agent


@dataclass
class CurriculumPhase:
    """A phase in the curriculum learning process"""

    name: str
    n_episodes: int
    env_config: EnvConfig


@dataclass
class CurriculumArgs:
    """Arguments for curriculum training."""

    # Base configuration for the agent and environment
    config: Config

    # Path to the curriculum JSON file
    curriculum_file: str = "curriculum.json"


def load_curriculum_from_json(
    base_config: Config, json_path: str
) -> list[CurriculumPhase]:
    """Load curriculum phases from a JSON file and update base_config treadmills."""

    print(f"Loading curriculum from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    phases = []
    all_active_treadmills = set(base_config.env.active_treadmills)

    for phase_data in data:
        env_config = copy.deepcopy(base_config.env)

        # Apply overrides
        for k, v in phase_data.get("env_overrides", {}).items():
            if k == "robot_strategy":
                v = RobotStrategy(v)
            elif k == "active_treadmills":
                v = tuple(v)  # Ensure it is a tuple as expected by the environment
                all_active_treadmills.update(v)
            setattr(env_config, k, v)

        phases.append(
            CurriculumPhase(
                name=phase_data["name"],
                n_episodes=phase_data["n_episodes"],
                env_config=env_config,
            )
        )

    # Crucial step for zero-padding: the agent's initial perceiver must have ALL treadmills
    # that will be used across the entire curriculum, to initialize the correct network size.
    base_config.env.active_treadmills = tuple(sorted(list(all_active_treadmills)))
    print(
        f"Agent's perceiver initialized with global active treadmills: {base_config.env.active_treadmills}"
    )

    print(f"Successfully loaded {len(phases)} phases.")
    return phases


def train_curriculum(base_config: Config, phases: list[CurriculumPhase]) -> None:
    """Train an agent sequentially across multiple phases"""

    # Create directory path for training output
    train_dir: str = f"runs/train_curriculum_{int(time.time())}_{base_config.decision.algorithm}_{base_config.learning.algorithm}"

    logger = None
    if base_config.log_events:
        # Initialize logging
        logger = SummaryWriter(f"{train_dir}")

    # We use the first phase's environment config to initialize the observation space properly
    initial_env_config = phases[0].env_config if phases else base_config.env
    temp_env = gym.make("CollabSort-v0", config=initial_env_config)

    # Create agent ONCE. This agent will persist across all phases.
    agent = create_agent(
        config=base_config,
        sample_obs=temp_env.observation_space.sample(),
        rng=temp_env.np_random,
    )
    temp_env.close()

    # Training time step (= number of time steps since beginning of training)
    global_training_step: int = 0
    global_episode: int = 0
    start_time = time.time()

    # Loop over each curriculum phase
    for phase_idx, phase in enumerate(phases):
        print(f"\n{'=' * 50}")
        print(f"Starting Phase {phase_idx + 1}/{len(phases)}: {phase.name}")
        print(f"{'=' * 50}\n")

        # Create the environment for this specific phase
        env = gym.make("CollabSort-v0", config=phase.env_config)

        # Loop over episodes for this phase
        for _ in trange(phase.n_episodes, desc=f"Training Phase {phase_idx + 1}"):
            # Reset environment and metrics for new episode
            obs, _ = env.reset()
            ep_metrics = EpisodeMetrics()
            ep_over: bool = False

            # Episode loop
            while not ep_over:
                # Agent chooses an action
                action: Action = agent.act(
                    obs=obs,
                    training_step=global_training_step,
                )

                # Take action and observe result
                next_obs, reward, terminated, truncated, info = env.step(action=action)
                reward: float = float(reward)

                # Use this experience to update agent
                agent.update(
                    next_obs=next_obs,
                    reward=reward,
                    done=terminated or truncated,
                )

                # Update episode metrics
                ep_metrics.reward += reward
                ep_metrics.collisions += info["n_collisions"]
                ep_metrics.collected_objects += info["n_placed_objects"]
                ep_metrics.step += 1

                # Move to next state
                global_training_step += 1
                obs = next_obs
                ep_over = (
                    terminated
                    or truncated
                    or ep_metrics.step >= base_config.n_steps_episode
                )

            # Log episode data globally
            ep_metrics.sps = int(
                global_training_step / max(1, time.time() - start_time)
            )
            ep_metrics.log(
                logger=logger,
                episode=global_episode,
            )
            agent.log_episode(logger=logger, episode=global_episode)

            # Increment global episode counter
            global_episode += 1

        # Close the environment for this phase
        env.close()

    if base_config.save_state:
        agent.save_state(dir=train_dir)
        save_cfg(config=base_config, dir=train_dir)
        print(f"\nTraining completed. State saved in {train_dir}")


if __name__ == "__main__":
    # Load configuration and arguments from CLI
    args: CurriculumArgs = tyro.cli(CurriculumArgs)

    # Build the curriculum phases from the JSON file
    curriculum = load_curriculum_from_json(
        base_config=args.config, json_path=args.curriculum_file
    )

    # Launch curriculum learning
    train_curriculum(base_config=args.config, phases=curriculum)
