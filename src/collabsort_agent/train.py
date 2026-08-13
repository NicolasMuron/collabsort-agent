"""
Train an agent.
"""

import copy
import json
import os
import time
from dataclasses import dataclass

import gymnasium as gym

# Use a non-interactive backend for matplotlib to avoid Tkinter
# objects being created on worker threads (prevents Tkinter cleanup errors)
import matplotlib
import numpy as np
import torch
import tyro
from gym_collabsort.config import Action, RobotStrategy
from gym_collabsort.config import Config as EnvConfig
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from collabsort_agent.config import load_cfg

matplotlib.use("Agg")

from collabsort_agent.common import EpisodeMetrics, create_agent
from collabsort_agent.config import Config, save_cfg
from collabsort_agent.decision.epsilon_greedy import EpsilonGreedy
from collabsort_agent.metrics_tracker import HeatmapTracker


@dataclass
class CurriculumPhase:
    """A phase in the curriculum learning process"""

    name: str
    n_episodes: int
    env_config: EnvConfig


@dataclass
class TrainArgs:
    """Arguments for training."""

    config: Config
    curriculum_file: str | None = None
    pretrained_state_dir: str | None = None


def load_phases(
    base_config: Config, json_path: str | None, pretrained_state_dir: str | None = None
) -> list[CurriculumPhase]:
    """Load curriculum phases from a JSON file and update base_config treadmills."""

    phases = []
    all_active_treadmills = set(base_config.env.active_treadmills)

    if json_path is None:
        phases.append(
            CurriculumPhase(
                name="Default Phase",
                n_episodes=base_config.n_episodes,
                env_config=copy.deepcopy(base_config.env),
            )
        )
    else:
        print(f"Loading curriculum from {json_path}...")
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

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

    if pretrained_state_dir is not None and os.path.exists(pretrained_state_dir):
        pretrained_cfg = load_cfg(dir=pretrained_state_dir)
        all_active_treadmills.update(pretrained_cfg.env.active_treadmills)
        base_config.perception = pretrained_cfg.perception
        base_config.memory = pretrained_cfg.memory
        print(
            "Perception and Memory configs overridden by the pretrained model's config."
        )

    # Crucial step for zero-padding: the agent's initial perceiver must have ALL treadmills
    # that will be used across the entire curriculum, to initialize the correct network size.
    base_config.env.active_treadmills = tuple(sorted(all_active_treadmills))
    print(
        f"Agent's perceiver initialized with global active treadmills: {base_config.env.active_treadmills}"
    )

    if json_path is not None:
        print(f"Successfully loaded {len(phases)} phases.")

    return phases


def train(
    base_config: Config,
    phases: list[CurriculumPhase],
    pretrained_state_dir: str | None = None,
) -> None:
    """Execute training over one or multiple curriculum phases."""

    # Allow PyTorch to use TF32 (tensor float 32) on Ampere+ GPUs.
    torch.set_float32_matmul_precision("high")

    # Determine if we are in curriculum mode
    is_curriculum = len(phases) > 1
    prefix = "train_curriculum" if is_curriculum else "train"

    # Create directory path for training output
    train_dir = f"runs/{prefix}_{int(time.time())}_{base_config.decision.algorithm}_{base_config.learning.algorithm}"

    logger = None
    if base_config.log_events:
        logger = SummaryWriter(f"{train_dir}", flush_secs=60)

    # Temporary environment to sample observation space for agent creation
    # We use base_config.env because it has been updated in load_phases to include all
    # treadmills across the curriculum, ensuring the agent is sized for the max observation.
    temp_env = gym.make("CollabSort-v0", config=base_config.env)

    # Create agent
    agent = create_agent(
        config=base_config,
        sample_obs=temp_env.observation_space.sample(),
        rng=temp_env.np_random,
    )

    if pretrained_state_dir is not None:
        print(f"Loading pretrained agent state from: {pretrained_state_dir}")
        agent.load_state(dir=pretrained_state_dir)

    # Calculate total training steps across all phases for exploration decay reset
    total_training_steps = sum(
        p.n_episodes * base_config.n_steps_episode for p in phases
    )

    # Reset exploration decay if using curriculum and not resetting per phase
    if (
        is_curriculum
        and not base_config.decision.reset_exploration_per_phase
        and isinstance(agent.deliberator, EpsilonGreedy)
    ):
        agent.deliberator.exploration_decay.reset(total_steps=total_training_steps)

    temp_env.close()

    # Initialize global training step and episode counters
    global_training_step: int = 0
    global_episode: int = 0
    start_time = time.time()
    heatmap_tracker = HeatmapTracker(log_freq=20)

    # Loop over each curriculum phase
    for phase_idx, phase in enumerate(phases):
        if is_curriculum:
            print(f"\n{'=' * 50}")
            print(f"Starting Phase {phase_idx + 1}/{len(phases)}: {phase.name}")
            print(f"{'=' * 50}\n")

        # Reset agent's exploration decay if required by configuration
        if is_curriculum and base_config.decision.reset_exploration_per_phase:
            phase_steps = max(1, phase.n_episodes * base_config.n_steps_episode)
            agent.deliberator.reset_for_phase(phase_steps=phase_steps)

        # Create the environment for this specific phase
        env = gym.make("CollabSort-v0", config=phase.env_config)
        agent.reset()

        # If the agent's memory requires past-state input, perform a dummy call to get_extended_state to ensure the memory is initialized correctly.
        if agent.memory.requires_past_state():
            sample_obs = env.observation_space.sample()
            sample_sensory = agent.perceiver.get_sensory_state(obs=sample_obs)
            expected_size = len(agent.perceiver.get_past_state(obs=sample_obs))
            agent.memory.get_extended_state(
                sensory_state=sample_sensory,
                expected_past_state_size=expected_size,
            )

        phase_training_step = 0
        desc = f"Phase {phase_idx + 1}" if is_curriculum else "Training progress"

        # Loop over episodes for this phase
        for _ in trange(phase.n_episodes, desc=desc):
            # Reset environment for new episode
            obs, _ = env.reset()
            ep_metrics = EpisodeMetrics()
            action_history: list[int] = []
            ep_over: bool = False
            episode_visitation = np.zeros(base_config.env.n_rows + 1, dtype=np.int32)
            episode_collisions = np.zeros(base_config.env.n_rows + 1, dtype=np.int32)
            episode_spatial_actions: dict[int, dict[int, int]] = {}

            # Previous action string for oscillation counting
            prev_action_str: str | None = None

            # Episode loop
            while not ep_over:
                # Extract agent row for spatial tracking
                agent_row = int(obs["self"]["coords"][0])

                # Record visitation
                if 1 <= agent_row <= base_config.env.n_rows:
                    episode_visitation[agent_row] += 1

                # Agent chooses an action
                decision_step = (
                    phase_training_step
                    if (
                        is_curriculum
                        and base_config.decision.reset_exploration_per_phase
                    )
                    else global_training_step
                )
                action: Action = agent.act(obs=obs, training_step=decision_step)

                # Compute best possible immediate reward from this state
                if (
                    base_config.env.enable_reward_change
                    and phase_training_step >= base_config.env.reward_change_step
                ):
                    active_agent_rewards = base_config.env.agent_rewards_after
                else:
                    active_agent_rewards = base_config.env.agent_rewards

                agent_row = int(obs["self"]["coords"][0])
                robot_row = int(obs["robot"][0])
                dist = agent_row - robot_row

                # Action NONE
                reward_none = float(base_config.env.step_reward)
                if dist == 1:
                    reward_none += float(base_config.env.collision_penalty)

                # Action DOWN
                reward_down = float(
                    base_config.env.step_reward + base_config.env.movement_penalty
                )

                # Action UP
                reward_up = float(
                    base_config.env.step_reward + base_config.env.movement_penalty
                )
                if dist <= 2:
                    reward_up += float(base_config.env.collision_penalty)

                # Action PICK
                reward_pick = float(
                    base_config.env.step_reward + base_config.env.failed_action_penalty
                )

                possible_rewards = {
                    Action.NONE: reward_none,
                    Action.DOWN: reward_down,
                    Action.UP: reward_up,
                    Action.PICK: reward_pick,
                }

                # Action PICK
                if obs["self"]["picked_object"] == 0:
                    agent_coords = obs["self"]["coords"]
                    pickable = [
                        obj
                        for obj in obs["moving_objects"]
                        if obj["coords"][0] == agent_coords[0]
                        and obj["coords"][1] == agent_coords[1]
                    ]
                    if len(pickable) == 1:
                        obj = pickable[0]
                        pick_val = float(
                            active_agent_rewards[obj["color"], obj["shape"]]
                        )
                        reward_pick = float(base_config.env.step_reward + pick_val)
                        if dist == 1:
                            reward_pick += float(base_config.env.collision_penalty)
                        possible_rewards[Action.PICK] = reward_pick

                best_reward = max(possible_rewards.values())
                ep_metrics.optimized_reward += best_reward

                # Find which actions were optimal
                best_actions = [
                    act for act, rew in possible_rewards.items() if rew == best_reward
                ]
                for act in best_actions:
                    ep_metrics.optimal_action_counts[act.name] += 1

                # Track if agent's action was optimal
                # Since action could be an integer or Action enum due to act() return type
                act_enum = Action(action)
                ep_metrics.agent_action_counts[act_enum.name] += 1
                if act_enum in best_actions:
                    ep_metrics.optimal_action_matches += 1
                    ep_metrics.optimal_matches_by_action[act_enum.name] += 1

                # Count oscillations (UP/DOWN direction changes)
                current_action_str = action.name

                if prev_action_str is not None and (
                    (prev_action_str == "UP" and current_action_str == "DOWN")
                    or (prev_action_str == "DOWN" and current_action_str == "UP")
                ):
                    ep_metrics.oscillations += 1

                prev_action_str = current_action_str
                if current_action_str in ("UP", "DOWN"):
                    ep_metrics.movement_actions += 1

                action_idx = int(action.value)
                action_history.append(action_idx)

                if agent_row not in episode_spatial_actions:
                    episode_spatial_actions[agent_row] = {}
                if action_idx not in episode_spatial_actions[agent_row]:
                    episode_spatial_actions[agent_row][action_idx] = 0
                episode_spatial_actions[agent_row][action_idx] += 1

                # Take action and observe result
                next_obs, reward, terminated, truncated, info = env.step(action=action)
                reward = float(reward)

                # Use this experience to update agent
                agent.update(
                    next_obs=next_obs, reward=reward, done=terminated or truncated
                )

                # Update episode metrics
                ep_metrics.reward += reward
                ep_metrics.collisions += info["n_collisions"]
                ep_metrics.collected_objects += info["n_placed_objects"]
                ep_metrics.robot_collected_objects += info.get(
                    "robot_placed_objects", 0
                )
                ep_metrics.missed_objects += info.get("n_fallen_objects", 0)

                if info.get("agent_picked_value") is not None:
                    ep_metrics.picked_values.append(info["agent_picked_value"])

                if (
                    info.get("agent_collision")
                    and 1 <= agent_row <= base_config.env.n_rows
                ):
                    episode_collisions[agent_row] += 1

                ep_metrics.step += 1

                # Move to next state
                phase_training_step += 1
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
            ep_metrics.log(logger=logger, episode=global_episode)
            agent.log_episode(logger=logger, episode=global_episode)

            # --- HEATMAPS ---
            try:
                n_actions = int(agent.deliberator.estimator.n_actions)
            except (AttributeError, TypeError):
                n_actions = len(Action)

            heatmap_tracker.update(
                action_history=action_history,
                picked_values=ep_metrics.picked_values,
                episode_visitation=episode_visitation,
                episode_collisions=episode_collisions,
                spatial_actions=episode_spatial_actions,
                n_actions=n_actions,
            )
            heatmap_tracker.log_heatmaps(
                logger=logger, episode=global_episode, n_actions=n_actions
            )

            # Increment global episode counter
            global_episode += 1

        # Close the environment for this phase
        env.close()

    if base_config.save_state:
        agent.save_state(dir=train_dir)
        save_cfg(config=base_config, dir=train_dir)
        print(f"\nTraining completed. State saved in {train_dir}")

    if logger is not None:
        logger.close()


if __name__ == "__main__":
    # Load configuration and arguments from CLI
    args: TrainArgs = tyro.cli(TrainArgs)

    # Build the curriculum phases from the JSON file
    curriculum_phases = load_phases(
        base_config=args.config,
        json_path=args.curriculum_file,
        pretrained_state_dir=args.pretrained_state_dir,
    )

    # Launch the training process
    train(
        base_config=args.config,
        phases=curriculum_phases,
        pretrained_state_dir=args.pretrained_state_dir,
    )
