import copy
import copyreg
import sys
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import pygame
import tyro
from gym_collabsort.config import Action
from pygame import freetype
from torch.utils.tensorboard import SummaryWriter

from collabsort_agent.config import Config


# ---------------------------------------------------------------------------
# Make pygame.Surface and pygame.freetype.Font deepcopy-able.
# ---------------------------------------------------------------------------
def _surface_reduce(surface: pygame.Surface):
    size = surface.get_size()
    flags = surface.get_flags()
    return _surface_restore, (size, flags)


def _surface_restore(size, flags):
    pygame.display.init()
    return pygame.Surface(size, flags)


def _font_reduce(font):
    return _font_restore, ()


def _font_restore():
    freetype.init()
    return freetype.Font(None)


copyreg.pickle(pygame.Surface, _surface_reduce)
copyreg.pickle(freetype.Font, _font_reduce)
# ---------------------------------------------------------------------------


@dataclass
class EvalArgs:
    """Arguments for evaluation."""

    config: Config
    pretrained_state_dir: str | None = None
    beam_width: int = 10
    beam_heuristic_weight: float = 1.5
    start_episode: int = 0
    max_episode: int = 300
    episode_step: int = 50
    log_dir: str = "runs/oracle_vs_agent"


def _make_env(config: Config) -> gym.Env:
    """Create a new environment instance (no rendering)."""
    return gym.make("CollabSort-v0", config=config.env)


def _beam_heuristic(env: Any, weight: float) -> float:
    """Return a heuristic bonus for states where an object is reachable."""
    if weight <= 0.0:
        return 0.0

    agent_arm = env.board.agent_arm
    if agent_arm.picked_object is not None:
        return 0.0

    agent_coords = agent_arm.gripper.coords
    pickup_col = agent_arm.base.coords.col

    reachable_objects = []
    for obj in env.board.moving_objects:
        obj_coords = obj.coords
        col_distance = obj_coords.col - pickup_col
        row_distance = abs(obj_coords.row - agent_coords.row)
        if col_distance >= 0 and row_distance <= col_distance:
            reachable_objects.append((obj, col_distance, row_distance))

    if not reachable_objects:
        return 0.0

    best_value = max(
        float(obj.get_reward(rewards=env.current_agent_rewards))
        / (1.0 + 0.5 * row_dist)
        for obj, _, row_dist in reachable_objects
    )
    min_time = min(col_distance for _, col_distance, _ in reachable_objects)
    return weight * best_value / (1.0 + min_time)


def beam_search(
    config: Config,
    seed: int,
    beam_width: int,
    max_steps: int = 1000,
    heuristic_weight: float = 0.0,
) -> tuple[float, list[str]]:
    """Perform beam search for a given episode (seed)."""
    wrapper_env = _make_env(config)
    wrapper_env.reset(seed=seed)
    init_unwrapped: Any = wrapper_env.unwrapped

    beam: list[tuple[float, Any, list[str]]] = [
        (0.0, copy.deepcopy(init_unwrapped), [])
    ]

    for step in range(max_steps):
        print(
            f"Step {step + 1}/{max_steps} - Beam size: {len(beam)}",
            end="\r",
            flush=True,
        )
        candidates: list[tuple[float, float, Any, list[str]]] = []

        for score, env_snapshot, history in beam:
            for action in Action:
                env_clone: Any = copy.deepcopy(env_snapshot)
                _, reward, _, _, _ = env_clone.step(action.value)

                step_reward = float(reward)
                new_score = score + step_reward

                heuristic_bonus = _beam_heuristic(
                    env=env_clone, weight=heuristic_weight
                )

                if step_reward <= -5.0:
                    heuristic_bonus = 0.0

                eval_score = new_score + heuristic_bonus
                new_history = history + [action.name]
                candidates.append((eval_score, new_score, env_clone, new_history))

        if not candidates:
            break

        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = [(item[1], item[2], item[3]) for item in candidates[:beam_width]]

    print("\nBeam search completed.")
    best_score, _, best_history = beam[0]
    return best_score, best_history


def evaluate(
    config: Config,
    pretrained_state_dir: str | None,
    beam_width: int = 10,
    heuristic_weight: float = 1.0,
    start_episode: int = 0,
    max_episode: int = 300,
    episode_step: int = 50,
    log_dir: str = "runs/oracle_vs_agent",
) -> None:
    """Evaluate Agent vs Oracle on episodes 0, 50, 100, 150, 200, 250, 300."""

    print("========================================")
    print("Starting Evaluation (Agent vs Oracle)")
    print(f"Episodes: from {start_episode} to {max_episode} (step: {episode_step})")
    print(
        f"Agent Model: {pretrained_state_dir if pretrained_state_dir else 'Random (No model provided)'}"
    )
    print("========================================")

    writer = SummaryWriter(log_dir=log_dir)

    # Load agent
    env_agent = _make_env(config)
    from collabsort_agent.train import create_agent

    agent = create_agent(
        config=config,
        sample_obs=env_agent.observation_space.sample(),
        rng=env_agent.np_random,
    )
    if pretrained_state_dir is not None:
        agent.load_state(dir=pretrained_state_dir)

    if hasattr(agent, "eval") and callable(agent.eval):
        agent.eval()

    # Loop over episodes 0, 50, 100, ..., 300
    for episode_num in range(start_episode, max_episode + 1, episode_step):
        # The seed IS the episode number!
        seed = episode_num

        print("\n========================================")
        print(f"EVALUATING EPISODE {episode_num} (Seed = {seed})")
        print("========================================")

        # --- 1. Evaluate Agent on Episode N ---
        obs, _ = env_agent.reset(seed=seed)
        agent_score = 0.0
        ep_over = False
        agent_steps = 0

        while not ep_over:
            action = agent.act(obs=obs, training_step=sys.maxsize)
            obs, reward, terminated, truncated, _ = env_agent.step(action)
            agent_score += float(reward)
            agent_steps += 1
            ep_over = terminated or truncated or agent_steps >= config.n_steps_episode

        print(f"Agent Score: {agent_score:.2f} ({agent_steps} steps)")

        # --- 2. Evaluate Oracle on Episode N ---
        start_time = time.time()
        oracle_score, _ = beam_search(
            config=config,
            seed=seed,
            beam_width=beam_width,
            max_steps=config.n_steps_episode,
            heuristic_weight=heuristic_weight,
        )
        elapsed = time.time() - start_time

        ratio = (agent_score / oracle_score * 100.0) if oracle_score != 0 else 0.0

        print(f"Oracle Score: {oracle_score:.2f} (Calculated in {elapsed:.2f}s)")
        print(f"Performance:  {ratio:.1f}%")

        # Log to TensorBoard (X-axis = episode number: 0, 50, 100...)
        writer.add_scalar("Score/Agent", agent_score, episode_num)
        writer.add_scalar("Score/Oracle", oracle_score, episode_num)
        writer.add_scalar("Performance/Ratio_percentage", ratio, episode_num)

    writer.close()
    print(f"\nEvaluation finished. TensorBoard logs saved to '{log_dir}'")


if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    evaluate(
        config=args.config,
        pretrained_state_dir=args.pretrained_state_dir,
        beam_width=args.beam_width,
        heuristic_weight=args.beam_heuristic_weight,
        start_episode=args.start_episode,
        max_episode=args.max_episode,
        episode_step=args.episode_step,
        log_dir=args.log_dir,
    )
