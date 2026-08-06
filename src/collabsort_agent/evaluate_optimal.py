import copy
import copyreg
import re
import time
from dataclasses import dataclass
from pathlib import Path
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
# Since we run in RenderMode.NONE, we only need the geometry (size, flags),
# not the pixel content. We serialize them as blank surfaces / fonts.
# ---------------------------------------------------------------------------
def _surface_reduce(surface: pygame.Surface):
    size = surface.get_size()
    flags = surface.get_flags()
    return _surface_restore, (size, flags)


def _surface_restore(size, flags):
    pygame.display.init()  # required for Surface creation without a display
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
    seed: int = 42
    beam_width: int = 20
    beam_heuristic_weight: float = 1.0
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
    """
    Perform beam search using copy.deepcopy on CollabSortEnv (unwrapped).

    Gymnasium wrappers (OrderEnforcing, etc.) may override __deepcopy__ to
    return self, breaking the beam search. We bypass them by deepcopying only
    the unwrapped CollabSortEnv and calling step() on it directly.

    Returns (best_accumulated_reward, best_action_sequence).
    """
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

                # Safety guard: Discard heuristic bonus if the action caused a failed pick or collision
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


def evaluate_agent_on_seed(
    config: Config,
    seed: int,
    agent: Any,
) -> float:
    """Run a single deterministic evaluation episode for a given agent instance."""
    env_agent = _make_env(config)

    # Disable exploration / set agent to evaluation mode if supported
    if hasattr(agent, "eval"):
        agent.eval()

    obs, _ = env_agent.reset(seed=seed)
    agent_score = 0.0
    ep_over = False
    agent_steps = 0

    while not ep_over:
        # Pass float('inf') to disable exploration (greedy deterministic policy)
        action = agent.act(obs=obs, training_step=float("inf"))
        obs, reward, terminated, truncated, _ = env_agent.step(action)
        agent_score += float(reward)
        agent_steps += 1
        ep_over = terminated or truncated or agent_steps >= config.n_steps_episode

    return agent_score


def evaluate(
    config: Config,
    seed: int,
    beam_width: int,
    pretrained_state_dir: str | None,
    heuristic_weight: float = 0.0,
    log_dir: str = "runs/oracle_vs_agent",
) -> None:
    """Evaluate optimal actions using beam search and compare with an agent across training checkpoints."""

    print("========================================")
    print("Starting Evaluation (Agent vs Oracle)")
    print(
        f"Seed: {seed}, Beam Width: {beam_width}, Max Steps: {config.n_steps_episode}"
    )
    print(
        f"Agent Model Dir: {pretrained_state_dir if pretrained_state_dir else 'Random (No model provided)'}"
    )
    print("========================================")

    writer = SummaryWriter(log_dir=log_dir)

    # --- 1. Compute Oracle Score (Cached/Computed once for this seed) ---
    print("\n--- Evaluating Oracle (Beam Search) ---")
    start_time = time.time()
    oracle_score, best_history = beam_search(
        config=config,
        seed=seed,
        beam_width=beam_width,
        max_steps=config.n_steps_episode,
        heuristic_weight=heuristic_weight,
    )
    elapsed = time.time() - start_time
    print(f"Oracle completed in {elapsed:.2f}s with score: {oracle_score:.2f}")

    # --- 2. Find Checkpoints for Full Training Run ---
    from collabsort_agent.train import create_agent

    dummy_env = _make_env(config)
    agent = create_agent(
        config=config,
        sample_obs=dummy_env.observation_space.sample(),
        rng=dummy_env.np_random,
    )

    checkpoints = []
    if pretrained_state_dir is not None:
        state_path = Path(pretrained_state_dir)
        if state_path.is_dir():
            # Look for step subdirectories or checkpoint files in training directory
            subdirs = [d for d in state_path.iterdir() if d.is_dir()]
            for d in subdirs:
                match = re.search(r"(\d+)", d.name)
                step_num = int(match.group(1)) if match else 0
                checkpoints.append((step_num, str(d)))
            checkpoints.sort(key=lambda x: x[0])

    if not checkpoints:
        # Fallback to evaluating the single path provided or a raw untrained agent
        checkpoints = [(0, pretrained_state_dir)]

    print(f"\n--- Evaluating Agent across {len(checkpoints)} checkpoint(s) ---")

    # --- 3. Evaluate Agent Checkpoints and Log to TensorBoard ---
    for step_num, ckpt_dir in checkpoints:
        if ckpt_dir is not None:
            agent.load_state(dir=ckpt_dir)

        agent_score = evaluate_agent_on_seed(config, seed, agent)
        ratio = (agent_score / oracle_score * 100.0) if oracle_score != 0 else 0.0

        print(
            f"Step {step_num:8d} | Agent Score: {agent_score:7.2f} | Oracle Score: {oracle_score:7.2f} | Performance: {ratio:5.1f}%"
        )

        # Log metrics to TensorBoard
        writer.add_scalar("Score/Agent", agent_score, step_num)
        writer.add_scalar("Score/Oracle", oracle_score, step_num)
        writer.add_scalar("Performance/Ratio_percentage", ratio, step_num)

    writer.close()
    print(f"\nTensorBoard logs written to '{log_dir}'")

    # Save optimal actions to a file
    with open("optimal_actions.txt", "w") as f:
        f.write(f"Seed: {seed}\n")
        f.write(f"Oracle Score: {oracle_score}\n")
        f.write("Actions:\n")
        f.write(",".join(best_history) + "\n")
    print("Optimal action sequence saved to 'optimal_actions.txt'")


if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    evaluate(
        config=args.config,
        seed=args.seed,
        beam_width=args.beam_width,
        pretrained_state_dir=args.pretrained_state_dir,
        heuristic_weight=args.beam_heuristic_weight,
        log_dir=args.log_dir,
    )
