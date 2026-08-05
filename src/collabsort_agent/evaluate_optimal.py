import copy
import copyreg
import time
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import pygame
import tyro
from gym_collabsort.config import Action
from pygame import freetype

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
    beam_width: int = 50


def _make_env(config: Config) -> gym.Env:
    """Create a new environment instance (no rendering)."""
    return gym.make("CollabSort-v0", config=config.env)


def beam_search(
    config: Config, seed: int, beam_width: int, max_steps: int = 1000
) -> tuple[float, list[str]]:
    """
    Perform beam search using copy.deepcopy on CollabSortEnv (unwrapped).

    Gymnasium wrappers (OrderEnforcing, etc.) may override __deepcopy__ to
    return self, breaking the beam search. We bypass them by deepcopying only
    the unwrapped CollabSortEnv and calling step() on it directly.

    Returns (best_accumulated_reward, best_action_sequence).
    """
    # Initialise and reset a gym env, then extract the underlying CollabSortEnv
    wrapper_env = _make_env(config)
    wrapper_env.reset(seed=seed)
    init_unwrapped: Any = wrapper_env.unwrapped

    # A beam element: (accumulated_reward, unwrapped_env_snapshot, action_sequence)
    beam: list[tuple[float, Any, list[str]]] = [
        (0.0, copy.deepcopy(init_unwrapped), [])
    ]

    for step in range(max_steps):
        print(
            f"Step {step + 1}/{max_steps} - Beam size: {len(beam)}",
            end="\r",
            flush=True,
        )
        candidates: list[tuple[float, Any, list[str]]] = []

        for score, env_snapshot, history in beam:
            # Try all possible actions from this snapshot
            for action in Action:
                # Deepcopy only the unwrapped env — safe since copyreg handles pygame types
                env_clone: Any = copy.deepcopy(env_snapshot)
                # Call step directly on the CollabSortEnv (no gym wrapper overhead)
                _, reward, _, _, _ = env_clone.step(action.value)

                new_score = score + float(reward)
                new_history = history + [action.name]
                candidates.append((new_score, env_clone, new_history))

        if not candidates:
            break

        # Keep only the top K candidates
        candidates.sort(key=lambda x: x[0], reverse=True)
        beam = candidates[:beam_width]

    print("\nBeam search completed.")
    best_score, _, best_history = beam[0]
    return best_score, best_history


def evaluate(
    config: Config, seed: int, beam_width: int, pretrained_state_dir: str | None
) -> None:
    """Evaluate optimal actions using beam search and compare with an agent."""

    print("========================================")
    print("Starting Evaluation (Agent vs Oracle)")
    print(
        f"Seed: {seed}, Beam Width: {beam_width}, Max Steps: {config.n_steps_episode}"
    )
    print(
        f"Agent Model: {pretrained_state_dir if pretrained_state_dir else 'Random (No model provided)'}"
    )
    print("========================================")

    # --- 1. Evaluate Agent ---
    print("\n--- Evaluating Agent ---")
    env_agent = _make_env(config)

    from collabsort_agent.train import create_agent

    agent = create_agent(
        config=config,
        sample_obs=env_agent.observation_space.sample(),
        rng=env_agent.np_random,
    )
    if pretrained_state_dir is not None:
        agent.load_state(dir=pretrained_state_dir)

    obs, _ = env_agent.reset(seed=seed)
    agent_score = 0.0
    ep_over = False
    agent_steps = 0

    while not ep_over:
        action = agent.act(obs=obs, training_step=agent_steps)
        obs, reward, terminated, truncated, _ = env_agent.step(action)
        agent_score += float(reward)
        agent_steps += 1
        ep_over = terminated or truncated or agent_steps >= config.n_steps_episode

    print(f"Agent finished in {agent_steps} steps with score: {agent_score:.2f}")

    # --- 2. Evaluate Oracle (Beam Search) ---
    print("\n--- Evaluating Oracle (Beam Search) ---")

    start_time = time.time()
    best_score, best_history = beam_search(
        config=config,
        seed=seed,
        beam_width=beam_width,
        max_steps=config.n_steps_episode,
    )
    elapsed = time.time() - start_time

    print("\n" + "=" * 50)
    print(f"Evaluation finished in {elapsed:.2f} seconds")
    print(f"Agent Score:  {agent_score:.2f}")
    print(f"Oracle Score: {best_score:.2f}")
    if best_score != 0:
        ratio = (agent_score / best_score) * 100
        print(f"Performance:  {ratio:.1f}% of theoretical maximum")
    print("=" * 50)

    # Save optimal actions to a file
    with open("optimal_actions.txt", "w") as f:
        f.write(f"Seed: {seed}\n")
        f.write(f"Oracle Score: {best_score}\n")
        f.write(f"Agent Score: {agent_score}\n")
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
    )
