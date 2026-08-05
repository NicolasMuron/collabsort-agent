import copy
import time
from dataclasses import dataclass

import gymnasium as gym
import tyro
from gym_collabsort.config import Action

from collabsort_agent.config import Config


@dataclass
class EvalArgs:
    """Arguments for evaluation."""

    config: Config
    pretrained_state_dir: str | None = None
    seed: int = 42
    beam_width: int = 100


def beam_search(env, beam_width: int, max_steps: int = 1000) -> tuple[float, list[str]]:
    """
    Perform beam search to find the optimal sequence of actions.
    """
    # A beam element is a tuple: (accumulated_reward, environment_state, action_sequence)
    beam = [(0.0, env, [])]

    for step in range(max_steps):
        print(f"Step {step+1}/{max_steps} - Beam size: {len(beam)}", end="\r")
        candidates = []

        # For each element in the current beam
        for score, current_env, history in beam:
            # Check if this environment is already done
            terminated = (
                current_env.n_removed_objects >= current_env.config.n_objects
                and current_env.board.agent_arm.picked_object is None
                and current_env.board.robot_arm.picked_object is None
            )
            
            if terminated:
                # Terminal state, just propagate it
                candidates.append((score, current_env, history))
                continue

            # Try all possible actions
            for action in Action:
                # Clone the environment to simulate the action without affecting others
                env_clone = copy.deepcopy(current_env)

                # Take action
                _, reward, done, truncated, _ = env_clone.step(action.value)

                new_score = score + reward
                new_history = history + [action.name]

                candidates.append((new_score, env_clone, new_history))

        if not candidates:
            break

        # Sort candidates by descending score
        candidates.sort(key=lambda x: x[0], reverse=True)

        # Keep only the top 'beam_width' candidates
        beam = candidates[:beam_width]

        # If all top candidates are in a terminal state, we can stop early
        all_done = True
        for _, e, _ in beam:
            terminated = (
                e.n_removed_objects >= e.config.n_objects
                and e.board.agent_arm.picked_object is None
                and e.board.robot_arm.picked_object is None
            )
            if not terminated:
                all_done = False
                break

        if all_done:
            print(f"\nAll beam paths reached terminal state early at step {step+1}.")
            break

    print("\nBeam search completed.")
    best_score, best_env, best_history = beam[0]
    return best_score, best_history


def evaluate(config: Config, seed: int, beam_width: int, pretrained_state_dir: str | None) -> None:
    """Evaluate optimal actions using beam search and compare with an agent."""
    
    print("========================================")
    print("Starting Evaluation (Agent vs Oracle)")
    print(f"Seed: {seed}, Beam Width: {beam_width}, Max Steps: {config.n_steps_episode}")
    print(f"Agent Model: {pretrained_state_dir if pretrained_state_dir else 'Random (No model provided)'}")
    print("========================================")

    # --- 1. Evaluate Agent ---
    print("\n--- Evaluating Agent ---")
    env_agent = gym.make("CollabSort-v0", config=config.env)
    
    # We must import create_agent here to avoid circular imports if any, or at the top
    from collabsort_agent.train import create_agent
    
    agent = create_agent(
        config=config, sample_obs=env_agent.observation_space.sample(), rng=env_agent.np_random
    )
    if pretrained_state_dir is not None:
        agent.load_state(dir=pretrained_state_dir)

    obs, _ = env_agent.reset(seed=seed)
    agent_score = 0
    ep_over = False
    agent_steps = 0

    while not ep_over:
        action = agent.act(obs=obs, training_step=agent_steps)
        obs, reward, terminated, truncated, _ = env_agent.step(action)
        agent_score += float(reward)
        agent_steps += 1
        ep_over = terminated or truncated or agent_steps >= config.n_steps_episode

    print(f"Agent finished in {agent_steps} steps with score: {agent_score}")

    # --- 2. Evaluate Oracle (Beam Search) ---
    print("\n--- Evaluating Oracle (Beam Search) ---")

    # Initialize environment
    # Note: For deepcopy to work, the environment must NOT have active PyGame surfaces, 
    # so we rely on the default RenderMode.NONE.
    env = gym.make("CollabSort-v0", config=config.env)

    # Reset environment with specific seed
    env.reset(seed=seed)

    start_time = time.time()

    # Run Beam Search
    best_score, best_history = beam_search(
        env, beam_width=beam_width, max_steps=config.n_steps_episode
    )

    elapsed = time.time() - start_time

    print("\n" + "=" * 50)
    print(f"Evaluation finished in {elapsed:.2f} seconds")
    print(f"Agent Score:  {agent_score:.2f}")
    print(f"Oracle Score: {best_score:.2f}")
    if best_score > 0:
        ratio = (agent_score / best_score) * 100
        print(f"Performance:  {ratio:.1f}% of theoretical maximum")
    print("=" * 50)
    
    # Save optimal actions to a file if needed
    with open("optimal_actions.txt", "w") as f:
        f.write(f"Seed: {seed}\n")
        f.write(f"Optimal Score: {best_score}\n")
        f.write("Actions:\n")
        f.write(",".join(best_history) + "\n")
    print("Optimal action sequence saved to 'optimal_actions.txt'")


if __name__ == "__main__":
    args: EvalArgs = tyro.cli(EvalArgs)
    evaluate(
        config=args.config, 
        seed=args.seed, 
        beam_width=args.beam_width, 
        pretrained_state_dir=args.pretrained_state_dir
    )
