"""
Train an agent.
"""

import time
from dataclasses import dataclass

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import tyro
from gym_collabsort.config import Action
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from collabsort_agent.agent import Agent
from collabsort_agent.config import Config, save_cfg

# Decisions
from collabsort_agent.decision.ard import ARD
from collabsort_agent.decision.greedy import Greedy
from collabsort_agent.decision.decision_rule import WinAllRule
from collabsort_agent.decision.epsilon_greedy import EpsilonGreedy
from collabsort_agent.decision.exploration_decay import (
    ExponentialExplorationDecay,
    LinearExplorationDecay,
)

# Learners
from collabsort_agent.learning.dd_dqn import DoubleDuelingDQN
from collabsort_agent.learning.double_dqn import DoubleDQN
from collabsort_agent.learning.dueling_dqn import DuelingDQN
from collabsort_agent.learning.dqn import DQN
from collabsort_agent.learning.per import PER
from collabsort_agent.learning.noisy_dqn import NoisyDQN
from collabsort_agent.learning.n_step_learning import NStepLearning
from collabsort_agent.learning.q_learning import Qlearning
from collabsort_agent.learning.rainbow_dqn import RainbowDQN

from collabsort_agent.memory import Memory
from collabsort_agent.metacognition import MetaController
from collabsort_agent.perception import Perceiver


@dataclass
class EpisodeMetrics:
    """Episode metrics"""

    # Cumulated reward
    reward: float = 0

    # Number of collisions
    collisions: int = 0

    # Number of collected objects
    collected_objects: int = 0

    # Episode time step (= number of time steps since beginning of episode)
    step: int = 0

    # Number of steps per second
    sps: float = 0

    # Number of direction changes (UP/DOWN oscillations)
    oscillations: int = 0

    def log(
        self,
        logger: SummaryWriter | None,
        episode: int,
    ) -> None:
        """Log metrics"""

        if logger is not None:
            logger.add_scalar(
                tag="training/cumulated_reward",
                scalar_value=self.reward,
                global_step=episode,
            )
            logger.add_scalar(
                tag="training/collisions",
                scalar_value=self.collisions,
                global_step=episode,
            )
            logger.add_scalar(
                tag="training/collected_objects",
                scalar_value=self.collected_objects,
                global_step=episode,
            )
            logger.add_scalar(
                tag="training/steps_per_seconds",
                scalar_value=self.sps,
                global_step=episode,
            )
            logger.add_scalar(
                tag="training/policy_oscillations",
                scalar_value=self.oscillations,
                global_step=episode,
            )


def _build_estimator(
    algo_name: str,
    config: Config,
    n_actions: int,
    state_size: int,
    meta_ctrl: MetaController,
):
    """Factory helper to build the value estimator dynamically."""
    c_learn = config.learning

    if algo_name == "ql":
        return Qlearning(config=c_learn, n_actions=n_actions, meta_ctrl=meta_ctrl)
    elif algo_name == "dqn":
        return DQN(config=c_learn, n_actions=n_actions, state_size=state_size)
    elif algo_name == "dueling_dqn":
        return DuelingDQN(config=c_learn, n_actions=n_actions, state_size=state_size)
    elif algo_name == "ddqn":
        return DoubleDQN(config=c_learn, n_actions=n_actions, state_size=state_size)
    elif algo_name == "dd_dqn":
        return DoubleDuelingDQN(
            config=c_learn, n_actions=n_actions, state_size=state_size
        )
    elif algo_name == "per":
        return PER(config=c_learn, n_actions=n_actions, state_size=state_size)
    elif algo_name == "rainbow":
        return RainbowDQN(
            config=c_learn,
            n_actions=n_actions,
            state_size=state_size,
            n_step=c_learn.n_step,
        )
    elif algo_name == "n_step":
        return NStepLearning(
            config=c_learn,
            n_actions=n_actions,
            state_size=state_size,
            n_step=c_learn.n_step,
        )
    elif algo_name == "noisy":
        return NoisyDQN(config=c_learn, n_actions=n_actions, state_size=state_size)

    raise ValueError(f"Unrecognized learning algorithm: {algo_name}")


def _build_deliberator(
    algo_name: str,
    config: Config,
    estimator,
    rng: np.random.Generator,
    meta_ctrl: MetaController,
):
    """Factory helper to build the deliberator dynamically."""
    if algo_name == "eps":
        if config.decision.exploration_decay == "lin":
            decay = LinearExplorationDecay(
                config=config.decision, total_steps=config.total_steps
            )
        elif config.decision.exploration_decay == "exp":
            decay = ExponentialExplorationDecay(
                config=config.decision, total_steps=config.total_steps
            )
        else:
            raise ValueError(
                f"Unrecognized exploration decay: {config.decision.exploration_decay}"
            )

        return EpsilonGreedy(
            config=config.decision,
            estimator=estimator,
            exploration_decay=decay,
            rng=rng,
        )

    if algo_name == "ard":
        decision_rule = (
            WinAllRule(rng=rng) if config.decision.decision_rule == "win-all" else None
        )
        return ARD(
            config=config.decision,
            estimator=estimator,
            decision_rule=decision_rule,
            meta_ctrl=meta_ctrl,
            rng=rng,
        )

    if algo_name == "gre":
        return Greedy(config=config.decision, estimator=estimator, rng=rng)

    raise ValueError(f"Unrecognized decision algorithm: {algo_name}")


def create_agent(config: Config, sample_obs: dict, rng: np.random.Generator) -> Agent:
    """Create an agent with a specific configuration"""

    # Initialize perception & memory
    perceiver = Perceiver(
        config=config.perception,
        treadmill_rows=config.env.treadmill_rows,
        upper_treadmill_row=config.env.upper_treadmill_row,
        middle_treadmill_row=config.env.middle_treadmill_row,
    )
    sample_sensory_state = perceiver.get_sensory_state(obs=sample_obs)

    if config.memory.type != "none":
        raise ValueError(f"Unrecognized memory type: {config.memory.type}")
    memory = Memory()

    sample_extended_state = memory.get_extended_state(
        sensory_state=sample_sensory_state
    )

    # Initialize metacognition & dimensions
    meta_ctrl = MetaController(
        config=config.meta, learning_cfg=config.learning, decision_cfg=config.decision
    )
    extended_state_size = len(sample_extended_state)
    n_actions = len(Action) + len(memory.get_actions())

    # Dynamic build
    estimator = _build_estimator(
        config.learning.algorithm, config, n_actions, extended_state_size, meta_ctrl
    )
    deliberator = _build_deliberator(
        config.decision.algorithm, config, estimator, rng, meta_ctrl
    )

    return Agent(perceiver=perceiver, memory=memory, deliberator=deliberator)


def train(config: Config) -> None:
    """Train an agent"""

    # Allow PyTorch to use TF32 (tensor float 32) on Ampere+ GPUs.
    torch.set_float32_matmul_precision("high")

    # Create directory path for training output
    train_dir: str = f"runs/train_{int(time.time())}_{config.decision.algorithm}_{config.learning.algorithm}"

    logger = None
    if config.log_events:
        logger = SummaryWriter(f"{train_dir}", flush_secs=60)

    # Initialize environment
    env = gym.make("CollabSort-v0", config=config.env)

    # Create agent
    agent = create_agent(
        config=config, sample_obs=env.observation_space.sample(), rng=env.np_random
    )

    # Training time step (= number of time steps since beginning of training)
    training_step: int = 0

    start_time = time.time()

    FIGURE_LOG_FREQ = 20
    action_counts_history: list[np.ndarray] = []
    q_values_history: list[np.ndarray] = []

    # Global loop
    for episode in trange(config.n_episodes, desc="Training progress"):
        # Reset environment and metrics for new episode
        obs, _ = env.reset()
        ep_metrics = EpisodeMetrics()
        action_history: list[int] = []
        ep_q_values: list[np.ndarray] = []
        ep_over: bool = False

        # Previous action string for oscillation counting
        prev_action_str: str | None = None

        # Episode loop
        while not ep_over:
            try:
                sensory_state = agent.perceiver.get_sensory_state(obs=obs)
                ext_state = agent.memory.get_extended_state(sensory_state=sensory_state)
                q_vals = agent.deliberator.estimator.get_action_values(ext_state)
                ep_q_values.append(q_vals)
            except Exception:
                pass

            # Agent chooses an action
            action: Action = agent.act(
                obs=obs,
                training_step=training_step,
            )

            # Count oscillations (UP/DOWN direction changes)
            try:
                current_action_str = Action(action).name
            except Exception:
                current_action_str = getattr(action, "name", str(action))

            if prev_action_str is not None:
                if (prev_action_str == "UP" and current_action_str == "DOWN") or (
                    prev_action_str == "DOWN" and current_action_str == "UP"
                ):
                    ep_metrics.oscillations += 1

            prev_action_str = current_action_str

            try:
                action_history.append(int(action.value))
            except Exception:
                try:
                    action_history.append(int(getattr(action, "value", 0)))
                except Exception:
                    action_history.append(0)

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
            training_step += 1
            obs = next_obs
            ep_over = (
                terminated or truncated or ep_metrics.step >= config.n_steps_episode
            )

        # Log episode data
        ep_metrics.sps = int(training_step / (time.time() - start_time))
        ep_metrics.log(
            logger=logger,
            episode=episode,
        )
        agent.log_episode(logger=logger, episode=episode)

        # Save Q-values history for heatmap visualization
        if len(ep_q_values) > 0:
            q_values_history.append(np.mean(ep_q_values, axis=0))

        # --- LOGGING OF ACTION DISTRIBUTION AND Q-VALUES HEATMAP ---
        if logger is not None and len(action_history) > 0:
            try:
                n_actions = int(agent.deliberator.estimator.n_actions)
            except Exception:
                n_actions = int(len(Action))

            counts = np.bincount(
                np.array(action_history, dtype=np.int32), minlength=n_actions
            )

            # Record action counts for heatmap
            try:
                action_counts_history.append(counts)
            except Exception:
                action_counts_history = [counts]

            # Labeled bar plot for the episode (easier to read categorical distribution)
            try:
                fig, ax = plt.subplots(figsize=(6, 3))
                labels = []
                for i in range(n_actions):
                    try:
                        labels.append(Action(i).name)
                    except Exception:
                        labels.append(f"action_{i}")

                ax.bar(range(n_actions), counts, color="C0")
                ax.set_xticks(range(n_actions))
                ax.set_xticklabels(labels, rotation=45, ha="right")
                ax.set_ylabel("count")
                ax.set_title("Action distribution this episode")
                fig.tight_layout()
                logger.add_figure(
                    tag="actions/distribution_figure", figure=fig, global_step=episode
                )
                plt.close(fig)
            except Exception:
                # Don't fail training if plotting/logging figure fails
                pass

            # Periodically log the heatmap of actions over recorded episodes
            try:
                if episode % FIGURE_LOG_FREQ == 0 and len(action_counts_history) > 0:
                    mat = np.vstack(action_counts_history)
                    fig, ax = plt.subplots(figsize=(6, 3))
                    im = ax.imshow(mat.T, aspect="auto", cmap="viridis")
                    ax.set_xlabel("episode")
                    ax.set_ylabel("action")
                    ax.set_title("Action distribution heatmap")
                    ax.set_yticks(range(mat.shape[1]))
                    labels = []
                    for i in range(mat.shape[1]):
                        try:
                            labels.append(Action(i).name)
                        except Exception:
                            labels.append(f"action_{i}")
                    ax.set_yticklabels(labels)
                    fig.colorbar(im, ax=ax, label="count")
                    fig.tight_layout()
                    logger.add_figure("actions/heatmap", fig, global_step=episode)
                    plt.close(fig)
            except Exception:
                pass

            # --- HEATMAP OF Q-VALUES PER ACTION ---
            try:
                if episode % FIGURE_LOG_FREQ == 0 and len(q_values_history) > 0:
                    q_mat = np.vstack(q_values_history)
                    fig_q, ax_q = plt.subplots(figsize=(6, 3))
                    im_q = ax_q.imshow(q_mat.T, aspect="auto", cmap="coolwarm")
                    ax_q.set_xlabel("episode")
                    ax_q.set_ylabel("action")
                    ax_q.set_title("Mean Q-Values Heatmap per Action")
                    ax_q.set_yticks(range(q_mat.shape[1]))

                    labels = []
                    for i in range(q_mat.shape[1]):
                        try:
                            labels.append(Action(i).name)
                        except Exception:
                            labels.append(f"action_{i}")
                    ax_q.set_yticklabels(labels)

                    fig_q.colorbar(im_q, ax=ax_q, label="Q-Value")
                    fig_q.tight_layout()
                    logger.add_figure(
                        "charts/q_values_heatmap", fig_q, global_step=episode
                    )
                    plt.close(fig_q)
            except Exception:
                pass

    env.close()

    if config.save_state:
        agent.save_state(dir=train_dir)
        save_cfg(config=config, dir=train_dir)


if __name__ == "__main__":  # pragma: no cover
    # Create training configuration from command line args
    config: Config = tyro.cli(Config)

    train(config=config)
