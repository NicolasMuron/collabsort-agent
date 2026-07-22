"""
Train an agent.
"""

import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import tyro
from gym_collabsort.config import Action
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

from collabsort_agent.agent import Agent
from collabsort_agent.config import Config, save_cfg

# Decisions
from collabsort_agent.decision.ard import ARD
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
from collabsort_agent.learning.n_step_learning import NStepLearning
from collabsort_agent.learning.q_learning import Qlearning

from collabsort_agent.memory import Memory
from collabsort_agent.metacognition import MetaController
from collabsort_agent.perception import Perceiver


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
    elif algo_name == "n_step":
        return NStepLearning(
            config=c_learn,
            n_actions=n_actions,
            state_size=state_size,
            n_step=c_learn.n_step,
        )

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


def train(config: Config) -> None:
    """Train an agent"""

    # Create directory path for training output
    train_dir: str = f"runs/train_{int(time.time())}_{config.decision.algorithm}_{config.learning.algorithm}"

    logger = None
    if config.log_events:
        # Initialize logging
        logger = SummaryWriter(f"{train_dir}")

    # Initialize environment
    env = gym.make("CollabSort-v0", config=config.env)

    # Create agent
    agent = create_agent(
        config=config, sample_obs=env.observation_space.sample(), rng=env.np_random
    )

    # Training time step (= number of time steps since beginning of training)
    training_step: int = 0

    start_time = time.time()

    # Global loop
    for episode in trange(config.n_episodes, desc="Training progress"):
        # Reset environment and metrics for new episode
        obs, _ = env.reset()
        ep_metrics = EpisodeMetrics()
        ep_over: bool = False

        # Episode loop
        while not ep_over:
            # Agent chooses an action
            action: Action = agent.act(
                obs=obs,
                training_step=training_step,
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

    env.close()

    if config.save_state:
        agent.save_state(dir=train_dir)
        save_cfg(config=config, dir=train_dir)


if __name__ == "__main__":  # pragma: no cover
    # Create training configuration from command line args
    config: Config = tyro.cli(Config)

    train(config=config)
