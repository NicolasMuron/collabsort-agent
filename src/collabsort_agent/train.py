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
from collabsort_agent.decision.ard import ARD
from collabsort_agent.decision.decision_rule import WinAllRule
from collabsort_agent.decision.epsilon_greedy import EpsilonGreedy
from collabsort_agent.decision.exploration_decay import (
    ExponentialExplorationDecay,
    LinearExplorationDecay,
)
from collabsort_agent.learning.dd_dqn import DoubleDuelingDQN
from collabsort_agent.learning.double_dqn import DoubleDQN
from collabsort_agent.learning.dueling_dqn import DuelingDQN
from collabsort_agent.learning.dqn import DQN
from collabsort_agent.learning.per import PER
from collabsort_agent.learning.q_learning import Qlearning
from collabsort_agent.memory import Memory
from collabsort_agent.metacognition import MetaController
from collabsort_agent.perception import Perceiver


def create_agent(config: Config, sample_obs: dict, rng: np.random.Generator) -> Agent:
    """Create an agent with a specific configuration"""

    # Initialize perception
    perceiver = Perceiver(
        config=config.perception,
        treadmill_rows=[config.env.upper_treadmill_row, config.env.lower_treadmill_row],
    )
    sample_sensory_state = perceiver.get_sensory_state(obs=sample_obs)

    if config.memory.type == "none":
        memory = Memory()
    else:
        raise Exception(f"Unrecognized memory type: {config.memory.type}")

    sample_extended_state = memory.get_extended_state(
        sensory_state=sample_sensory_state
    )

    # Initialize metacognition
    meta_ctrl = MetaController(
        config=config.meta, learning_cfg=config.learning, decision_cfg=config.decision
    )

    # Compute decision hyperparameters
    extended_state_size = len(sample_extended_state)
    n_actions = len(Action) + len(memory.get_actions())

    # Initialize learning
    if config.learning.algorithm == "ql":
        estimator = Qlearning(
            config=config.learning, n_actions=n_actions, meta_ctrl=meta_ctrl
        )
    elif config.learning.algorithm == "dqn":
        estimator = DQN(
            config=config.learning,
            n_actions=n_actions,
            state_size=extended_state_size,
        )
    elif config.learning.algorithm == "dueling_dqn":    
        estimator = DuelingDQN(
            config=config.learning,
            n_actions=n_actions,
            state_size=extended_state_size,
        )    
    elif config.learning.algorithm == "ddqn":
        estimator = DoubleDQN(
            config=config.learning,
            n_actions=n_actions,
            state_size=extended_state_size,
        )
    elif config.learning.algorithm == "dd_dqn":
        estimator = DoubleDuelingDQN(
            config=config.learning,
            n_actions=n_actions,
            state_size=extended_state_size,
        )
    elif config.learning.algorithm == "per":
        estimator = PER(
            config=config.learning,
            n_actions=n_actions,
            state_size=extended_state_size,
        )             
    else:
        raise Exception(f"Unrecognized learning algorithm: {config.learning.algorithm}")

    # Initialize decision-making
    if config.decision.algorithm == "eps":
        # Initialize exploration probability decay algorithm
        if config.decision.exploration_decay == "lin":
            exploration_decay = LinearExplorationDecay(
                config=config.decision, total_steps=config.total_steps
            )
        elif config.decision.exploration_decay == "exp":
            exploration_decay = ExponentialExplorationDecay(
                config=config.decision, total_steps=config.total_steps
            )
        else:
            raise Exception(
                f"Unrecognized exploration decay: {config.decision.exploration_decay}"
            )

        deliberator = EpsilonGreedy(
            config=config.decision,
            estimator=estimator,
            exploration_decay=exploration_decay,
            rng=rng,
        )
    elif config.decision.algorithm == "ard":
        if config.decision.decision_rule == "win-all":
            decision_rule = WinAllRule(rng=rng)

        deliberator = ARD(
            config=config.decision,
            estimator=estimator,
            decision_rule=decision_rule,
            meta_ctrl=meta_ctrl,
            rng=rng,
        )
    else:
        raise Exception(f"Unrecognized decision algorithm: {config.decision.algorithm}")

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
