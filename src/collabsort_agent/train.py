"""
Train an agent.
"""

import time
from dataclasses import dataclass, field

import gymnasium as gym
import numpy as np
import torch
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
from collabsort_agent.learning.dqn import DQN
from collabsort_agent.learning.dueling_dqn import DuelingDQN
from collabsort_agent.learning.n_step_learning import NStepLearning
from collabsort_agent.learning.per import PER
from collabsort_agent.learning.q_learning import Qlearning
from collabsort_agent.memory.memory import Memory
from collabsort_agent.memory.stack import StackMemory
from collabsort_agent.metacognition import Hyperparameters
from collabsort_agent.metacognition.confidence import (
    BayesianConfidence,
    GapConfidence,
    TDErrorCalibration,
)
from collabsort_agent.metacognition.controller import MetaController
from collabsort_agent.metacognition.monitoring import MetaMonitoring
from collabsort_agent.metrics_tracker import HeatmapTracker
from collabsort_agent.perception import Perceiver


@dataclass
class EpisodeMetrics:
    """Episode metrics"""

    # Cumulated reward
    reward: float = 0

    # Optimized cumulated reward
    optimized_reward: float = 0

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

    # Number of movement actions
    movement_actions: int = 0

    # Picked object values for this episode
    picked_values: list[float] = field(default_factory=list)

    # Added metrics
    robot_collected_objects: int = 0
    missed_objects: int = 0

    optimal_action_matches: int = 0
    optimal_matches_by_action: dict = field(
        default_factory=lambda: {"NONE": 0, "UP": 0, "DOWN": 0, "PICK": 0}
    )
    optimal_action_counts: dict = field(
        default_factory=lambda: {"NONE": 0, "UP": 0, "DOWN": 0, "PICK": 0}
    )
    agent_action_counts: dict = field(
        default_factory=lambda: {"NONE": 0, "UP": 0, "DOWN": 0, "PICK": 0}
    )

    def log(
        self,
        logger: SummaryWriter | None,
        episode: int,
    ) -> None:
        """Log metrics"""

        if logger is not None:
            logger.add_scalars(
                main_tag="training/rewards",
                tag_scalar_dict={
                    "real": self.reward,
                    "optimized": self.optimized_reward,
                },
                global_step=episode,
            )

            if self.step > 0:
                logger.add_scalar(
                    tag="training/optimal_action_match_ratio",
                    scalar_value=self.optimal_action_matches / self.step,
                    global_step=episode,
                )
                if self.optimal_action_matches > 0:
                    import matplotlib.pyplot as plt

                    actions = ["NONE", "UP", "DOWN", "PICK"]
                    executed = [self.agent_action_counts.get(a, 0) for a in actions]
                    matched = [
                        self.optimal_matches_by_action.get(a, 0) for a in actions
                    ]

                    fig, ax = plt.subplots(figsize=(6, 4))

                    # Plot executed first (background)
                    ax.bar(
                        actions, executed, label="Réalisée (Total)", color="lightgray"
                    )
                    # Plot matched on top (foreground)
                    ax.bar(actions, matched, label="Optimale (Succès)", color="green")

                    ax.set_ylabel("Nombre")
                    ax.set_title("Actions réalisées vs optimales")
                    ax.legend()

                    logger.add_figure(
                        "training/actions_bar_chart", fig, global_step=episode
                    )
                    plt.close(fig)

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

            if self.step > 0:
                logger.add_scalar(
                    tag="training/movement_ratio",
                    scalar_value=self.movement_actions / self.step,
                    global_step=episode,
                )
            if len(self.picked_values) > 0:
                logger.add_scalar(
                    tag="training/avg_reward_per_object",
                    scalar_value=sum(self.picked_values) / len(self.picked_values),
                    global_step=episode,
                )

            logger.add_scalar(
                tag="training/robot_collected_objects",
                scalar_value=self.robot_collected_objects,
                global_step=episode,
            )
            logger.add_scalar(
                tag="training/missed_objects",
                scalar_value=self.missed_objects,
                global_step=episode,
            )


def _build_estimator(
    algo_name: str,
    config: Config,
    n_actions: int,
    state_size: int,
    hyperparameters: Hyperparameters,
):
    """Factory helper to build the value estimator dynamically."""
    c_learn = config.learning

    if algo_name == "ql":
        return Qlearning(
            config=c_learn, n_actions=n_actions, hyperparameters=hyperparameters
        )
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
    hyperparameters: Hyperparameters,
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

        if config.meta.confidence_method == "gap":
            confidence_method = GapConfidence(
                decision_cfg=config.decision, hyperparameters=hyperparameters
            )
        elif config.meta.confidence_method == "bayesian":
            confidence_method = BayesianConfidence(
                decision_cfg=config.decision, hyperparameters=hyperparameters
            )
        else:
            raise ValueError(
                f"Unrecognized confidence method: {config.meta.confidence_method}"
            )
        if config.meta.confidence_calibration_method == "none":
            calibration_method = None
        elif config.meta.confidence_calibration_method == "td_error":
            calibration_method = TDErrorCalibration(config=config.meta)
        else:
            raise ValueError(
                "Unrecognized confidence calibration method: "
                f"{config.meta.confidence_calibration_method}"
            )

        meta_monitoring = MetaMonitoring(
            config=config.meta,
            confidence_method=confidence_method,
            calibration_method=calibration_method,
        )

        return ARD(
            config=config.decision,
            estimator=estimator,
            decision_rule=decision_rule,
            hyperparameters=hyperparameters,
            meta_monitoring=meta_monitoring,
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

    # Build memory module
    mem_type = config.memory.type

    if mem_type == "none":
        memory: Memory = Memory()
    elif mem_type == "stack":
        memory = StackMemory(config=config.memory)
    else:
        raise ValueError(f"Unrecognized memory type: {mem_type}")

    sample_extended_state = memory.get_extended_state(
        sensory_state=sample_sensory_state
    )

    # Initialize metacognition & dimensions
    hyperparameters = Hyperparameters(
        decision_cfg=config.decision, learning_cfg=config.learning
    )
    meta_ctrl = MetaController(
        config=config.meta,
        learning_cfg=config.learning,
        decision_cfg=config.decision,
        hyperparameters=hyperparameters,
    )
    extended_state_size = len(sample_extended_state)
    n_actions = len(Action) + len(memory.get_actions())

    # Dynamic build
    estimator = _build_estimator(
        config.learning.algorithm,
        config,
        n_actions,
        extended_state_size,
        hyperparameters,
    )
    deliberator = _build_deliberator(
        config.decision.algorithm, config, estimator, rng, hyperparameters, meta_ctrl
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

    heatmap_tracker = HeatmapTracker(log_freq=20)

    # Global loop
    for episode in trange(config.n_episodes, desc="Training progress"):
        # Reset environment and memory for new episode
        obs, _ = env.reset()
        ep_metrics = EpisodeMetrics()
        action_history: list[int] = []
        ep_over: bool = False
        episode_visitation = np.zeros(config.env.n_rows + 1, dtype=np.int32)
        episode_collisions = np.zeros(config.env.n_rows + 1, dtype=np.int32)
        episode_spatial_actions: dict[int, dict[int, int]] = {}

        # Previous action string for oscillation counting
        prev_action_str: str | None = None

        # Episode loop
        while not ep_over:
            # Extract agent row for spatial tracking
            agent_row = int(obs["self"]["coords"][0])

            # Record visitation
            if 1 <= agent_row <= config.env.n_rows:
                episode_visitation[agent_row] += 1

            # Agent chooses an action
            action: Action = agent.act(
                obs=obs,
                training_step=training_step,
            )

            # Compute best possible immediate reward from this state
            if (
                config.env.enable_reward_change
                and training_step >= config.env.reward_change_step
            ):
                active_agent_rewards = config.env.agent_rewards_after
            else:
                active_agent_rewards = config.env.agent_rewards

            agent_row = int(obs["self"]["coords"][0])
            robot_row = int(obs["robot"][0])
            dist = agent_row - robot_row

            # Action NONE
            reward_none = float(config.env.step_reward)
            if dist == 1:
                reward_none += float(config.env.collision_penalty)

            # Action DOWN
            reward_down = float(config.env.step_reward + config.env.movement_penalty)

            # Action UP
            reward_up = float(config.env.step_reward + config.env.movement_penalty)
            if dist <= 2:
                reward_up += float(config.env.collision_penalty)

            possible_rewards = {
                Action.NONE: reward_none,
                Action.DOWN: reward_down,
                Action.UP: reward_up,
                # PICK in the void is strictly worse than NONE for stats purposes (avoids polluting the PICK match counter),
                # but stays numerically close to NONE so it doesn't distort optimized_reward.
                Action.PICK: reward_none - 1e-6,
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
                    pick_val = float(active_agent_rewards[obj["color"], obj["shape"]])
                    reward_pick = float(config.env.step_reward + pick_val)
                    if dist == 1:
                        reward_pick += float(config.env.collision_penalty)
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
            ep_metrics.robot_collected_objects += info.get("robot_placed_objects", 0)
            ep_metrics.missed_objects += info.get("n_fallen_objects", 0)

            if info.get("agent_picked_value") is not None:
                ep_metrics.picked_values.append(info["agent_picked_value"])

            if info.get("agent_collision") and 1 <= agent_row <= config.env.n_rows:
                episode_collisions[agent_row] += 1

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
            logger=logger, episode=episode, n_actions=n_actions
        )

    env.close()

    if config.save_state:
        agent.save_state(dir=train_dir)
        save_cfg(config=config, dir=train_dir)


if __name__ == "__main__":  # pragma: no cover
    # Create training configuration from command line args
    config: Config = tyro.cli(Config)

    train(config=config)
