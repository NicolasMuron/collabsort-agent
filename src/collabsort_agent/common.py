from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import numpy as np
from gym_collabsort.config import Action
from torch.utils.tensorboard import SummaryWriter

from collabsort_agent.agent import Agent
from collabsort_agent.config import Config
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
from collabsort_agent.memory.history_memory import HistoryMemory
from collabsort_agent.memory.memory import Memory
from collabsort_agent.metacognition import Hyperparameters
from collabsort_agent.metacognition.confidence import (
    BayesianConfidence,
    GapConfidence,
    TDErrorCalibration,
)
from collabsort_agent.metacognition.controller import MetaController
from collabsort_agent.metacognition.monitoring import MetaMonitoring
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

    # Optimal reward computed by the DP
    optimal_reward: float = 0
    # Agent action counts by action name
    agent_action_counts: dict[str, int] = field(default_factory=dict)
    # Optimal matches by action name
    optimal_matches_by_action: dict[str, int] = field(default_factory=dict)
    # Total number of optimal action matches
    optimal_action_matches: int = 0

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
                    "agent": self.reward,
                    "optimal": self.optimal_reward,
                },
                global_step=episode,
            )

            if self.step > 0:
                logger.add_scalar(
                    tag="training/optimal_action_match_ratio",
                    scalar_value=(
                        self.optimal_action_matches / self.step if self.step > 0 else 0
                    ),
                    global_step=episode,
                )
                if self.optimal_action_matches > 0:
                    actions = ["NONE", "UP", "DOWN", "PICK"]
                    executed = [self.agent_action_counts.get(a, 0) for a in actions]
                    matched = [
                        self.optimal_matches_by_action.get(a, 0) for a in actions
                    ]

                    fig, ax = plt.subplots(figsize=(6, 4))

                    # Plot executed first (background)
                    ax.bar(
                        actions,
                        executed,
                        label="Réalisée (Total)",
                        color="lightgray",
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
    elif mem_type == "history":
        memory = HistoryMemory(config=config.memory)
    else:
        raise ValueError(f"Unrecognized memory type: {mem_type}")

    sample_extended_state = memory.get_extended_state(
        sensory_state=sample_sensory_state,
        expected_past_state_size=len(perceiver.get_past_state(obs=sample_obs)),
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
