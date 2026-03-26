"""
Definitions for the epsilon-greedy decision-making algorithm.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from collabsort_agent.decision import Config as DecisionConfig
from collabsort_agent.decision import Deliberator
from collabsort_agent.decision.exploration_decay import ExplorationDecay
from collabsort_agent.learning import ActionValueEstimator


class EpsilonGreedy(Deliberator):
    """Epsilon-greedy algorithm for decision-making."""

    def __init__(
        self,
        config: DecisionConfig,
        estimator: ActionValueEstimator,
        exploration_decay: ExplorationDecay,
    ) -> None:
        super().__init__(config=config, estimator=estimator)

        self.exploration_decay = exploration_decay

        # Current exploration probability (set on first choose_action call)
        self.epsilon: float = self.config.epsilon_start

    def choose_action(self, state: np.ndarray, training_step: int | None) -> int:
        if training_step is not None:
            # Update exploration probability
            self.epsilon = self.exploration_decay.get_epsilon(
                training_step=training_step
            )

        # With probability epsilon: explore (choose a random action)
        if np.random.random() < self.epsilon:
            return int(np.random.randint(0, self.estimator.n_actions))

        # With probability (1-epsilon): exploit (greedily choose the best known action)
        action_values = self.estimator.get_action_values(state=state)
        return int(np.argmax(action_values).item())

    def log_episode(self, logger: SummaryWriter, episode: int) -> None:
        logger.add_scalar(
            tag="decision/exploration_probability",
            scalar_value=self.epsilon,
            global_step=episode,
        )

    def save_state(self, dir: str) -> None:
        Path(dir).mkdir(parents=True, exist_ok=True)
        file_path = f"{dir}/{self.state_filename}"
        torch.save(
            {
                "epsilon": self.epsilon,
            },
            file_path,
        )

    def load_state(self, dir: str) -> None:
        file_path = f"{dir}/{self.state_filename}"
        checkpoint = torch.load(file_path)

        self.epsilon = float(checkpoint["epsilon"])
