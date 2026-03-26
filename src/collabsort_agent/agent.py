"""
Agent definitions.
"""

import numpy as np
from gym_collabsort.config import Action
from torch.utils.tensorboard import SummaryWriter

from collabsort_agent.decision import Deliberator
from collabsort_agent.memory import Memory
from collabsort_agent.perception import Perceiver


class Agent:
    """An agent interacting with its environment."""

    def __init__(
        self, perceiver: Perceiver, memory: Memory, deliberator: Deliberator
    ) -> None:
        self.perceiver = perceiver
        self.memory = memory
        self.deliberator = deliberator

        # Current extended state (sensory + memory)
        self.current_extended_state: np.ndarray | None = None

        # Newest action chosen by the agent
        self.current_action: Action | None = None

    def act(self, obs: dict, training_step: int | None = None) -> Action:
        """Select an action"""

        sensory_state = self.perceiver.get_sensory_state(obs=obs)
        extended_state = self.memory.get_extended_state(sensory_state=sensory_state)

        self.current_extended_state = extended_state

        self.current_action = Action(
            self.deliberator.choose_action(
                state=extended_state, training_step=training_step
            )
        )
        return self.current_action

    def update(self, next_obs: dict, reward: float, done: bool) -> None:
        """Update agent after an action"""

        if self.current_extended_state is None or not self.current_action:
            raise Exception("Trying to update agent with non-existent state")

        # Compute next extended state (sensory + memory) for the transition
        next_sensory_state = self.perceiver.get_sensory_state(obs=next_obs)
        next_extended_state = self.memory.get_extended_state(
            sensory_state=next_sensory_state
        )

        # Update action values
        self.deliberator.estimator.update_action_values(
            state=self.current_extended_state,
            action=self.current_action.value,
            reward=reward,
            next_state=next_extended_state,
            done=done,
        )

    def log_episode(self, logger: SummaryWriter | None, episode: int) -> None:
        """Log agent information after an episode"""

        if logger is not None:
            self.deliberator.log_episode(logger=logger, episode=episode)
            self.deliberator.estimator.log_episode(logger=logger, episode=episode)

    def save_state(self, dir: str) -> None:
        """Save the agent state to disk"""

        self.deliberator.save_state(dir=dir)
        self.deliberator.estimator.save_state(dir=dir)

    def load_state(self, dir: str) -> None:
        """Load the agent state from disk"""

        self.deliberator.estimator.load_state(dir=dir)
        self.deliberator.load_state(dir=dir)
