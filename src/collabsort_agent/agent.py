"""
Agent definitions.
"""

from typing import Any

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

        # Current future sensory state (without memory extension)
        self.current_sensory_state: np.ndarray | None = None
        self.current_past_state: np.ndarray | None = None
        self.current_agent_position: tuple[int, int] | None = None
        self.current_robot_position: tuple[int, int] | None = None

        # Newest action chosen by the agent
        self.current_action: Action | None = None

        # Cached next state to prevent double-updating memory modules
        self.next_extended_state: np.ndarray | None = None
        self.next_sensory_state: np.ndarray | None = None
        self.next_past_state: np.ndarray | None = None
        self.next_agent_position: tuple[int, int] | None = None
        self.next_robot_position: tuple[int, int] | None = None

    def reset(self) -> None:
        """Reset agent state at the beginning of a new episode"""

        self.memory.reset()
        self.current_extended_state = None
        self.current_sensory_state = None
        self.current_past_state = None
        self.current_agent_position = None
        self.current_robot_position = None
        self.current_action = None
        self.next_extended_state = None
        self.next_sensory_state = None
        self.next_past_state = None
        self.next_agent_position = None
        self.next_robot_position = None

    def eval(self) -> None:
        """Switch any trainable components into evaluation mode."""

        estimator = self.deliberator.estimator
        if hasattr(estimator, "q_network"):
            q_network: Any = estimator.q_network
            if q_network is not None and hasattr(q_network, "eval"):
                q_network.eval()

        if hasattr(estimator, "target_network"):
            target_network: Any = estimator.target_network
            if target_network is not None and hasattr(target_network, "eval"):
                target_network.eval()

    def act(
        self,
        obs: dict,
        training_step: int,
    ) -> Action:
        """Select an action"""

        if self.next_extended_state is not None:
            extended_state = self.next_extended_state
            self.current_sensory_state = self.next_sensory_state
            self.current_past_state = self.next_past_state
            self.current_agent_position = self.next_agent_position
            self.current_robot_position = self.next_robot_position
            self.next_extended_state = None
            self.next_sensory_state = None
            self.next_past_state = None
            self.next_agent_position = None
            self.next_robot_position = None
        else:
            future_state = self.perceiver.get_sensory_state(obs=obs)
            self.current_sensory_state = future_state
            self.current_agent_position = tuple(obs["self"]["coords"])
            self.current_robot_position = tuple(obs["robot"])

            if self.memory.requires_past_state():
                past_state = self.perceiver.get_past_state(obs=obs)
                self.current_past_state = past_state
            else:
                self.current_past_state = None

            extended_state = self.memory.get_extended_state(sensory_state=future_state)

        self.current_extended_state = extended_state

        self.current_action = Action(
            self.deliberator.choose_action(
                state=extended_state,
                training_step=training_step,
            )
        )
        return self.current_action

    def update(self, next_obs: dict, reward: float, done: bool) -> None:
        """Update agent after an action"""

        if self.current_extended_state is None or self.current_action is None:
            raise RuntimeError("Trying to update agent with non-existent state")

        # Compute next sensory state first, then update memory with the current transition.
        next_sensory_state = self.perceiver.get_sensory_state(obs=next_obs)

        if (
            hasattr(self.memory, "store_transition")
            and self.current_past_state is not None
            and self.current_agent_position is not None
            and self.current_robot_position is not None
        ):
            self.memory.store_transition(
                past_state=self.current_past_state,
                action=self.current_action.value,
                reward=reward,
                agent_position=self.current_agent_position,
                robot_position=self.current_robot_position,
            )

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

        self.next_extended_state = next_extended_state
        self.next_sensory_state = next_sensory_state
        self.next_agent_position = tuple(next_obs["self"]["coords"])
        self.next_robot_position = tuple(next_obs["robot"])

        if self.memory.requires_past_state():
            self.next_past_state = self.perceiver.get_past_state(obs=next_obs)
        else:
            self.next_past_state = None

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
