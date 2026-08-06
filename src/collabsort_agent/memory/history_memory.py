"""
Stack memory implementation.
"""

from collections import deque

import numpy as np

from collabsort_agent.memory import MemoryConfig
from collabsort_agent.memory.memory import Memory


class HistoryMemory(Memory):
    """Historical memory storing past sensory states, actions, rewards, and positions.

    The extended state is composed of the current future sensory state followed by
    the last ``history_size`` transition blocks. Each block contains the past state
    for the left-side columns, the agent action, the agent reward, the agent position,
    and the robot position.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self.history_size = config.history_size
        self._buffer: deque[tuple[np.ndarray, float, float, np.ndarray, np.ndarray]] = (
            deque(maxlen=self.history_size)
        )
        self._past_state_size: int | None = None

    def reset(self) -> None:
        """Clear all stored history at the start of a new episode."""
        self._buffer.clear()
        self._past_state_size = None

    def get_extended_state(
        self,
        sensory_state: np.ndarray,
        expected_past_state_size: int | None = None,
    ) -> np.ndarray:
        """Return the current sensory state extended with recent past transitions."""
        if self._past_state_size is None:
            if expected_past_state_size is not None:
                self._past_state_size = expected_past_state_size
            elif self._buffer:
                self._past_state_size = len(self._buffer[0][0])

        if self._past_state_size is None:
            return sensory_state

        block_size = self._past_state_size + 2 + 4 + 2
        zero_block = np.zeros(block_size, dtype=np.float32)

        past_blocks: list[np.ndarray] = []
        for (
            stored_state,
            action,
            reward,
            agent_position,
            robot_position,
        ) in reversed(self._buffer):
            past_blocks.append(stored_state)
            past_blocks.append(
                np.array(
                    [
                        action,
                        reward,
                        *agent_position.tolist(),
                        *robot_position.tolist(),
                        *(agent_position - robot_position).tolist(),
                    ],
                    dtype=np.float32,
                )
            )

        past_array = np.concatenate(
            past_blocks + [zero_block] * (self.history_size - len(self._buffer))
        )
        return np.concatenate([sensory_state, past_array])

    def requires_past_state(self) -> bool:
        """Return whether this memory type needs past-state input."""

        return True

    def store_transition(
        self,
        past_state: np.ndarray,
        action: int,
        reward: float,
        agent_position: tuple[int, int],
        robot_position: tuple[int, int],
    ) -> None:
        """Store a past transition for future extended-state construction."""
        if self._past_state_size is None:
            self._past_state_size = len(past_state)

        agent_position_arr = np.array(agent_position, dtype=np.float32)
        robot_position_arr = np.array(robot_position, dtype=np.float32)

        self._buffer.append(
            (
                past_state.copy(),
                float(action),
                float(reward),
                agent_position_arr,
                robot_position_arr,
            )
        )
