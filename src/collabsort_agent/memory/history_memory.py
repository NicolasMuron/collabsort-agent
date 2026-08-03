"""
Stack memory implementation.
"""

from collections import deque

import numpy as np

from collabsort_agent.memory import MemoryConfig
from collabsort_agent.memory.memory import Memory, MemoryAction


class HistoryMemory(Memory):
    """Historical memory storing past sensory states, actions and rewards.

    The extended state is composed of the current sensory state followed by
    the last ``history_size`` transition blocks. Each block contains the
    previous sensory state, the executed agent action and the received reward.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self.history_size = config.history_size
        self._buffer: deque[tuple[np.ndarray, float, float]] = deque(
            maxlen=self.history_size
        )
        self._state_size: int | None = None

    def reset(self) -> None:
        """Clear all stored history at the start of a new episode."""
        self._buffer.clear()
        self._state_size = None

    def get_extended_state(self, sensory_state: np.ndarray) -> np.ndarray:
        """Return the current sensory state extended with recent past transitions."""
        if self._state_size is None:
            self._state_size = len(sensory_state)

        block_size = self._state_size + 2
        zero_block = np.zeros(block_size, dtype=np.float32)

        past_blocks: list[np.ndarray] = []
        for stored_state, action, reward in reversed(self._buffer):
            past_blocks.append(stored_state)
            past_blocks.append(np.array([action, reward], dtype=np.float32))

        padding = [zero_block] * (self.history_size - len(self._buffer))
        if past_blocks:
            past_array = np.concatenate(past_blocks + padding)
        else:
            past_array = (
                np.concatenate(padding) if padding else np.array([], dtype=np.float32)
            )

        return np.concatenate([sensory_state, past_array])

    def store_transition(
        self, sensory_state: np.ndarray, action: int, reward: float
    ) -> None:
        """Store a past transition for future extended-state construction."""
        if self._state_size is None:
            self._state_size = len(sensory_state)

        self._buffer.append((sensory_state.copy(), float(action), float(reward)))

    def get_actions(self) -> list[MemoryAction]:
        return []
