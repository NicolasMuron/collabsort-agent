"""
Stack memory implementation.
"""

from collections import deque

import numpy as np

from collabsort_agent.memory import Config as MemoryConfig
from collabsort_agent.memory import Memory, MemoryAction


class StackMemory(Memory):
    """Frame-stacking memory: concatenates the last ``stack_size`` sensory states.

    Benefits:
    - Lets the agent perceive the velocity of objects on treadmills.
    - Provides a short movement history so the agent can avoid oscillating.

    The extended state has size ``stack_size * len(sensory_state)``.
    Frames missing at the start of an episode are filled with zeros.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self.stack_size = config.stack_size
        self._buffer: deque[np.ndarray] = deque(maxlen=self.stack_size)
        self._state_size: int | None = None

    def reset(self) -> None:
        """Clear the frame buffer at the start of a new episode."""
        self._buffer.clear()
        self._state_size = None

    def get_extended_state(self, sensory_state: np.ndarray) -> np.ndarray:
        """Return the concatenation of the last ``stack_size`` sensory states."""
        if self._state_size is None:
            self._state_size = len(sensory_state)

        self._buffer.append(sensory_state.copy())

        # Pad with zero frames if the buffer is not full yet
        n_missing = self.stack_size - len(self._buffer)
        padding = [np.zeros(self._state_size, dtype=np.float32)] * n_missing
        return np.concatenate(padding + list(self._buffer))

    def get_actions(self) -> list[MemoryAction]:
        return []
