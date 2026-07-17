"""
Stack+Target memory implementation.
"""

import numpy as np

from collabsort_agent.memory.memory import Config as MemoryConfig
from collabsort_agent.memory.memory import MemoryAction
from collabsort_agent.memory.stack import StackMemory
from collabsort_agent.memory.target import TargetMemory


class StackTargetMemory(StackMemory):
    """Combination of :class:`TargetMemory` and :class:`StackMemory`.

    The sensory state is first extended with target-memory features, then the
    last ``stack_size`` such extended states are concatenated.
    """

    def __init__(
        self,
        config: MemoryConfig,
        treadmill_rows: list[int],
        n_cols_per_row: dict[int, int],
    ) -> None:
        super().__init__(config=config)
        self._target = TargetMemory(
            config=config,
            treadmill_rows=treadmill_rows,
            n_cols_per_row=n_cols_per_row,
        )

    def reset(self) -> None:
        self._target.reset()
        super().reset()

    def get_extended_state(self, sensory_state: np.ndarray) -> np.ndarray:
        target_state = self._target.get_extended_state(sensory_state)
        return super().get_extended_state(target_state)

    def get_actions(self) -> list[MemoryAction]:
        return []
