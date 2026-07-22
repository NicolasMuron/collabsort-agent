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

    Only the sensory state is stacked over time. Target memory features are
    appended ONCE at the end of the stacked representation.
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
        stacked_sensory = super().get_extended_state(sensory_state)

        # We retrieve the Target memory features from the TargetMemory instance
        target_extended = self._target.get_extended_state(sensory_state)
        target_features = target_extended[len(sensory_state) :]

        return np.concatenate([stacked_sensory, target_features])

    def get_actions(self) -> list[MemoryAction]:
        return []
