"""
Common definitions for agent memory.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np


class MemoryAction(Enum):
    """Possible memory actions"""

    CUE_RETRIEVAL = 0
    ADVANCE_RETRIEVAL = 1
    STORE_IN_WM = 2


@dataclass
class MemoryConfig:
    """Memory configuration"""

    # Memory type to use
    type: Literal["none", "stack"] = "none"

    # Number of past frames to stack
    stack_size: int = 5


class Memory:
    """Base class for memory types"""

    def reset(self) -> None:
        """Reset memory state at the beginning of a new episode"""

    def get_extended_state(self, sensory_state: np.ndarray) -> np.ndarray:
        """Return extended state including sensory and memory states"""

        # No extension: extended state = sensory state
        return sensory_state

    def get_actions(self) -> list[MemoryAction]:
        """Return the number of memory actions"""

        # No memory actions
        return []
