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
class Config:
    """Memory configuration"""

    # Memory type to use
    type: Literal["none", "target", "stack", "stack+target"] = "stack"

    # Number of past frames to stack (for "stack" and "stack+target")
    stack_size: int = 3

    # Number of steps before a target memory entry is considered stale
    # (for "target" and "stack+target")
    target_max_age: int = 5


class Memory:
    """Base class for memory types (no-op: extended state = sensory state)"""

    def reset(self) -> None:
        """Reset memory state at the beginning of a new episode"""
        pass

    def get_extended_state(self, sensory_state: np.ndarray) -> np.ndarray:
        """Return extended state including sensory and memory states"""

        # No extension: extended state = sensory state
        return sensory_state

    def get_actions(self) -> list[MemoryAction]:
        """Return the list of memory-specific actions"""

        # No memory actions
        return []
