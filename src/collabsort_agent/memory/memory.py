"""
Common definitions for agent memory.
"""

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Literal

import numpy as np
import torch
import torch.nn as nn


class MemoryAction(Enum):
    """Possible memory actions"""

    CUE_RETRIEVAL = 0
    ADVANCE_RETRIEVAL = 1
    STORE_IN_WM = 2


@dataclass
class Config:
    """Memory configuration"""

    # Memory type to use
    type: Literal["none", "target", "stack", "stack+target", "gru"] = "gru"

    # Number of past frames to stack (for "stack" and "stack+target")
    stack_size: int = 10

    # Number of steps before a target memory entry is considered stale
    # (for "target" and "stack+target")
    target_max_age: int = 20

    # Size of the frozen GRU hidden state (for "gru" and "gru+occupancy")
    gru_hidden_size: int = 8


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


class StackMemory(Memory):
    """Frame-stacking memory: concatenates the last ``stack_size`` sensory states.

    Benefits:
    - Lets the agent perceive the velocity of objects on treadmills.
    - Provides a short movement history so the agent can avoid oscillating.

    The extended state has size ``stack_size * len(sensory_state)``.
    Frames missing at the start of an episode are filled with zeros.
    """

    def __init__(self, config: Config) -> None:
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


class GRUMemory(Memory):
    """Frozen GRU feature encoder used as a non-trainable temporal memory."""

    def __init__(self, config: Config, input_size: int) -> None:
        self.input_size = input_size
        self.hidden_size = config.gru_hidden_size
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=self.hidden_size,
            batch_first=True,
        )
        self.gru.eval()
        for param in self.gru.parameters():
            param.requires_grad_(False)
        self._hidden_state = torch.zeros(1, 1, self.hidden_size, dtype=torch.float32)

    def reset(self) -> None:
        self._hidden_state = torch.zeros(1, 1, self.hidden_size, dtype=torch.float32)

    def get_extended_state(self, sensory_state: np.ndarray) -> np.ndarray:
        sensory_state = np.asarray(sensory_state, dtype=np.float32)
        x = torch.from_numpy(sensory_state).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            _, hidden_state = self.gru(x, self._hidden_state)
            self._hidden_state = hidden_state

        hidden_features = self._hidden_state[0, 0].cpu().numpy().astype(np.float32)
        return np.concatenate([sensory_state, hidden_features])

    def get_actions(self) -> list[MemoryAction]:
        return []


class TargetMemory(Memory):
    """Target memory: remembers the last object seen on each active treadmill.

    When an object leaves the agent's perception window, the agent keeps a
    fading memory of that object's color, shape and age (normalised to [0, 1]).
    This is useful for anticipating object positions between perception steps.

    The extended state adds ``4 * len(treadmill_rows)`` features after the
    sensory state: [present, color, shape, age_normalised] per treadmill row.
    Age is 0.0 when the object was just seen and rises to 1.0 over
    ``target_max_age`` steps, at which point the memory entry is cleared.

    The class assumes the sensory-state layout produced by ``Perceiver``:
        [agent_row, agent_col, picked_object,        # 3 features
         robot_row, robot_col,                        # 2 features
         (present, color, shape) × n_cols × n_rows]  # 3 × cols × rows
    """

    # Layout constants (must match perception.py)
    _AGENT_FEATURES: int = 3  # agent_row, agent_col, picked_object
    _ROBOT_FEATURES: int = 2  # robot_row, robot_col
    _OBJ_FEATURES: int = 3  # present, color, shape
    _MEMORY_FEATURES: int = 4  # present, color, shape, age_normalised

    def __init__(
        self,
        config: Config,
        treadmill_rows: list[int],
        n_cols_per_row: dict[int, int],
    ) -> None:
        """
        Parameters
        ----------
        config:
            Memory configuration (uses ``target_max_age``).
        treadmill_rows:
            Ordered list of active treadmill row indices.
        n_cols_per_row:
            Mapping ``{row: n_perceived_columns}`` for each treadmill row.
            Must match the values used by the ``Perceiver``.
        """
        self.treadmill_rows = treadmill_rows
        self.n_cols_per_row = n_cols_per_row
        self.max_age = config.target_max_age
        self._targets: dict[int, np.ndarray] = {}
        self.reset()

    def reset(self) -> None:
        """Clear all target memories at the start of a new episode."""
        self._targets = {
            row: np.zeros(self._MEMORY_FEATURES, dtype=np.float32)
            for row in self.treadmill_rows
        }

    def _parse_objects(self, sensory_state: np.ndarray) -> dict[int, tuple | None]:
        """Extract the first visible object per treadmill from the sensory state.

        Returns a dict mapping each treadmill row to ``(color, shape)`` if an
        object was visible, or ``None`` otherwise.
        """
        offset = self._AGENT_FEATURES + self._ROBOT_FEATURES
        result: dict[int, tuple | None] = {}

        for row in self.treadmill_rows:
            n_cols = self.n_cols_per_row.get(row, 1)
            found = None
            for c in range(n_cols):
                idx = offset + c * self._OBJ_FEATURES
                if idx + 2 < len(sensory_state) and sensory_state[idx] > 0:
                    found = (sensory_state[idx + 1], sensory_state[idx + 2])
                    break
            result[row] = found
            offset += n_cols * self._OBJ_FEATURES

        return result

    def get_extended_state(self, sensory_state: np.ndarray) -> np.ndarray:
        """Return the sensory state extended with per-treadmill target memory."""
        objects = self._parse_objects(sensory_state)
        memory_features: list[np.ndarray] = []

        for row in self.treadmill_rows:
            obj = objects[row]
            if obj is not None:
                # Object currently visible: refresh memory entry
                self._targets[row] = np.array(
                    [1.0, obj[0], obj[1], 0.0], dtype=np.float32
                )
            elif self._targets[row][0] > 0:
                # Object not visible but was seen recently: age the entry
                new_age = self._targets[row][3] + 1.0 / self.max_age
                if new_age >= 1.0:
                    # Memory expired: clear the entry
                    self._targets[row] = np.zeros(
                        self._MEMORY_FEATURES, dtype=np.float32
                    )
                else:
                    self._targets[row][3] = new_age

            memory_features.append(self._targets[row].copy())

        return np.concatenate([sensory_state] + memory_features)

    def get_actions(self) -> list[MemoryAction]:
        return []


class StackTargetMemory(Memory):
    """Combination of :class:`TargetMemory` and :class:`StackMemory`.

    The sensory state is first extended with target-memory features, then the
    last ``stack_size`` such extended states are concatenated.
    """

    def __init__(
        self,
        config: Config,
        treadmill_rows: list[int],
        n_cols_per_row: dict[int, int],
    ) -> None:
        self._target = TargetMemory(
            config=config,
            treadmill_rows=treadmill_rows,
            n_cols_per_row=n_cols_per_row,
        )
        self._stack = StackMemory(config=config)

    def reset(self) -> None:
        self._target.reset()
        self._stack.reset()

    def get_extended_state(self, sensory_state: np.ndarray) -> np.ndarray:
        # 1. Extend with target-memory features
        target_state = self._target.get_extended_state(sensory_state)
        # 2. Stack the extended states across time
        return self._stack.get_extended_state(target_state)

    def get_actions(self) -> list[MemoryAction]:
        return []
