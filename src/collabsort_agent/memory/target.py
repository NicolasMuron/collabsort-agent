"""
Target memory implementation.
"""

import numpy as np

from collabsort_agent.memory import Config as MemoryConfig
from collabsort_agent.memory import Memory, MemoryAction


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
        config: MemoryConfig,
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
