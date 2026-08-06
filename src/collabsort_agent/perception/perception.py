"""
Perception-related definitions.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class PerceptionConfig:
    """Perception configuration."""

    # Number of future columns visible to the right of the agent
    n_future_cols: int = 4
    # Number of past columns visible to the left of the agent
    n_past_cols: int = 4
    # Enable 45-degree cone vision (more columns on upper rows)
    cone_perception: bool = False


class Perceiver:
    """Class implementing the agent perception sense"""

    def __init__(
        self,
        config: PerceptionConfig,
        treadmill_rows: list[int],
        upper_treadmill_row: int | None = None,
        middle_treadmill_row: int | None = None,
    ) -> None:
        self.config = config
        self.treadmill_rows = treadmill_rows
        self.upper_treadmill_row = upper_treadmill_row
        self.middle_treadmill_row = middle_treadmill_row

    def _get_visible_column_count(self, base_cols: int, row: int) -> int:
        """Return how many columns are visible for a treadmill row on one side."""

        if not self.config.cone_perception:
            return base_cols

        if self.upper_treadmill_row is not None and row == self.upper_treadmill_row:
            return base_cols + 2

        if self.middle_treadmill_row is not None and row == self.middle_treadmill_row:
            return base_cols + 1

        return base_cols

    def _get_column_indices(self, agent_col: int, count: int, side: str) -> list[int]:
        """Return the absolute column indices for future or past columns."""

        if count <= 0:
            return []

        if side == "future":
            return [agent_col + col for col in range(count)]

        return [agent_col - count + col for col in range(count)]

    def _append_columns(
        self, state_features: list[float], columns: list[int], row: int, obj_map: dict
    ) -> None:
        """Append object features for the given column indices."""

        for col in columns:
            obj_found = obj_map.get((row, col))
            if obj_found:
                state_features.extend(
                    [
                        1.0,
                        obj_found["color"],
                        obj_found["shape"],
                    ]
                )
            else:
                state_features.extend([0.0, 0.0, 0.0])

    def get_future_state(self, obs: dict) -> np.ndarray:
        """Return the sensory state for future columns only."""

        state_features: list[float] = []

        # Agent features
        agent: dict = obs["self"]
        agent_row: int = agent["coords"][0]
        agent_col: int = agent["coords"][1]
        picked_object: int = agent["picked_object"]
        state_features.extend([agent_row, agent_col, picked_object])

        # Robot features
        robot: dict = obs["robot"]
        robot_row: int = robot[0]
        robot_col: int = robot[1]
        state_features.extend([robot_row, robot_col])

        # Build a dict keyed by (row, col) for O(1) object lookup
        objects: tuple[dict] = obs["moving_objects"]
        obj_map: dict = {(obj["coords"][0], obj["coords"][1]): obj for obj in objects}

        for row in self.treadmill_rows:
            future_cols = self._get_visible_column_count(
                base_cols=self.config.n_future_cols, row=row
            )
            future_columns = self._get_column_indices(
                agent_col=agent_col, count=future_cols, side="future"
            )
            self._append_columns(
                state_features=state_features,
                columns=future_columns,
                row=row,
                obj_map=obj_map,
            )

        return np.array(state_features, dtype=np.float32)

    def get_past_state(self, obs: dict) -> np.ndarray:
        """Return the sensory state for past columns only."""

        state_features: list[float] = []

        agent_col: int = obs["self"]["coords"][1]

        objects: tuple[dict] = obs["moving_objects"]
        obj_map: dict = {(obj["coords"][0], obj["coords"][1]): obj for obj in objects}

        for row in self.treadmill_rows:
            past_cols = self._get_visible_column_count(
                base_cols=self.config.n_past_cols, row=row
            )
            past_columns = self._get_column_indices(
                agent_col=agent_col, count=past_cols, side="past"
            )
            self._append_columns(
                state_features=state_features,
                columns=past_columns,
                row=row,
                obj_map=obj_map,
            )

        return np.array(state_features, dtype=np.float32)

    def get_sensory_state(self, obs: dict) -> np.ndarray:
        """Flatten an observation into a vector: the sensory state"""

        return self.get_future_state(obs=obs)
