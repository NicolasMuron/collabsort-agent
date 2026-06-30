"""
Perception-related definitions.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Config:
    """Perception configuration"""

    # Number of perceived columns in an observation
    n_perceived_cols: int = 3


class Perceiver:
    """Class implementing the agent perception sense"""

    def __init__(self, config: Config, treadmill_rows: list[int]) -> None:
        self.config = config
        self.treadmill_rows = treadmill_rows

    def get_sensory_state(self, obs: dict) -> np.ndarray:
        """Flatten an observation into a vector: the sensory state"""

        state_features = []

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

        perceived_cols = [
            agent_col + col for col in range(self.config.n_perceived_cols)
        ]
        for row in self.treadmill_rows:
            for col in perceived_cols:
                obj_found = obj_map.get((row, col))
                if obj_found:
                    state_features.extend(
                        [
                            1.0,  # Object present
                            obj_found["color"],
                            obj_found["shape"],
                        ]
                    )
                else:
                    state_features.extend([0.0, 0.0, 0.0])

        # Return a 1D array containing all features
        return np.array(state_features, dtype=np.float32)
