import numpy as np

from collabsort_agent.memory.memory import Config
from collabsort_agent.memory.target import TargetMemory


def test_target_memory_remembers_and_ages_objects():
    config = Config(type="target", target_max_age=2)
    treadmill_rows = [1, 2]
    n_cols_per_row = {1: 1, 2: 1}
    memory = TargetMemory(
        config=config, treadmill_rows=treadmill_rows, n_cols_per_row=n_cols_per_row
    )

    sensory_state = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float32
    )
    extended = memory.get_extended_state(sensory_state)

    assert extended.shape[0] == sensory_state.shape[0] + len(treadmill_rows) * 4

    assert extended[-8] == 1.0
    assert extended[-7] == 2.0
    assert extended[-6] == 3.0
    assert extended[-5] == 0.0
