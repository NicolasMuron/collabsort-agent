import numpy as np

from collabsort_agent.memory.memory import Config
from collabsort_agent.memory.stack_target import StackTargetMemory


def test_stack_target_memory_combines_target_and_stack():
    config = Config(type="stack+target", stack_size=2, target_max_age=2)
    treadmill_rows = [1]
    n_cols_per_row = {1: 1}
    memory = StackTargetMemory(
        config=config, treadmill_rows=treadmill_rows, n_cols_per_row=n_cols_per_row
    )

    sensory_state = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    extended = memory.get_extended_state(sensory_state)

    expected_single_size = sensory_state.shape[0] + 4
    assert extended.shape[0] == expected_single_size * config.stack_size

    assert np.allclose(extended[-4:], np.array([1.0, 2.0, 3.0, 0.0], dtype=np.float32))
