import numpy as np

from collabsort_agent.memory.memory import Config
from collabsort_agent.memory.stack import StackMemory


def test_stack_memory_pads_and_stacks_frames():
    config = Config(type="stack", stack_size=3)
    memory = StackMemory(config=config)

    frame = np.array([1.0, 2.0], dtype=np.float32)
    extended = memory.get_extended_state(frame)

    assert extended.shape == (6,)
    assert np.allclose(extended[:2], np.zeros(2, dtype=np.float32))
    assert np.allclose(extended[2:4], np.zeros(2, dtype=np.float32))
    assert np.allclose(extended[4:], frame)

    extended2 = memory.get_extended_state(frame)
    assert extended2.shape == (6,)
    assert np.allclose(extended2[:2], np.zeros(2, dtype=np.float32))
    assert np.allclose(extended2[2:4], frame)
    assert np.allclose(extended2[4:], frame)
