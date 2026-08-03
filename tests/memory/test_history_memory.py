import numpy as np

from collabsort_agent.memory import MemoryConfig
from collabsort_agent.memory.history_memory import HistoryMemory


def test_history_memory_pads_and_stores_transitions():
    config = MemoryConfig(type="history", history_size=3)
    memory = HistoryMemory(config=config)

    frame = np.array([1.0, 2.0], dtype=np.float32)
    extended = memory.get_extended_state(frame)

    assert extended.shape == (2 + (3 * (2 + 2)),)
    assert np.allclose(extended[:2], frame)
    assert np.allclose(extended[2:], np.zeros(12, dtype=np.float32))

    memory.store_transition(frame, action=1, reward=0.5)
    extended2 = memory.get_extended_state(frame)

    assert extended2.shape == (14,)
    assert np.allclose(extended2[:2], frame)
    assert np.allclose(
        extended2[2:6], np.concatenate([frame, np.array([1.0, 0.5], dtype=np.float32)])
    )
    assert np.allclose(extended2[6:], np.zeros(8, dtype=np.float32))

    memory.store_transition(
        np.array([2.0, 3.0], dtype=np.float32), action=2, reward=-1.0
    )
    extended3 = memory.get_extended_state(frame)

    assert extended3.shape == (14,)
    assert np.allclose(extended3[:2], frame)
    assert np.allclose(
        extended3[2:6],
        np.concatenate(
            [
                np.array([2.0, 3.0], dtype=np.float32),
                np.array([2.0, -1.0], dtype=np.float32),
            ]
        ),
    )
    assert np.allclose(
        extended3[6:10], np.concatenate([frame, np.array([1.0, 0.5], dtype=np.float32)])
    )
    assert np.allclose(extended3[10:], np.zeros(4, dtype=np.float32))
