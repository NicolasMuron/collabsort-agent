import numpy as np

from collabsort_agent.memory import MemoryConfig
from collabsort_agent.memory.history_memory import HistoryMemory


def test_history_memory_pads_and_stores_transitions():
    config = MemoryConfig(type="history", history_size=3)
    memory = HistoryMemory(config=config)

    frame = np.array([1.0, 2.0], dtype=np.float32)
    extended = memory.get_extended_state(frame, expected_past_state_size=2)

    assert extended.shape == (32,)
    assert np.allclose(extended[:2], frame)
    assert np.allclose(extended[2:], np.zeros(30, dtype=np.float32))

    past_state = np.array([0.1, 0.2], dtype=np.float32)
    memory.store_transition(
        past_state,
        action=1,
        reward=0.5,
        agent_position=(1, 2),
        robot_position=(0, 3),
    )

    extended2 = memory.get_extended_state(frame)
    assert extended2.shape == (32,)
    assert np.allclose(extended2[:2], frame)
    assert np.allclose(
        extended2[2:12],
        np.concatenate(
            [
                past_state,
                np.array([1.0, 0.5, 1.0, 2.0, 0.0, 3.0, 1.0, -1.0], dtype=np.float32),
            ]
        ),
    )
    assert np.allclose(extended2[12:], np.zeros(20, dtype=np.float32))

    memory.store_transition(
        np.array([0.3, 0.4], dtype=np.float32),
        action=2,
        reward=-1.0,
        agent_position=(2, 3),
        robot_position=(1, 4),
    )
    extended3 = memory.get_extended_state(frame)

    assert extended3.shape == (32,)
    assert np.allclose(extended3[:2], frame)
    assert np.allclose(
        extended3[2:12],
        np.concatenate(
            [
                np.array([0.3, 0.4], dtype=np.float32),
                np.array([2.0, -1.0, 2.0, 3.0, 1.0, 4.0, 1.0, -1.0], dtype=np.float32),
            ]
        ),
    )
    assert np.allclose(
        extended3[12:22],
        np.concatenate(
            [
                past_state,
                np.array([1.0, 0.5, 1.0, 2.0, 0.0, 3.0, 1.0, -1.0], dtype=np.float32),
            ]
        ),
    )
