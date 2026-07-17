import numpy as np
import torch

from collabsort_agent.memory.memory import Config
from collabsort_agent.memory.gru import GRUMemory


def test_gru_memory_extends_state_with_hidden_features():
    torch.manual_seed(0)
    config = Config(type="gru", gru_hidden_size=4)
    memory = GRUMemory(config=config, input_size=3)

    sensory_state = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    extended_state = memory.get_extended_state(sensory_state)

    assert extended_state.shape == (7,)
    assert np.allclose(extended_state[:3], sensory_state)
    assert extended_state[3:].shape == (4,)

    memory.reset()
    reset_state = memory.get_extended_state(sensory_state)
    assert np.allclose(reset_state[:3], sensory_state)
    assert reset_state[3:].shape == (4,)
