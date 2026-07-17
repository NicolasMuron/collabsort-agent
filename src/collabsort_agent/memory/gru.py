"""
GRU memory implementation.
"""

import numpy as np
import torch
import torch.nn as nn

from collabsort_agent.memory import Config as MemoryConfig
from collabsort_agent.memory import Memory, MemoryAction


class GRUMemory(Memory):
    """Frozen GRU feature encoder used as a non-trainable temporal memory."""

    def __init__(self, config: MemoryConfig, input_size: int) -> None:
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
