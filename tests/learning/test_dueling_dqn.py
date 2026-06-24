"""
Unit tests for the Dueling DQN algorithm.
"""

import numpy as np
import torch

from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.dueling_dqn import DuelingDQN, Dueling_Network
from tests.learning.test_dqn import TestDQN


class TestDuelingDQN(TestDQN):
    def _make_dqn(
        self,
        n_actions: int = 4,
        state_size: int = 5,
        batch_size: int = 4,
        replay_buffer_size: int = 100,
        target_sync_freq: int = 500,
    ) -> DuelingDQN:
        config = LearningConfig(
            batch_size=batch_size,
            replay_buffer_size=replay_buffer_size,
            target_network_sync_freq=target_sync_freq,
        )
        return DuelingDQN(config=config, n_actions=n_actions, state_size=state_size)

    def test_dueling_network_instances(self) -> None:
        dqn = self._make_dqn()

        assert isinstance(dqn.q_network, Dueling_Network)
        assert isinstance(dqn.target_network, Dueling_Network)

    def test_dueling_network_output_shape(self) -> None:
        network = Dueling_Network(state_size=5, action_size=4)
        inputs = torch.randn((3, 5), dtype=torch.float32)

        outputs = network(inputs)

        assert outputs.shape == (3, 4)

    def test_dueling_network_advantage_normalization(self) -> None:
        network = Dueling_Network(state_size=5, action_size=4)
        inputs = torch.randn((2, 5), dtype=torch.float32)

        features = network.feature_layer(inputs)
        value = network.value_stream(features)
        advantages = network.advantage_stream(features)
        expected_outputs = value + (advantages - advantages.mean(dim=1, keepdim=True))
        outputs = network(inputs)
        normalized_advantages = advantages - advantages.mean(dim=1, keepdim=True)

        assert torch.allclose(outputs, expected_outputs)
        assert torch.allclose(
            normalized_advantages.mean(dim=1, keepdim=True), torch.zeros_like(value)
        )

    def test_dueling_streams_dimensions(self) -> None:
        """Thoroughly check the dimensions of each internal branch."""
        state_size = 5
        action_size = 4
        network = Dueling_Network(state_size=state_size, action_size=action_size)
        inputs = torch.randn((3, state_size), dtype=torch.float32)

        # We extract the common characteristics
        features = network.feature_layer(inputs)
        assert features.shape[0] == 3  # Doit conserver le batch_size

        # The “Value” branch must output a single dimension per statement (Batch_size, 1)
        value = network.value_stream(features)
        assert value.shape == (3, 1)

        # The Advantage branch must output one dimension per action (Batch_size, n_actions)
        advantages = network.advantage_stream(features)
        assert advantages.shape == (3, action_size)

    def test_dueling_identitcal_features_sharing(self) -> None:
        """Ensures that the “Value” branch and the “Advantage” branch share exactly the same input stream."""
        state_size = 5
        network = Dueling_Network(state_size=state_size, action_size=4)
        inputs = torch.randn((1, state_size), dtype=torch.float32)

        # We perform a complete forward pass
        _ = network(inputs)

        # We check that both streams were fed by the same activations
        # If the branches were not connected to the same feature_layer, this test would fail
        assert (
            network.value_stream[0].in_features
            == network.advantage_stream[0].in_features
        )
