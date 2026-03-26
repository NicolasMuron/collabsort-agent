"""
Unit tests for the DQN algorithm.
"""

import numpy as np
import torch

from collabsort_agent.decision import Config as DecisionConfig
from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.dqn import DQN
from collabsort_agent.learning.q_learning import Qlearning
from collabsort_agent.metacognition import Config as MetaConfig
from collabsort_agent.metacognition import MetaController


def test_dqn() -> None:
    """Test DQN algorithm"""

    # Learning hyperparameters
    config = LearningConfig()
    initial_state = np.array([0, 2, -1, 7, 0.5], dtype=np.float32)
    n_actions = 4

    dqn = DQN(
        config=config,
        n_actions=n_actions,
        state_size=len(initial_state),
    )

    q_values = dqn.get_action_values(state=initial_state)
    assert q_values.shape == (n_actions,)

    dqn.update_action_values(
        state=initial_state, action=0, reward=1, next_state=initial_state * 1.1
    )
    assert len(dqn.replay_buffer) == 1


def test_dqn_target_network_syncs() -> None:
    """Target network should be synced to online network after sync_freq learn() calls"""

    config = LearningConfig(
        target_network_sync_freq=5, batch_size=4, replay_buffer_size=100
    )
    state = np.zeros(4, dtype=np.float32)
    n_actions = 2

    dqn = DQN(
        config=config,
        n_actions=n_actions,
        state_size=4,
    )

    # Fill replay buffer above batch_size threshold
    for i in range(config.batch_size):
        dqn._store_transition(state=state, action=0, reward=float(i), next_state=state)

    # Run enough learn steps to trigger at least one sync
    for _ in range(config.target_network_sync_freq):
        dqn._learn()

    # After sync, target and online networks should have identical weights
    for p_online, p_target in zip(
        dqn.q_network.parameters(), dqn.target_network.parameters(), strict=True
    ):
        assert (p_online.data == p_target.data).all(), "Target network not synced"


def test_dqn_save_and_load(tmp_path) -> None:
    """Saving and loading should restore DQN internal state"""

    config = LearningConfig(lr=1e-3)
    state = np.zeros(4, dtype=np.float32)

    dqn = DQN(
        config=config,
        n_actions=2,
        state_size=4,
    )

    with torch.no_grad():
        for i, param in enumerate(dqn.q_network.parameters()):
            param.fill_(0.1 * (i + 1))
        for i, param in enumerate(dqn.target_network.parameters()):
            param.fill_(0.2 * (i + 1))

    saved_lr = 5e-4
    dqn.optimizer.param_groups[0]["lr"] = saved_lr

    run_dir = tmp_path / "save_roundtrip"
    dqn.save_state(dir=str(run_dir))

    restored = DQN(
        config=config,
        n_actions=2,
        state_size=len(state),
    )
    restored.load_state(dir=str(run_dir))

    for p_saved, p_restored in zip(
        dqn.q_network.parameters(), restored.q_network.parameters(), strict=True
    ):
        assert torch.equal(p_saved, p_restored)

    for p_saved, p_restored in zip(
        dqn.target_network.parameters(),
        restored.target_network.parameters(),
        strict=True,
    ):
        assert torch.equal(p_saved, p_restored)

    assert restored.optimizer.param_groups[0]["lr"] == saved_lr


def test_q_learning() -> None:
    """Test Q-Learning"""

    # Learning hyperparameters
    learning_cfg = LearningConfig()
    initial_state = np.array([0, 2, -1, 7, 0.5], dtype=np.float32)
    n_actions = 4

    ql = Qlearning(
        config=learning_cfg,
        n_actions=n_actions,
        meta_ctrl=MetaController(
            config=MetaConfig(),
            learning_cfg=learning_cfg,
            decision_cfg=DecisionConfig(),
        ),
    )

    q_values = ql.get_action_values(state=initial_state)
    assert q_values.shape == (n_actions,)
