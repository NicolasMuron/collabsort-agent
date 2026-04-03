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


class TestDQN:
    def test_action_values_shape(self) -> None:
        state = np.zeros(5, dtype=np.float32)
        n_actions = 4

        dqn = self._make_dqn(n_actions=n_actions, state_size=len(state))

        q_values = dqn.get_action_values(state=state)
        assert q_values.shape == (n_actions,)

    def test_replay_buffer_size(self) -> None:
        replay_buffer_size = 10
        state = np.zeros(5, dtype=np.float32)

        dqn = self._make_dqn(replay_buffer_size=replay_buffer_size)

        dqn._store_transition(state=state, action=0, reward=1, next_state=state * 1.1)
        assert len(dqn.replay_buffer) == 1

        # Replau buffer size should be capped
        for _ in range(replay_buffer_size + 5):
            dqn._store_transition(state=state, action=0, reward=0.0, next_state=state)
        assert len(dqn.replay_buffer) == replay_buffer_size

    def test_update_and_buffer(self) -> None:
        state = np.zeros(5, dtype=np.float32)
        batch_size = 8

        dqn = self._make_dqn(batch_size=batch_size)

        params_before = [p.clone() for p in dqn.q_network.parameters()]

        # Store fewer transitions than batch_size
        for _ in range(4):
            dqn.update_action_values(
                state=state, action=0, reward=0.0, next_state=state
            )

        for p_before, p_after in zip(
            params_before, dqn.q_network.parameters(), strict=True
        ):
            assert torch.equal(p_before, p_after), (
                "Weights should not change below batch_size"
            )

        # Fill buffer and trigger learning
        for i in range(10):
            dqn.update_action_values(
                state=state, action=i % 4, reward=float(i), next_state=state
            )

        changed = any(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(
                params_before, dqn.q_network.parameters(), strict=True
            )
        )
        assert changed, "Network weights should change after enough transitions"

    def test_target_network_syncs(self) -> None:
        """Target network should be synced to online network after sync_freq _learn() calls"""

        state = np.zeros(4, dtype=np.float32)
        n_actions = 2
        target_network_sync_freq = 5
        batch_size = 4
        replay_buffer_size = 100

        dqn = self._make_dqn(
            n_actions=n_actions,
            state_size=len(state),
            batch_size=batch_size,
            replay_buffer_size=replay_buffer_size,
            target_sync_freq=target_network_sync_freq,
        )

        # Diverge the target network manually
        with torch.no_grad():
            for p in dqn.target_network.parameters():
                p.fill_(99.0)

        # Fill replay buffer above batch_size threshold
        for i in range(batch_size):
            dqn._store_transition(
                state=state, action=0, reward=float(i), next_state=state
            )

        # Run fewer than sync_freq steps
        for _ in range(target_network_sync_freq - 1):
            dqn._learn()

        # Target network should still differ from online network
        all_equal = all(
            (p_online.data == p_target.data).all()
            for p_online, p_target in zip(
                dqn.q_network.parameters(), dqn.target_network.parameters(), strict=True
            )
        )
        assert not all_equal, "Target network should not sync before sync_freq steps"

        # Run one more learning step to trigger at least one sync
        dqn._learn()

        # After sync, target and online networks should have identical weights
        for p_online, p_target in zip(
            dqn.q_network.parameters(), dqn.target_network.parameters(), strict=True
        ):
            assert (p_online.data == p_target.data).all(), "Target network not synced"

    def test_done_flag(self) -> None:
        state = np.zeros(5, dtype=np.float32)
        batch_size = 4

        dqn = self._make_dqn(batch_size=batch_size)

        # Store transitions with done=True
        for _ in range(batch_size):
            dqn._store_transition(
                state=state, action=0, reward=1.0, next_state=state * 2, done=True
            )

        # Should run without error
        dqn._learn()

    def test_save_and_load(self, tmp_path) -> None:
        """Saving and loading should restore DQN internal state"""

        state = np.zeros(4, dtype=np.float32)

        dqn = self._make_dqn(state_size=len(state))

        with torch.no_grad():
            for i, param in enumerate(dqn.q_network.parameters()):
                param.fill_(0.1 * (i + 1))
            for i, param in enumerate(dqn.target_network.parameters()):
                param.fill_(0.2 * (i + 1))

        saved_lr = 5e-4
        dqn.optimizer.param_groups[0]["lr"] = saved_lr

        run_dir = tmp_path / "save_roundtrip"
        dqn.save_state(dir=str(run_dir))

        restored = self._make_dqn(state_size=len(state))
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

    def _make_dqn(
        self,
        n_actions: int = 4,
        state_size: int = 5,
        batch_size: int = 4,
        replay_buffer_size: int = 100,
        target_sync_freq: int = 500,
    ) -> DQN:
        config = LearningConfig(
            batch_size=batch_size,
            replay_buffer_size=replay_buffer_size,
            target_network_sync_freq=target_sync_freq,
        )
        return DQN(config=config, n_actions=n_actions, state_size=state_size)


class TestQLearning:
    def test_action_values(self) -> None:
        state = np.array([0, 2, -1, 7, 0.5], dtype=np.float32)

        ql = self._make_qlearning(n_actions=4, q_start=3.0)

        q_values = ql.get_action_values(state=state)
        assert q_values.shape == (4,)
        assert (q_values == 3.0).all()

        # get_action_values() should return a copy
        q_values[:] = 999.0
        assert not (ql.get_action_values(state) == 999.0).any()

    def test_update_action_value(self) -> None:
        state = np.array([0.0, 1.0], dtype=np.float32)

        ql = self._make_qlearning(n_actions=3)

        q_before = ql.get_action_values(state).copy()
        ql.update_action_values(state=state, action=0, reward=5.0, next_state=state)
        q_after = ql.get_action_values(state)

        # Actions 1 and 2 should be unchanged
        assert q_after[0] != q_before[0], "Q-value for updated action should change"
        assert q_after[1] == q_before[1]
        assert q_after[2] == q_before[2]

    def test_update_from_different_states(self) -> None:
        s1 = np.array([1.0, 0.0], dtype=np.float32)
        s2 = np.array([0.0, 1.0], dtype=np.float32)

        ql = self._make_qlearning(n_actions=2)

        ql.update_action_values(state=s1, action=0, reward=5.0, next_state=s1)
        # s2 should still have the initial q_start value
        assert (ql.get_action_values(s2) == LearningConfig().q_start).all()

    def test_done_flag(self) -> None:
        """When done=True, target should be r only (no future reward)."""

        config = LearningConfig(gamma=0.99, q_start=0.0)
        state = np.array([1.0], dtype=np.float32)
        next_state = np.array([999.0], dtype=np.float32)  # shouldn't matter

        ql = Qlearning(config=config, n_actions=2, meta_ctrl=self._make_meta())

        ql.update_action_values(
            state=state, action=0, reward=1.0, next_state=next_state, done=True
        )
        # With q_start=0, alpha=0.1: Q(s,a) = 0 + 0.1*(1 - 0) = 0.1
        assert abs(ql.get_action_values(state)[0] - 0.1) < 1e-6

    def _make_qlearning(self, n_actions: int = 4, q_start: float = 0.0) -> Qlearning:
        return Qlearning(
            config=LearningConfig(q_start=q_start),
            n_actions=n_actions,
            meta_ctrl=self._make_meta(),
        )

    def _make_meta(self) -> MetaController:
        return MetaController(
            config=MetaConfig(),
            learning_cfg=LearningConfig(),
            decision_cfg=DecisionConfig(),
        )
