"""
Unit tests for the QL algorithm.
"""

import numpy as np

from collabsort_agent.decision import Config as DecisionConfig
from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.q_learning import Qlearning
from collabsort_agent.metacognition import Config as MetaConfig
from collabsort_agent.metacognition import MetaController


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