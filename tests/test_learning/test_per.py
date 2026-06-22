"""
Unit tests for the PER (Prioritized Experience Replay) algorithm and SumTree.
"""
import numpy as np
import torch

from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.per import PER, SumTree
from tests.test_learning.test_dqn import TestDQN


class TestSumTree:
    def test_tree_initialization(self) -> None:
        capacity = 4
        tree = SumTree(capacity=capacity)
        assert tree.capacity == capacity
        assert len(tree.tree) == 2 * capacity - 1
        assert len(tree.data) == capacity
        assert tree.total_priority() == 0.0

    def test_add_and_propagate(self) -> None:
        capacity = 4
        tree = SumTree(capacity=capacity)
        
        # Add a transition with an initial priority of 1.5
        tree.add(priority=1.5, data=("state", "action"))
        assert tree.total_priority() == 1.5
        assert tree.n_entries == 1

        # Add a second transition with an initial priority of 2.5
        tree.add(priority=2.5, data=("next_state", "reward"))
        assert tree.total_priority() == 4.0
        assert tree.n_entries == 2

    def test_update_priority(self) -> None:
        capacity = 4
        tree = SumTree(capacity=capacity)
        tree.add(priority=1.0, data=("s", "a"))  # index 3 in tree array
        
        tree_idx = 3
        tree.update(tree_idx, 5.0)
        assert tree.total_priority() == 5.0

    def test_get_leaf(self) -> None:
        capacity = 4
        tree = SumTree(capacity=capacity)
        tree.add(priority=1.0, data="data1")  # leaf index 3, cumulative priority range [0, 1]
        tree.add(priority=2.0, data="data2")  # leaf index 4, cumulative priority range [1, 3]
        
        # Searching inside the cumulative tree ranges
        idx, p, data = tree.get_leaf(0.5)
        assert data == "data1"
        assert p == 1.0
        
        idx, p, data = tree.get_leaf(2.5)
        assert data == "data2"
        assert p == 2.0

        

class TestPER(TestDQN):
    def _make_dqn(self, **kwargs) -> PER:
        """Factory method override to inject PER instead of standard DQN."""
        config = LearningConfig(
            batch_size=kwargs.get('batch_size', 32),
            replay_buffer_size=kwargs.get('replay_buffer_size', 1000),
            target_network_sync_freq=kwargs.get('target_sync_freq', 100),
        )
        agent = PER(config=config, n_actions=kwargs.get('n_actions', 4), state_size=kwargs.get('state_size', 5))
        agent.losses = []
        agent.mean_q_values = []
        return agent

    def test_per_store_transition(self) -> None:
        """Verify that transitions are added with maximum current priority."""
        agent = self._make_dqn()
        state = np.zeros(5, dtype=np.float32)
        
        # First insertion: tree is empty, uses default max priority = 1.0
        agent._store_transition(state=state, action=0, reward=1.0, next_state=state)
        assert agent.tree.total_priority() == 1.0
        assert agent.tree.n_entries == 1

    def test_per_sample_and_importance_sampling(self) -> None:
        """Verify sampling mechanism and Importance Sampling weight normalization."""
        agent = self._make_dqn(batch_size=2)
        state = np.zeros(5, dtype=np.float32)
        
        # Add 2 samples to satisfy batch_size requirements
        agent._store_transition(state=state, action=0, reward=1.0, next_state=state)
        agent._store_transition(state=state, action=1, reward=2.0, next_state=state)
        
        minibatch, idxs, is_weights = agent._sample()
        assert len(minibatch) == 2
        assert len(idxs) == 2
        assert len(is_weights) == 2
        # Max weight normalization should bound the maximum weight to 1.0
        assert np.max(is_weights) == 1.0 

    def test_per_learn_execution(self) -> None:
        """Verify execution of the custom weighted _learn loop."""
        agent = self._make_dqn(batch_size=2)
        state = np.zeros(5, dtype=np.float32)
        
        agent._store_transition(state=state, action=0, reward=1.0, next_state=state)
        agent._store_transition(state=state, action=1, reward=2.0, next_state=state)
        
        agent._learn()
        assert len(agent.losses) == 1

    # -------------------------------------------------------------------------
    # EDGE CASE COGNITIVE & WHITE-BOX COVERAGE TESTS
    # -------------------------------------------------------------------------

    def test_sum_tree_propagate_root_security(self) -> None:
        """Covers branch condition: if idx <= 0 inside _propagate."""
        tree = SumTree(capacity=2)
        # Call with 0 index should return immediately without crash
        tree._propagate(idx=0, change=10.0) 
        assert tree.total_priority() == 0.0

    def test_sum_tree_get_leaf_loop_termination(self) -> None:
        """Covers the binary search traversal logic down to a leaf node."""
        tree = SumTree(capacity=2)
        tree.add(priority=5.0, data="left")
        tree.add(priority=10.0, data="right")
        
        # Requesting priority inside the right-hand cumulative range
        idx, p, data = tree.get_leaf(12.0)
        assert data == "right"
        assert p == 10.0

    def test_per_sample_invalid_data_loop_resampling(self) -> None:
        """Covers loop condition: while (not isinstance(data, tuple)) inside _sample."""
        config = LearningConfig(batch_size=1, replay_buffer_size=2)
        agent = PER(config=config, n_actions=2, state_size=2)
        
        agent.tree.add(priority=5.0, data=("state1", 0, 0.0, "next_state1", False))
        
        # Inject an invalid non-tuple object at data index 0 to trigger resampling loop
        agent.tree.data[0] = 0
        
        minibatch, idxs, is_weights = agent._sample()
        assert len(minibatch) == 1

    def test_per_sample_max_weight_zero_security(self) -> None:
        """Covers fallback safety mechanism: if max_weight == 0: max_weight = 1.0."""
        config = LearningConfig(batch_size=2, replay_buffer_size=4)
        agent = PER(config=config, n_actions=2, state_size=2)
        
        state = np.zeros(2, dtype=np.float32)
        agent._store_transition(state=state, action=0, reward=1.0, next_state=state)
        agent._store_transition(state=state, action=0, reward=1.0, next_state=state)
        
        # Manually force all priority leaves to zero
        agent.tree.tree[:] = 0.0
        
        minibatch, idxs, is_weights = agent._sample()
        # Weights should not cause NaN values or ZeroDivisionError exceptions
        assert not np.isnan(is_weights).any()
        