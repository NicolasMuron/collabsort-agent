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
        
        # Ajout d'une transition avec une priorité de 1.5
        tree.add(priority=1.5, data=("state", "action"))
        assert tree.total_priority() == 1.5
        assert tree.n_entries == 1

        # Ajout d'une deuxième transition avec une priorité de 2.5
        tree.add(priority=2.5, data=("next_state", "reward"))
        assert tree.total_priority() == 4.0
        assert tree.n_entries == 2

    def test_update_priority(self) -> None:
        capacity = 2
        tree = SumTree(capacity=capacity)
        tree.add(priority=1.0, data="data1") # Index de feuille = 1 + 2 - 1 = 2
        tree.add(priority=2.0, data="data2") # Index de feuille = 3
        
        assert tree.total_priority() == 3.0

        # Mise à jour de la première feuille (index 1 dans l'arbre complet)
        tree.update(tree_index=1, priority=5.0)
        assert tree.total_priority() == 7.0

    def test_get_leaf(self) -> None:
        capacity = 2
        tree = SumTree(capacity=capacity)
        tree.add(priority=1.0, data="first")  # Occupe l'intervalle [0.0, 1.0[
        tree.add(priority=3.0, data="second") # Occupe l'intervalle [1.0, 4.0[

        # Recherche dans le premier segment
        idx1, p1, data1 = tree.get_leaf(0.5)
        assert data1 == "first"
        assert p1 == 1.0

        # Recherche dans le second segment
        idx2, p2, data2 = tree.get_leaf(2.5)
        assert data2 == "second"
        assert p2 == 3.0


class TestPER(TestDQN):
    
    def _make_dqn(
        self,
        n_actions: int = 4,
        state_size: int = 5,
        batch_size: int = 4,
        replay_buffer_size: int = 100,
        target_sync_freq: int = 500,
    ) -> PER:
        """Surcharge du helper pour instancier PER à la place de DQN."""
        config = LearningConfig(
            batch_size=batch_size,
            replay_buffer_size=replay_buffer_size,
            target_network_sync_freq=target_sync_freq,
        )
        # Forcer l'attribut du nombre de pas en dur pour éviter le bug d'importation circulaire
        agent = PER(config=config, n_actions=n_actions, state_size=state_size)
        agent.n_steps = 300000 * agent.ratio
        agent.per_beta_increment = (1 - agent.per_beta) / agent.n_steps
        return agent

    def test_per_buffer_type(self) -> None:
        per_agent = self._make_dqn(replay_buffer_size=32)
        # Vérifie que le buffer par défaut (deque) a bien été remplacé par un SumTree
        assert isinstance(per_agent.tree, SumTree)
        assert per_agent.tree.capacity == 32

    def test_store_transition_max_priority(self) -> None:
        per_agent = self._make_dqn()
        state = np.zeros(5, dtype=np.float32)

        # Stocker une première transition
        per_agent._store_transition(state=state, action=0, reward=1.0, next_state=state)
        
        # Ton code force la priorité max à 1 à l'initialisation : max_priority = 1
        assert per_agent.tree.total_priority() == 1.0
        assert per_agent.tree.n_entries == 1

    def test_sample_weights_shape_and_normalization(self) -> None:
        batch_size = 4
        per_agent = self._make_dqn(batch_size=batch_size, replay_buffer_size=10)
        state = np.zeros(5, dtype=np.float32)

        # Remplir le buffer pour pouvoir échantillonner un batch complet
        for i in range(batch_size + 2):
            per_agent._store_transition(state=state, action=0, reward=float(i), next_state=state)

        minibatch, idxs, is_weights = per_agent._sample()

        # Vérifications des structures renvoyées
        assert len(minibatch) == batch_size
        assert len(idxs) == batch_size
        assert len(is_weights) == batch_size
        
        # Vérification de la normalisation de l'Importance Sampling (le poids max doit valoir 1.0)
        assert np.isclose(np.max(is_weights), 1.0)

    def test_beta_increment(self) -> None:
        per_agent = self._make_dqn(batch_size=2)
        state = np.zeros(5, dtype=np.float32)

        # Injecter des fausses données
        for _ in range(4):
            per_agent._store_transition(state=state, action=0, reward=1.0, next_state=state)

        initial_beta = per_agent.per_beta
        
        # Simuler un appel à _sample() qui doit déclencher l'incrément de beta
        _, _, _ = per_agent._sample()
        
        # Beta doit avoir augmenté
        assert per_agent.per_beta > initial_beta
        assert per_agent.per_beta == min(1.0, initial_beta + per_agent.per_beta_increment)
        
    def test_sum_tree_propagate_root_or_negative(self) -> None:
        """Couvre la ligne 39 (sécurité `if idx <= 0:` de _propagate)."""
        tree = SumTree(capacity=2)
        # Appeler _propagate directement avec un index <= 0 doit retourner immédiatement
        assert tree._propagate(idx=0, change=1.0) is None
        assert tree._propagate(idx=-5, change=1.0) is None


    def test_per_store_transition_with_zero_priority(self) -> None:
        """Couvre la ligne 126 (sécurité `if max_priority == 0: max_priority = 1.0`)."""
        config = LearningConfig(batch_size=2, replay_buffer_size=4)
        agent = PER(config=config, n_actions=2, state_size=2)
        
        # On insère une transition avec une priorité nulle forcée
        agent.tree.add(priority=0.0, data=("state", 0, 0.0, "next_state", False))
        assert agent.tree.n_entries == 1
        
        # On stocke une nouvelle transition : le code doit basculer sur 1.0 par défaut
        state = np.zeros(2, dtype=np.float32)
        agent._store_transition(state=state, action=0, reward=1.0, next_state=state)
        
        # La feuille 1 (index global 4) doit avoir reçu la priorité par défaut 1.0
        assert agent.tree.tree[4] == 1.0


    def test_per_sample_empty_or_zero_priority_tree(self) -> None:
        """Couvre la ligne 138 (sécurité `if total_p == 0: total_p = 1.0`)."""
        config = LearningConfig(batch_size=2, replay_buffer_size=4)
        agent = PER(config=config, n_actions=2, state_size=2)
        
        agent.tree.add(priority=0.0, data=("state1", 0, 0.0, "next_state1", False))
        agent.tree.add(priority=0.0, data=("state2", 0, 0.0, "next_state2", False))
        agent.tree.tree[:] = 0.0  # On s'assure que total_priority() renvoie absolument 0.0
        
        # L'échantillonnage ne doit pas lever de ZeroDivisionError
        minibatch, idxs, is_weights = agent._sample()
        assert len(minibatch) == 2


    def test_per_sample_empty_leaf_resampling(self) -> None:
        """Couvre les lignes 159-161 (la boucle `while (not isinstance(data, ...))`)."""
        config = LearningConfig(batch_size=1, replay_buffer_size=2)
        agent = PER(config=config, n_actions=2, state_size=2)
        
        agent.tree.add(priority=5.0, data=("state1", 0, 0.0, "next_state1", False))
        
        # On injecte une donnée invalide (un entier) à l'index 0 pour activer le while de ré-échantillonnage
        agent.tree.data[0] = 0
        
        minibatch, idxs, is_weights = agent._sample()
        assert len(minibatch) == 1


    def test_per_sample_max_weight_zero_security(self) -> None:
        """Couvre la ligne 179 (sécurité `if max_weight == 0: max_weight = 1.0`)."""
        config = LearningConfig(batch_size=2, replay_buffer_size=4)
        agent = PER(config=config, n_actions=2, state_size=2)
        
        state = np.zeros(2, dtype=np.float32)
        agent._store_transition(state=state, action=0, reward=1.0, next_state=state)
        agent._store_transition(state=state, action=0, reward=2.0, next_state=state)
        
        original_power = np.power
        try:
            # On force temporairement np.power à retourner des zéros pour simuler max_weight == 0
            np.power = lambda *args, **kwargs: np.zeros(config.batch_size)
            minibatch, idxs, is_weights = agent._sample()
            assert np.all(is_weights == 0.0)
        finally:
            np.power = original_power