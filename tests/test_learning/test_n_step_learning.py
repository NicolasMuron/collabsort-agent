import numpy as np
import pytest
import torch
from tests.test_learning.test_dqn import TestDQN
from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.n_step_learning import NStepLearning


class TestNStepLearning(TestDQN):  # On garde l'héritage pour tester la conformité DQN
    
    def _make_dqn(self, **kwargs) -> NStepLearning:
        config = LearningConfig(
            batch_size=kwargs.get('batch_size', 32),
            replay_buffer_size=kwargs.get('replay_buffer_size', 1000),
            target_network_sync_freq=kwargs.get('target_sync_freq', 100),
            gamma=0.99
        )
        
        # Astuce : Par défaut à 1 pour que les tests hérités (DQN standard) passent.
        # Mais si le test demande explicitement une autre valeur, on l'applique.
        n_step = kwargs.get('n_step', 1)
        
        agent = NStepLearning(
            config=config, 
            n_actions=kwargs.get('n_actions', 4), 
            state_size=kwargs.get('state_size', 5),
            n_step=n_step
        )
        agent.losses = []
        agent.mean_q_values = []
        return agent

    # -------------------------------------------------------------------------
    # ON NE SURCHARGE PLUS RIEN ! 
    # À la place, on écrit de NOUVEAUX tests spécifiques pour valider le comportement N=5
    # -------------------------------------------------------------------------

    def test_n_step_replay_buffer_delay_with_n5(self) -> None:
        """Vérifie le comportement de la taille du buffer spécifiquement pour N=5."""
        state = np.zeros(5, dtype=np.float32)
        
        # On force n_step=5 ici !
        dqn = self._make_dqn(replay_buffer_size=10, n_step=5)

        # Les 4 premières transitions remplissent le buffer local, rien dans le global
        for _ in range(4):
            dqn._store_transition(state=state, action=0, reward=1.0, next_state=state)
        assert len(dqn.replay_buffer) == 0
        assert len(dqn.n_step_buffer) == 4

        # La 5ème transition déclenche le premier envoi dans le replay buffer global
        dqn._store_transition(state=state, action=0, reward=1.0, next_state=state)
        assert len(dqn.replay_buffer) == 1

    def test_store_transition_fills_buffer_with_n3(self):
        """Vérifie le calcul mathématique exact des gains cumulés pour N=3."""
        agent = self._make_dqn(n_step=3)
        state = np.zeros(5, dtype=np.float32)

        agent._store_transition(state, 0, 1.0, state, False)
        agent._store_transition(state, 1, 2.0, state, False)
        
        # 3ème étape -> calcul du rendement R
        agent._store_transition(state, 2, 3.0, state, False)
        assert len(agent.replay_buffer) == 1

        stored = agent.replay_buffer.buffer[0]
        expected_R = 1.0 + (0.99 * 2.0) + (0.99**2 * 3.0)
        assert pytest.approx(stored[2]) == expected_R

    def test_store_transition_done_flushes_buffer(self):
        """Vérifie que done=True force la vidange immédiate même si N=5."""
        agent = self._make_dqn(n_step=5)
        state = np.zeros(5, dtype=np.float32)

        agent._store_transition(state, 0, 1.0, state, False)
        agent._store_transition(state, 0, 1.0, state, False)
        
        # Épisode fini, le buffer doit se vider directement
        agent._store_transition(state, 0, 1.0, state, True)
        assert len(agent.n_step_buffer) == 0
        assert len(agent.replay_buffer) == 3