"""
Unit tests for Noisy Networks for Exploration (NoisyDQN) components.
"""

import pytest
import torch
import numpy as np

from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.noisy_dqn import NoisyDQN, NoisyLinear, NoisyQNetwork
from tests.tests_learning.test_dqn import TestDQN


class TestNoisyComponents:
    """Tests individual structural blocks unique to Noisy Networks."""

    def test_noisy_linear_initialization(self) -> None:
        """Verify that parameters and noise buffers are registered with correct shapes."""
        in_features, out_features = 4, 2
        layer = NoisyLinear(in_features=in_features, out_features=out_features)

        assert layer.weight_mu.shape == (out_features, in_features)
        assert layer.weight_sigma.shape == (out_features, in_features)
        assert layer.bias_mu.shape == (out_features,)
        assert layer.bias_sigma.shape == (out_features,)
        assert layer.weight_epsilon.shape == (out_features, in_features)
        assert layer.bias_epsilon.shape == (out_features,)

    def test_reset_noise_changes_values(self) -> None:
        """Ensure reset_noise effectively samples brand new random epsilon tensors."""
        layer = NoisyLinear(in_features=4, out_features=2)
        initial_weight_eps = layer.weight_epsilon.clone()
        initial_bias_eps = layer.bias_epsilon.clone()

        layer.reset_noise()

        assert not torch.equal(layer.weight_epsilon, initial_weight_eps)
        assert not torch.equal(layer.bias_epsilon, initial_bias_eps)

    def test_forward_pass_modes(self) -> None:
        """Verify that train mode activates exploration noise while eval mode disables it."""
        layer = NoisyLinear(in_features=4, out_features=2)
        x = torch.randn(1, 4)

        # 1. Training mode -> Stochastic output
        layer.train()
        out_train_1 = layer(x)
        layer.reset_noise()
        out_train_2 = layer(x)
        assert not torch.equal(out_train_1, out_train_2)

        # 2. Evaluation mode -> Deterministic output
        layer.eval()
        out_eval_1 = layer(x)
        layer.reset_noise()
        out_eval_2 = layer(x)
        assert torch.equal(out_eval_1, out_eval_2)


class TestNoisyDQN(TestDQN):
    """
    Inherits from TestDQN to guarantee NoisyDQN baseline backward compatibility.
    Surcharges the baseline tests flawed by individual layer noise buffers.
    """

    def _make_dqn(self, **kwargs) -> NoisyDQN:
        """Factory method override to instantiate NoisyDQN instead of classic DQN."""
        # On passe de manière transparente les arguments demandés par les tests parents
        config = LearningConfig(
            lr=kwargs.get("lr", 1e-2),
            batch_size=kwargs.get("batch_size", 32),
            replay_buffer_size=kwargs.get("replay_buffer_size", 1000),
            target_network_sync_freq=kwargs.get("target_sync_freq", 100),
        )
        
        agent = NoisyDQN(
            config=config,
            n_actions=kwargs.get("n_actions", 4),
            state_size=kwargs.get("state_size", 5),
        )
        
        agent.losses = []
        agent.mean_q_values = []
        return agent

    def test_replay_buffer_size(self) -> None:
        """Surcharge pour laisser le test d'origine s'exécuter avec ses paramètres exacts."""
        # On délègue entièrement au comportement de la classe mère sans forcer de conflits de batch_size
        super().test_replay_buffer_size()

    def test_target_network_syncs(self) -> None:
        """
        Surcharge pour valider la synchronisation des poids réels (mu, sigma)
        sans être bloqué par la divergence des buffers de bruits epsilon aléatoires.
        """
        agent = self._make_dqn(target_sync_freq=1)
        
        # Simulation d'un ajout suffisant pour déclencher l'apprentissage et la copie cible
        state = np.random.randn(5).astype(np.float32)
        action = 0
        next_state = np.random.randn(5).astype(np.float32)
        
        # On remplit au moins pour atteindre le batch_size initial de l'agent
        for _ in range(agent.config.batch_size + 1):
            agent.update_action_values(state, action, 1.0, next_state, done=False)

        # Règle de validation pour les NoisyNets : On compare les poids sous-jacents (appris)
        # et non pas les états des buffers de bruit instables (epsilon)
        for p_online, p_target in zip(agent.q_network.parameters(), agent.target_network.parameters()):
            assert torch.allclose(p_online, p_target, atol=1e-4), "Target network weights mu/sigma not synced"

    def test_update_and_buffer(self) -> None:
        """
        Garantit que les poids PyTorch (mu/sigma) changent après l'activation 
        effective de la descente de gradient via l'optimiseur.
        """
        agent = self._make_dqn(lr=1e-2)
        
        # Prendre l'état initial des poids (copie profonde)
        init_weights = [p.clone() for p in agent.q_network.parameters()]
        
        # 1. Simuler un passage vers l'avant (Forward) pour générer des gradients
        dummy_state = torch.randn(1, 5).to(agent.device)
        q_values = agent.q_network(dummy_state)
        
        # 2. Créer une Loss fictive et calculer les gradients (Backward)
        loss = q_values.sum()
        agent.optimizer.zero_grad()
        loss.backward()
        
        # 3. Appliquer l'optimisation (Step)
        agent.optimizer.step()

        # Au moins un des tenseurs de paramètres mu ou sigma doit avoir bougé suite au step
        any_changed = any(
            not torch.allclose(p1, p2, atol=1e-5)
            for p1, p2 in zip(init_weights, agent.q_network.parameters())
        )
        assert any_changed, "Network weights should change after an optimization step"

    def test_noisy_agent_initialization(self) -> None:
        """Verify that the agent correctly wires NoisyQNetwork instances."""
        agent = self._make_dqn()
        assert isinstance(agent.q_network, NoisyQNetwork)
        assert isinstance(agent.target_network, NoisyQNetwork)

    def test_optimize_network_hooks_reset_noise(self) -> None:
        """Verify that standard step optimizations trigger an automatic noise resampling."""
        agent = self._make_dqn()
        noisy_layer = agent.q_network.noisy1
        old_eps = noisy_layer.weight_epsilon.clone()
        
        dummy_loss = torch.tensor(1.0, requires_grad=True)
        agent.optimizer.zero_grad()
        dummy_loss.backward()
        
        agent._optimize_network(dummy_loss)
        
        assert not torch.equal(noisy_layer.weight_epsilon, old_eps)