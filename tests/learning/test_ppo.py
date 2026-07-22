"""
Unit tests for the PPO algorithm (PPOEstimator).
"""

from unittest.mock import patch

import numpy as np
import torch

from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.ppo import (
    ActorCriticNetwork,
    ActorNetwork,
    CriticNetwork,
    PPOEstimator,
)

N_ACTIONS = 4
STATE_SIZE = 5


def make_config(
    k_epochs: int = 2,
    gamma: float = 0.99,
    lam: float = 0.95,
    eps_clip: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    lr_actor: float = 3e-4,
    lr_critic: float = 1e-3,
) -> LearningConfig:
    return LearningConfig(
        algorithm="ppo",
        gamma=gamma,
        lam=lam,
        eps_clip=eps_clip,
        k_epochs=k_epochs,
        entropy_coef=entropy_coef,
        value_coef=value_coef,
        lr_actor=lr_actor,
        lr_critic=lr_critic,
    )


def make_estimator(
    k_epochs: int = 2,
    gamma: float = 0.99,
    lam: float = 0.95,
    eps_clip: float = 0.2,
    entropy_coef: float = 0.01,
    value_coef: float = 0.5,
    lr_actor: float = 3e-4,
    lr_critic: float = 1e-3,
) -> PPOEstimator:
    return PPOEstimator(
        config=make_config(
            k_epochs=k_epochs,
            gamma=gamma,
            lam=lam,
            eps_clip=eps_clip,
            entropy_coef=entropy_coef,
            value_coef=value_coef,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
        ),
        n_actions=N_ACTIONS,
        state_size=STATE_SIZE,
    )


class TestNetworks:
    def test_actor_output_shape(self) -> None:
        net = ActorNetwork(input_size=STATE_SIZE, output_size=N_ACTIONS)
        x = torch.zeros(1, STATE_SIZE)
        out = net(x)
        assert out.shape == (1, N_ACTIONS)

    def test_actor_output_sums_to_one(self) -> None:
        net = ActorNetwork(input_size=STATE_SIZE, output_size=N_ACTIONS)
        x = torch.randn(8, STATE_SIZE)
        probs = net(x)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(8), atol=1e-5)

    def test_critic_output_shape(self) -> None:
        net = CriticNetwork(input_size=STATE_SIZE)
        x = torch.zeros(3, STATE_SIZE)
        out = net(x)
        assert out.shape == (3, 1)

    def test_actor_critic_forward(self) -> None:
        net = ActorCriticNetwork(input_size=STATE_SIZE, output_size=N_ACTIONS)
        x = torch.zeros(2, STATE_SIZE)
        probs, values = net(x)
        assert probs.shape == (2, N_ACTIONS)
        assert values.shape == (2, 1)

    def test_dynamic_hidden_sizes(self) -> None:
        """Networks should accept arbitrary depth via hidden_sizes tuple."""
        net = ActorCriticNetwork(
            input_size=STATE_SIZE, output_size=N_ACTIONS, hidden_sizes=(64, 64, 64)
        )
        x = torch.zeros(1, STATE_SIZE)
        probs, values = net(x)
        assert probs.shape == (1, N_ACTIONS)
        assert values.shape == (1, 1)


class TestPPOEstimator:
    def test_get_action_values_shape(self) -> None:
        """get_action_values must return a probability vector of length n_actions."""
        est = make_estimator()
        state = np.zeros(STATE_SIZE, dtype=np.float32)
        values = est.get_action_values(state)
        assert values.shape == (N_ACTIONS,)
        assert np.isclose(values.sum(), 1.0, atol=1e-5)

    def test_action_probabilities_valid(self) -> None:
        """All probabilities must be non-negative."""
        est = make_estimator()
        state = np.random.randn(STATE_SIZE).astype(np.float32)
        probs = est.get_action_values(state)
        assert (probs >= 0).all()

    def test_rollout_buffer_empty_after_update(self) -> None:
        """Buffers must be cleared after a terminal transition."""
        est = make_estimator()
        state = np.zeros(STATE_SIZE, dtype=np.float32)

        for i in range(5):
            done = i == 4
            est.update_action_values(
                state=state,
                action=i % N_ACTIONS,
                reward=1.0,
                next_state=state,
                done=done,
                terminated=done,
            )

        assert len(est._states) == 0
        assert len(est._rewards) == 0

    def test_weights_change_after_update(self) -> None:
        """Network parameters must change after a complete episode update."""
        est = make_estimator(k_epochs=2)
        state = np.zeros(STATE_SIZE, dtype=np.float32)

        params_before = [p.clone() for p in est.model.parameters()]

        # Run a complete episode (5 steps, last step done=True)
        for i in range(5):
            done = i == 4
            est.update_action_values(
                state=state,
                action=i % N_ACTIONS,
                reward=float(i),
                next_state=state,
                done=done,
                terminated=done,
            )

        changed = any(
            not torch.equal(p_before, p_after)
            for p_before, p_after in zip(
                params_before, est.model.parameters(), strict=True
            )
        )
        assert changed, "Network weights must change after a PPO update"

    def test_no_update_mid_episode(self) -> None:
        """Weights must not change before a done=True transition."""
        est = make_estimator()
        state = np.zeros(STATE_SIZE, dtype=np.float32)

        params_before = [p.clone() for p in est.model.parameters()]

        # Store transitions without episode end
        for _ in range(3):
            est.update_action_values(
                state=state, action=0, reward=1.0, next_state=state, done=False
            )

        for p_before, p_after in zip(
            params_before, est.model.parameters(), strict=True
        ):
            assert torch.equal(p_before, p_after), "Weights changed before episode end"

    def test_losses_recorded_after_episode(self) -> None:
        """self.losses and self.mean_q_values must be populated after an update."""
        est = make_estimator()
        state = np.zeros(STATE_SIZE, dtype=np.float32)

        for i in range(4):
            done = i == 3
            est.update_action_values(
                state=state,
                action=i % N_ACTIONS,
                reward=1.0,
                next_state=state,
                done=done,
                terminated=done,
            )

        assert len(est.losses) == 1
        assert len(est.mean_q_values) == 1

    def test_truncated_episode_bootstrap(self) -> None:
        """When terminated=False (truncated), last_value should come from next_state V."""
        est = make_estimator()
        state = np.zeros(STATE_SIZE, dtype=np.float32)
        next_state = np.ones(STATE_SIZE, dtype=np.float32)

        for i in range(3):
            done = i == 2
            est.update_action_values(
                state=state,
                action=i % N_ACTIONS,
                reward=1.0,
                next_state=next_state,
                done=done,
                terminated=False,  # truncated, not a true terminal
            )

        # Buffer should be cleared and weights should have changed
        assert len(est._states) == 0

    def test_multiple_episodes(self) -> None:
        """Multiple consecutive episodes must each trigger an independent update."""
        est = make_estimator()
        state = np.zeros(STATE_SIZE, dtype=np.float32)

        for episode in range(3):
            for i in range(5):
                done = i == 4
                est.update_action_values(
                    state=state,
                    action=i % N_ACTIONS,
                    reward=float(episode + i),
                    next_state=state,
                    done=done,
                    terminated=done,
                )

        # 3 episodes => 3 update calls => 3 loss entries
        assert len(est.losses) == 3

    def test_save_and_load(self, tmp_path) -> None:
        """Save/load round-trip must restore actor, critic, and optimizer state."""
        est = make_estimator()

        # Fix known weights for comparison
        with torch.no_grad():
            for i, p in enumerate(est.model.actor.parameters()):
                p.fill_(0.1 * (i + 1))
            for i, p in enumerate(est.model.critic.parameters()):
                p.fill_(0.2 * (i + 1))

        saved_lr = 1e-5
        est.optimizer.param_groups[0]["lr"] = saved_lr

        run_dir = tmp_path / "ppo_save"
        est.save_state(dir=str(run_dir))

        restored = make_estimator()
        restored.load_state(dir=str(run_dir))

        for p_saved, p_restored in zip(
            est.model.actor.parameters(), restored.model.actor.parameters(), strict=True
        ):
            assert torch.equal(p_saved, p_restored)

        for p_saved, p_restored in zip(
            est.model.critic.parameters(),
            restored.model.critic.parameters(),
            strict=True,
        ):
            assert torch.equal(p_saved, p_restored)

        assert restored.optimizer.param_groups[0]["lr"] == saved_lr

    def test_config_hyperparameters_used(self) -> None:
        """k_epochs from config must control the number of optimizer.step() calls.

        One complete episode triggers exactly k_epochs gradient steps.
        We count calls to optimizer.step() via unittest.mock.patch.
        """
        state = np.zeros(STATE_SIZE, dtype=np.float32)

        for k in (1, 3):
            est = make_estimator(k_epochs=k)

            with patch.object(
                est.optimizer, "step", wraps=est.optimizer.step
            ) as mock_step:
                for i in range(3):
                    done = i == 2
                    est.update_action_values(
                        state=state,
                        action=i % N_ACTIONS,
                        reward=1.0,
                        next_state=state,
                        done=done,
                        terminated=done,
                    )

                assert mock_step.call_count == k, (
                    f"Expected {k} optimizer.step() calls, got {mock_step.call_count}"
                )
