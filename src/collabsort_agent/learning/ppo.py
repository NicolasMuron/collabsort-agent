"""
Proximal Policy Optimization (PPO) implementation.

Key design choices (aligned with the rest of the project):
- A single combined Adam optimizer over actor + critic parameters.
- One backward pass per epoch on the combined loss
  L = L_actor + value_coef * L_critic - entropy_coef * entropy
  (avoids retain_graph and keeps gradients coherent).
- GAE (Generalized Advantage Estimation) with proper bootstrap:
  last_value = 0 when the episode terminates naturally (terminated=True),
  last_value = V(s') when truncated (time-limit) so we do not cut the return.
- Gradient clipping for training stability.
- Hyperparameters are read directly from LearningConfig (no silent getattr defaults).
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

from collabsort_agent.learning.learning import ActionValueEstimator
from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.dqn import get_device


# ──────────────────────────────────────────────────────────────────────────────
# Neural network modules
# ──────────────────────────────────────────────────────────────────────────────


class ActorNetwork(nn.Module):
    """Stochastic policy network: state -> action probability distribution."""

    def __init__(
        self, input_size: int, output_size: int, hidden_sizes: tuple = (100, 100)
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_size))
        layers.append(nn.Softmax(dim=-1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CriticNetwork(nn.Module):
    """Value network: state -> scalar state-value V(s)."""

    def __init__(self, input_size: int, hidden_sizes: tuple = (100, 100)) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActorCriticNetwork(nn.Module):
    """Shared-input Actor-Critic model (separate heads, no shared trunk)."""

    def __init__(
        self, input_size: int, output_size: int, hidden_sizes: tuple = (100, 100)
    ) -> None:
        super().__init__()
        self.actor = ActorNetwork(input_size, output_size, hidden_sizes)
        self.critic = CriticNetwork(input_size, hidden_sizes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(x), self.critic(x)


# ──────────────────────────────────────────────────────────────────────────────
# PPOEstimator — ActionValueEstimator wrapper
# ──────────────────────────────────────────────────────────────────────────────


class PPOEstimator(ActionValueEstimator):
    """PPO estimator integrated into the project's learning API.

    The estimator accumulates one full episode of on-policy transitions via
    ``update_action_values``, then triggers a PPO update at episode end.

    Action values exposed via ``get_action_values`` are the raw action
    probabilities from the policy network — compatible with ``Greedy``
    and any deliberator that picks argmax.
    """

    # On-policy rollout buffers (filled per episode, cleared after update)
    _states: list[torch.Tensor]
    _actions: list[torch.Tensor]
    _logprobs: list[torch.Tensor]
    _rewards: list[float]
    _dones: list[float]  # 1.0 = terminated (true terminal), 0.0 = truncated / ongoing
    _values: list[torch.Tensor]  # V(s) scalar for each transition

    def __init__(
        self,
        config: LearningConfig,
        n_actions: int,
        state_size: int,
        **kwargs,
    ) -> None:
        super().__init__(config=config, n_actions=n_actions)
        self.state_size = state_size
        self.device = get_device()

        self.model = ActorCriticNetwork(
            input_size=state_size,
            output_size=n_actions,
            hidden_sizes=(100, 100),
        ).to(self.device)

        # One optimizer with two param groups so actor and critic can have
        # independent learning rates while sharing a single backward pass.
        self.optimizer = optim.Adam(
            [
                {"params": self.model.actor.parameters(), "lr": config.lr_actor},
                {"params": self.model.critic.parameters(), "lr": config.lr_critic},
            ]
        )

        self._reset_rollout()

    # ── Rollout helpers ───────────────────────────────────────────────────────

    def _reset_rollout(self) -> None:
        """Clear the on-policy rollout buffers."""
        self._states = []
        self._actions = []
        self._logprobs = []
        self._rewards = []
        self._dones = []
        self._values = []

    def _to_tensor(self, state) -> torch.Tensor:
        """Convert a numpy array or tensor to a batched float32 device tensor."""
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        return state.to(self.device)

    # ── ActionValueEstimator interface ────────────────────────────────────────

    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        """Return action probabilities as 'values' (shape: [n_actions])."""
        state_t = self._to_tensor(state)
        with torch.no_grad():
            probs, _ = self.model(state_t)
        return probs.squeeze(0).cpu().numpy()

    def update_action_values(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
        terminated: bool | None = None,
    ) -> None:
        """Store one transition and trigger a PPO update at episode end.

        Args:
            state:      Current observation.
            action:     Discrete action index (0 ... n_actions-1).
            reward:     Scalar reward received.
            next_state: Next observation (used for bootstrap when truncated).
            done:       True if the episode is over (terminated OR truncated).
            terminated: True only when the episode ended in a *true* terminal
                        state (not a time-limit truncation).  When None, falls
                        back to ``done`` (conservative: treats truncation as
                        termination, which slightly under-estimates returns at
                        episode boundaries).
        """
        state_t = self._to_tensor(state)

        with torch.no_grad():
            probs, value = self.model(state_t)
            dist = Categorical(probs)
            action_t = torch.tensor([action], device=self.device)
            logprob = dist.log_prob(action_t)

        # Store on CPU to avoid accumulating GPU tensors during the rollout
        self._states.append(state_t.squeeze(0).cpu())
        self._actions.append(action_t.cpu())
        self._logprobs.append(logprob.squeeze(0).cpu())
        self._values.append(value.squeeze(0).squeeze(-1).cpu())
        self._rewards.append(float(reward))

        # _dones[t] = 1.0 only when the episode truly ends (no future returns).
        # For truncated episodes the bootstrap must come from V(next_state).
        if terminated is None:
            terminated = done
        self._dones.append(1.0 if terminated else 0.0)

        if done:
            if terminated:
                # True terminal: no future value
                last_value = 0.0
            else:
                # Truncated: bootstrap from the value of the next state
                next_t = self._to_tensor(next_state)
                with torch.no_grad():
                    _, next_v = self.model(next_t)
                last_value = float(next_v.squeeze().item())

            self._ppo_update(last_value=last_value)

    # ── PPO core ──────────────────────────────────────────────────────────────

    def _compute_gae(self, last_value: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GAE advantages and discounted returns (targets for the critic).

        Args:
            last_value: Bootstrap value V(s_{T+1}).
                        0.0 for true terminals, V(next_state) for truncated episodes.

        Returns:
            advantages: GAE estimates        shape [T]
            returns:    Discounted returns   shape [T]
        """
        cfg = self.config
        values_np = [v.item() for v in self._values] + [last_value]

        advantages: list[float] = []
        gae = 0.0
        for t in reversed(range(len(self._rewards))):
            # mask = 0 at a true terminal so future returns are not propagated
            mask = 1.0 - self._dones[t]
            delta = (
                self._rewards[t] + cfg.gamma * values_np[t + 1] * mask - values_np[t]
            )
            gae = delta + cfg.gamma * cfg.lam * mask * gae
            advantages.insert(0, gae)

        returns = [
            adv + val for adv, val in zip(advantages, values_np[:-1], strict=True)
        ]
        adv_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        return adv_t, ret_t

    def _ppo_update(self, last_value: float) -> None:
        """Run k_epochs of PPO gradient updates, then clear the rollout buffer."""
        if not self._states:
            return

        cfg = self.config
        advantages, returns = self._compute_gae(last_value)

        # Assemble batch tensors (move to device once for all epochs)
        states = torch.stack([s.to(self.device) for s in self._states])  # [T, obs]
        actions = (
            torch.stack([a.to(self.device) for a in self._actions]).long().squeeze(-1)
        )  # [T]
        old_logprobs = torch.stack(
            [lp.to(self.device) for lp in self._logprobs]
        ).detach()  # [T]

        # Normalize advantages for numerical stability across different reward scales
        advantages = (advantages - advantages.mean()) / (
            advantages.std(unbiased=False) + 1e-8
        )

        epoch_losses: list[float] = []
        epoch_values: list[float] = []

        for _ in range(cfg.k_epochs):
            action_probs, state_values = self.model(states)  # [T, A], [T, 1]
            dist = Categorical(action_probs)
            new_logprobs = dist.log_prob(actions)  # [T]
            entropy = dist.entropy().mean()  # scalar

            # Clipped surrogate objective (actor)
            ratio = torch.exp(new_logprobs - old_logprobs)
            surr1 = ratio * advantages
            surr2 = (
                torch.clamp(ratio, 1.0 - cfg.eps_clip, 1.0 + cfg.eps_clip) * advantages
            )
            actor_loss = -torch.min(surr1, surr2).mean()

            # MSE value loss (critic)
            critic_loss = nn.functional.mse_loss(state_values.squeeze(-1), returns)

            # Combined loss — single backward pass, no retain_graph needed.
            # The entropy bonus discourages premature convergence to a deterministic policy.
            loss = (
                actor_loss + cfg.value_coef * critic_loss - cfg.entropy_coef * entropy
            )

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Clip gradients to prevent exploding gradients over k_epochs
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()

            epoch_losses.append(loss.item())
            epoch_values.append(state_values.squeeze(-1).mean().item())

        # Record for log_episode (inherited from ActionValueEstimator)
        self.losses.append(float(np.mean(epoch_losses)))
        self.mean_q_values.append(float(np.mean(epoch_values)))

        self._reset_rollout()

    # ── Persistence ───────────────────────────────────────────────────────────

    def save_state(self, dir: str) -> None:
        Path(dir).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "actor": self.model.actor.state_dict(),
                "critic": self.model.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            Path(dir) / "ppo.pth",
        )

    def load_state(self, dir: str) -> None:
        checkpoint = torch.load(Path(dir) / "ppo.pth", map_location=self.device)
        self.model.actor.load_state_dict(checkpoint["actor"])
        self.model.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
