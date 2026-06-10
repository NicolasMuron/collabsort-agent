"""
Dueling DQN algorithm
"""

import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collabsort_agent.learning import ActionValueEstimator
from collabsort_agent.learning import Config as LearningConfig

def get_device() -> torch.device:
    """Return accelerated device if available, or fall back to CPU"""

    return torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    
class dueling_Network(nn.Module):
    """Dueling DQN architecture"""

    def __init__(self, state_size: int, action_size: int) -> None:
        super(dueling_Network, self).__init__()
        self.state_size = state_size
        self.action_size = action_size

        # Common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, 1)  # Output is a single value for the state
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(100, 50),
            nn.Tanh(),
            nn.Linear(50, action_size)  # Output is an advantage for each action
        )

    def forward(self, state: torch.Tensor, return_components: bool = False) -> torch.Tensor:
        features = self.feature_layer(state)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Combine value and advantages to get Q-values
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True)[0])
        if return_components:
            return q_values, value, advantages
        return q_values
    
class DuelingDQN(ActionValueEstimator):
    """Dueling DQN algorithm implementation"""
    
    def __init__(self, config: LearningConfig, n_actions: int, state_size: int) -> None:
        
        super().__init__(config=config, n_actions=n_actions)
        
        self.device = get_device()
        
        # Create duelingQ-network for estimating action values from dueling Network
        self.q_network = dueling_Network(state_size=state_size, action_size=n_actions).to(
            self.device
        )
        # Use SmoothL1Loss (Huber) rather than MSELoss.
        self.loss_fn = nn.SmoothL1Loss()
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.config.lr)
    
        # Create target network with fixed parameters (stabilizes training)
        self.target_network = dueling_Network(state_size=state_size, action_size=n_actions).to(
            self.device
        )
        self.target_network.load_state_dict(self.q_network.state_dict())
        # Target network must not accumulate gradients
        self.target_network.eval()
        
        # Create replay buffer for training the Q-network
        self.replay_buffer: deque = deque(maxlen=self.config.replay_buffer_size)

        # Step counter used to decide when to sync the target network
        self.learning_step: int = 0
        
        # --- NOUVELLES LISTES POUR L'ANALYSE DUELING ---
        self.low_v_gaps: list[float] = []
        self.high_v_gaps: list[float] = []
        
    
    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)     
        return q_values[0].detach().cpu().numpy()
    
    def update_action_values(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool) -> None:
        
        self._store_transition(
            state=state, action=action, reward=reward, next_state=next_state, done=done
        )
        self._learn()
        
    def _store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        """Store transition between states for future learning."""

        # Store transition in replay buffer
        self.replay_buffer.append((state, action, reward, next_state, done))

    def _learn(self) -> None:
        """Update the Q-network parameters."""
       
        if len(self.replay_buffer) < self.config.batch_size:
            return

        # Sample a batch of past experiences from replay buffer
        batch = random.sample(self.replay_buffer, self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch, strict=True) 
        
        # Obtain PyTorch tensors from NumPy arrays.
        # torch.from_numpy avoids allocating new memory
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(
            1
        )
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(
            self.device
        )
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        
        # Compute action values for the current states
        q_values_all, values, advantages = self.q_network(states, return_components=True)
        
        q_values = q_values_all.gather(1, actions).squeeze(1)
        self.mean_q_values.append(torch.mean(q_values).item())
        
        with torch.no_grad():
            # Création des masques Booléens (taille du batch)
            # v_values a la forme (batch_size, 1), on le squeeze pour avoir (batch_size,)
            v_s = values.squeeze(1)
    
            # 1. On calcule la moyenne et l'écart-type du batch actuel
            batch_mean_v = v_s.mean()
            batch_std_v = v_s.std()
            
            # On ajoute une sécurité (1e-8) pour éviter une division par zéro si le batch est uniforme
            if batch_std_v > 1e-8:
                # On centre et on réduit : v_normalized aura une moyenne de 0 et un std de 1
                v_normalized = (v_s - batch_mean_v) / batch_std_v
            else:
                # Si le batch est plat, on se contente de centrer
                v_normalized = v_s - batch_mean_v

            # 2. Les masques se basent maintenant sur cette distribution parfaitement équitable
            # 0.0 correspond exactement à la moyenne parfaite de ce batch standardisé
            low_v_mask = (v_normalized < 0.0)
            high_v_mask = (v_normalized >= 0.0)
            
            if low_v_mask.any():
                adv_low = advantages[low_v_mask]  # (n_states, n_actions)
                top2 = torch.topk(adv_low, k=2, dim=1).values
                gap = (top2[:, 0] - top2[:, 1]).mean().item()
                self.low_v_gaps.append(gap)
                    
            if high_v_mask.any():
                adv_high = advantages[high_v_mask]
                top2 = torch.topk(adv_high, k=2, dim=1).values
                gap = (top2[:, 0] - top2[:, 1]).mean().item()
                self.high_v_gaps.append(gap)
        
        # Using target_network (not q_network) to compute Q-targets.
        with torch.no_grad():
            q_next = self.target_network(next_states).max(1)[0]
            # Q_target = r + gamma * max_a' Q_target(s', a') * (1 - done)
            q_target = rewards + self.config.gamma * q_next * (1 - dones)

        loss = self.loss_fn(q_values, q_target)
        self.losses.append(loss.item())

        # Update Q-network parameters through a gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent exploding gradients.
        # max_norm=10 is a common conservative bound for DQN.
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)

        self.optimizer.step()

        # Periodically sync the target network with the online network.
        self.learning_step += 1
        if self.learning_step % self.config.target_network_sync_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
            
    def save_state(self, dir: str) -> None:
        Path(dir).mkdir(parents=True, exist_ok=True)
        file_path = f"{dir}/{self.state_filename}"
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            },
            file_path,
        )

    def load_state(self, dir: str) -> None:
        file_path = f"{dir}/{self.state_filename}"
        checkpoint = torch.load(file_path, map_location=self.device)

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        # Target network is only used for inference during target computation.
        self.target_network.eval()
    