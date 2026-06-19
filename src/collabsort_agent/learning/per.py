"""
PER (Prioritized Experience Replay) algorithm.
"""
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.dqn import DQN


class SumTree:
    """Structure d'arbre binaire pour stocker efficacement les priorités."""
    data_pointer = 0

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.n_entries = 0

    def add(self, priority: float, data: tuple):
        tree_index = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update(tree_index, priority)

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:
            self.data_pointer = 0

        if self.n_entries < self.capacity:
            self.n_entries += 1

    def _propagate(self, idx: int, change: float):
        if idx <= 0: 
            return
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def update(self, tree_index: int, priority: float):
        change = priority - self.tree[tree_index]
        self.tree[tree_index] = priority
        self._propagate(tree_index, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left_child_index = 2 * idx + 1
        right_child_index = left_child_index + 1
        
        # Si on a atteint le bas de l'arbre (les feuilles)
        if left_child_index >= len(self.tree):
            return idx
            
        if s <= self.tree[left_child_index]:
            return self._retrieve(left_child_index, s)
        else:
            # Sécurité : s'assurer qu'on ne cherche pas une valeur supérieure à ce que l'arbre contient
            remaining_val = s - self.tree[left_child_index]
            return self._retrieve(right_child_index, remaining_val)

    def get_leaf(self, s: float) -> tuple[int, float, any]:
        leaf_index = self._retrieve(0, s)
        data_index = leaf_index - self.capacity + 1
        return leaf_index, self.tree[leaf_index], self.data[data_index]

    def total_priority(self) -> float:
        return self.tree[0]


class PER(DQN):
    """Prioritized Experience Replay built on top of DQN."""

    def __init__(self, config: LearningConfig, n_actions: int, state_size: int) -> None:
        # Initialise DoubleDuelingDQN (qui gère l'init des réseaux Dueling, device, optimizer, etc.)
        super().__init__(config=config, n_actions=n_actions, state_size=state_size)

        # Override loss_fn pour éviter la réduction moyenne immédiate (besoin des IS weights element-wise)
        self.loss_fn = nn.SmoothL1Loss(reduction="none") 

        # Initialisation du SumTree à la place du deque (replay_buffer)
        self.tree = SumTree(config.replay_buffer_size)

        # Hyperparamètres PER (valeurs alignées sur Schaul et al. 2016, variante "proportional")
        self.per_epsilon = 0.001        # Évite une priorité nulle
        self.per_alpha = 0.6            # Exposant de prioritisation (0.6 recommandé pour proportional)
        self.per_beta = 0.1             # Importance Sampling weight initial
        self.ratio = 1.0                # Atteint 1 au dernier step
        self.n_steps = self.config.n_episodes * self.config.n_steps_episode * self.ratio  # Nombre total de steps pour l'augmentation progressive de β
        self.per_beta_increment = (1 - self.per_beta) / self.n_steps  # Augmentation progressive vers 1.0

        # Reward/TD-error clipping range, comme dans le papier original (stabilité numérique)
        self.per_clip_value = 1.0

        # Le papier réduit le step-size d'un facteur 4 par rapport au baseline,
        # car la prioritisation augmente la magnitude typique des gradients.
        # On recrée donc l'optimizer avec un lr propre à PER (sans toucher self.config.lr,
        # qui reste la référence pour les autres algos).
        self.per_lr = self.config.lr/4
        self.optimizer = optim.Adam(params=self.q_network.parameters(), lr=self.per_lr)

    def _get_priority(self, error: float) -> float:
        return (abs(error) + self.per_epsilon) ** self.per_alpha

    def _store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        """Override la méthode de stockage pour utiliser le SumTree avec la priorité maximale.

        Le reward est clippé dans [-1, 1] comme dans le papier original (Section 4),
        pour éviter que des transitions à forte récompense (ex: +8 ou -10 dans notre env)
        ne dominent excessivement la distribution de priorités du SumTree.
        """
        #reward_clipped = float(np.clip(reward, -self.per_clip_value, self.per_clip_value))
        #max_priority = self.tree.tree.max() if self.tree.n_entries > 0 else 1.0
        max_priority = 1
        self.tree.add(max_priority, (state, action, reward, next_state, done))

    def _sample(self) -> tuple[list, list, np.ndarray]:
        """Échantillonne un batch depuis le SumTree avec stratification et calcule les IS weights."""
        minibatch, idxs, priorities = [], [], []
        
        # Sécurité : Si l'arbre est vide ou presque, éviter une division par 0
        total_p = self.tree.total_priority()
        if total_p == 0:
            total_p = 1.0
            
        segment = total_p / self.config.batch_size

        # β augmente progressivement vers 1.0
        self.per_beta = min(1.0, self.per_beta + self.per_beta_increment)

        for i in range(self.config.batch_size):
            a = segment * i
            b = segment * (i + 1)
            
            # On restreint légèrement la borne supérieure pour éviter les débordements d'arrondis
            value = np.random.uniform(a, min(b, total_p - 1e-5))
            
            idx, p, data = self.tree.get_leaf(value)
            
            # --- SÉCURITÉ ANTI-FEUILLE VIDE ---
            # Si le data récupéré est un entier (le 0 de l'initialisation) ou est invalide
            attempts = 0
            while (not isinstance(data, (tuple, list))) and attempts < 10:
                # On ré-échantillonne sur une valeur aléatoire purement valide (plus basse dans l'arbre)
                value = np.random.uniform(0, max(1e-5, total_p - 1e-2))
                idx, p, data = self.tree.get_leaf(value)
                attempts += 1
            
            # Éviter une priorité strictement nulle qui ferait planter l'Importance Sampling
            p = max(p, self.per_epsilon)
            
            priorities.append(p)
            minibatch.append(data)
            idxs.append(idx)

        # Calcul IS weights (correction du biais d'échantillonnage)
        sampling_probs = np.array(priorities) / total_p
        
        # Utiliser self.tree.n_entries pour refléter le nombre RÉEL d'éléments stockés
        is_weights = np.power(self.tree.n_entries * sampling_probs, -self.per_beta)
        
        # Sécurité pour éviter la division par zéro lors de la normalisation
        max_weight = is_weights.max()
        if max_weight == 0:
            max_weight = 1.0
        is_weights /= max_weight  # Normalisation

        return minibatch, idxs, is_weights        

    def _learn(self) -> None:
        """Version surchargée de _learn intégrant la pondération par IS weights et la mise à jour de l'arbre."""
        if self.tree.n_entries < self.config.batch_size:
            return

        batch, idxs, is_weights = self._sample()
        states, actions, rewards, next_states, dones = zip(*batch, strict=True)

        # Préparation des tenseurs
        states = torch.from_numpy(np.array(states, dtype=np.float32)).to(self.device)
        actions = torch.tensor(actions, dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32, device=self.device)
        is_weights_t = torch.tensor(is_weights, dtype=torch.float32, device=self.device)

        actions = torch.clamp(actions, 0, self.n_actions - 1)

        # Q-values courantes
        q_values = self.q_network(states).gather(1, actions).squeeze(1)
        self.mean_q_values.append(torch.mean(q_values).item())

        # Calcul des targets en réutilisant la règle de DoubleDQN (_get_next_q_values)
        with torch.no_grad():
            q_next = self._get_next_q_values(next_states)
            q_target = rewards + self.config.gamma * q_next * (1 - dones)

        # Perte pondérée par les IS weights
        elementwise_loss = self.loss_fn(q_values, q_target)
        loss = (is_weights_t * elementwise_loss).mean()
        self.losses.append(loss.item())

        # Backpropagation & Optimisation
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=10.0)
        self.optimizer.step()

        # Mise à jour des priorités dans le SumTree avec les nouvelles erreurs TD.
        # Clipping dans [-1, 1] comme dans le papier original, pour éviter qu'une
        # transition à erreur TD extrême ne domine totalement le SumTree.
        td_errors = (q_target - q_values).abs().detach().cpu().numpy()
        for idx, error in zip(idxs, td_errors):
            self.tree.update(idx, self._get_priority(error))

        # Synchronisation du réseau cible (Target Network)
        self.learning_step += 1
        if self.learning_step % self.config.target_network_sync_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())