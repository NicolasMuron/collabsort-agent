"""
Noisy Networks for Exploration extension for DQN.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from collabsort_agent.learning.dqn import DQN

class NoisyLinear(nn.Module):
    """
    Couche linéaire bruitée (Noisy Linear Layer).
    Génère son propre bruit interne pour l'exploration autonome.
    """
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # Paramètres entraînés (Moyenne mu et Écart-type sigma)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        # Buffers PyTorch pour stocker les matrices de bruit aléatoire (non entraînés)
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Initialisation des poids selon Fortunato et al. (2017)."""
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size: int) -> torch.Tensor:
        """Génère un bruit gaussien factorisé."""
        x = torch.randn(size, device=self.weight_mu.device)
        return x.sign().mul(x.abs().sqrt())

    def reset_noise(self):
        """Ré-échantillonne de nouvelles matrices de bruit pour l'étape actuelle."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        
        # Produit extérieur pour le bruit des poids, vecteur pour le biais
        self.weight_epsilon.copy_(torch.outer(epsilon_out, epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # En mode entraînement, on applique le bruit appris
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            # En mode évaluation (test), le réseau devient déterministe (sans bruit)
            weight = self.weight_mu
            bias = self.bias_mu
            
        return F.linear(x, weight, bias)


class NoisyQNetwork(nn.Module):
    """Réseau de neurones Q-Network utilisant des couches NoisyLinear."""
    def __init__(self, input_size: int, output_size: int, hidden_sizes: tuple = (100, 100)):
        super().__init__()
        # La première couche reste classique pour extraire les features de l'état
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        
        # Les couches suivantes sont bruitées pour propager l'exploration en profondeur
        self.noisy1 = NoisyLinear(hidden_sizes[0], hidden_sizes[1])
        self.noisy2 = NoisyLinear(hidden_sizes[1], output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.noisy1(x))
        return self.noisy2(x)

    def reset_noise(self):
        """Demande aux couches bruitées de générer un nouveau bruit."""
        self.noisy1.reset_noise()
        self.noisy2.reset_noise()


class NoisyDQN(DQN):
    """Agent DQN qui utilise Noisy Networks au lieu d'Epsilon-Greedy."""
    def __init__(self, config, state_size, n_actions):
        super().__init__(config, state_size, n_actions)
        
        # On surcharge les réseaux de la classe mère DQN avec nos réseaux bruités
        self.q_network = NoisyQNetwork(state_size, n_actions).to(self.device)
        self.target_network = NoisyQNetwork(state_size, n_actions).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

    def _optimize_network(self, loss) -> None:
        # 1. On laisse DQN faire la rétropropagation et la mise à jour des gradients
        super()._optimize_network(loss)
        
        # 2. Règle absolue de NoisyNets : on réinitialise le bruit après CHAQUE étape de gradient
        self.q_network.reset_noise()
        self.target_network.reset_noise()