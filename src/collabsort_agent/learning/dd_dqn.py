"""
Double Dueling DQN algorithm
"""
<<<<<<< HEAD
from collabsort_agent.learning.double_dqn import DoubleDQN
from collabsort_agent.learning.dueling_dqn import Dueling_Network

=======

from collabsort_agent.learning.double_dqn import DoubleDQN
from collabsort_agent.learning.dueling_dqn import Dueling_Network


>>>>>>> main
class DoubleDuelingDQN(DoubleDQN):
    """
    Double Dueling DQN algorithm implementation.
    Inherits the DoubleDQN calculation rule and overrides the network with Dueling_Network.
    """
<<<<<<< HEAD
    def build_network(self):
        """Inject the Dueling architecture."""
        return Dueling_Network(state_size=self.state_size, action_size=self.n_actions)
=======

    def build_network(self):
        """Inject the Dueling architecture."""
        return Dueling_Network(state_size=self.state_size, action_size=self.n_actions)
>>>>>>> main
