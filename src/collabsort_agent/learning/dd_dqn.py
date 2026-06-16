"""
Double Dueling DQN algorithm
"""
from collabsort_agent.learning.double_dqn import DoubleDQN
from collabsort_agent.learning.dueling_dqn import DuelingDQN

class DoubleDuelingDQN(DoubleDQN, DuelingDQN):
    """
    Double Dueling DQN algorithm implementation.
    Inherits the DoubleDQN calculation rule and the DuelingDQN network.
    """
    pass