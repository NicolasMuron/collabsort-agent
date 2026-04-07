"""
Definitions for decision rules.
"""

from abc import ABC, abstractmethod

import numpy as np


class DecisionRule(ABC):
    """Base class for decision rules."""

    def __init__(
        self,
        rng: np.random.Generator,
    ) -> None:
        self.rng = rng

    @abstractmethod
    def get_winning_actions(
        self, n_actions: int, evidence: np.ndarray, theta: float, adv_accs: dict
    ) -> list[int]:
        """Return the winning action(s) during the accumulation process"""


class WinAllRule(DecisionRule):
    """Win-all rule: action i wins when all its advantage accumulators ≥ θ."""

    def get_winning_actions(
        self, n_actions: int, evidence: np.ndarray, theta: float, adv_accs: dict
    ) -> list[int]:
        crossed = evidence >= theta
        return [
            action
            for action in range(n_actions)
            if all(crossed[k] for k in adv_accs[action])
        ]
