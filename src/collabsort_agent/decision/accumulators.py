"""
Accumulators-related algorithms.
"""

import itertools

import numpy as np


class Accumulators:
    """Accumulators data"""

    def __init__(self, n_actions: int) -> None:
        # Ordered list of (i,j) pairs of actions for 0 <= i,j < n_actions and i != j.
        # Length = n_actions(n_actions - 1) = n_accumulators.
        # Example for n_actions = 3: (0,1),(0,2),(1,0),(1,2),(2,0),(2,1).
        self.action_pairs: list[tuple[int, int]] = list(
            itertools.permutations(range(n_actions), 2)
        )

        # Dictionary of advantage ("pro-action") accumulators indexes for each action.
        # Example for n_actions = 3: {0:[0,1], 1:[2,3], 2:[4,5]}.
        # These indexes are used to access individual accumulators in self._evidence.
        self.adv_accs: dict[int, list[int]] = {i: [] for i in range(n_actions)}
        for k, (i, _) in enumerate(self.action_pairs):
            self.adv_accs[i].append(k)

        # Cumulated evidence for each accumulator during a decision. Shape: (n_accumulators,)
        self.evidence = self.empty_evidence()

        # History matrix of evidence values at each time step of a decision. Used for plotting/debugging.
        # Each line stores all successive evidence values for one accumulator.
        # Shape: (n_accumulators, n_decision_steps).
        # Initialized as a 1D array. Each decision step will add a new column.
        self.evidence_history = self.empty_evidence()

        # Drift rates for the accumulators of the current/last decision. Shape: (n_accumulators,)
        # Stored so that _compute_confidence can access the drift rate of any
        # specific accumulator (needed for the Bayesian confidence estimate).
        self.drift_rates = self.empty_evidence()

    @property
    def n_accumulators(self) -> int:
        """Return the number of accumulators = n_actions(n_actions - 1)"""

        return len(self.action_pairs)

    def empty_evidence(self) -> np.ndarray:
        """Return empty evidence array"""

        return np.zeros((self.n_accumulators,), dtype=float)

    def min_evidence(self, actions: list[int]) -> list[float]:
        """Return the minimum value of all advantage accumulators for a list of actions"""

        return [np.min(self.evidence[self.adv_accs[action]]) for action in actions]

    def argmin_accumulator(self, action: int) -> int:
        """
        Return the global index (into self._evidence / self._drift_rates) of
        the "slowest" advantage accumulator for a given action, i.e. the one
        with the lowest evidence value. This is the accumulator that
        determines when/whether the action wins the race (win-all rule).
        """

        accs = self.adv_accs[action]
        local_argmin = int(np.argmin(self.evidence[accs]))
        return accs[local_argmin]
