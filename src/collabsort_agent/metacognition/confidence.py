"""
Confidence omputation algorithms.
"""

import math
from abc import ABC, abstractmethod

from collabsort_agent.decision.accumulators import Accumulators
from collabsort_agent.decision.decision import Config as DecisionConfig
from collabsort_agent.metacognition import Hyperparameters


class ConfidenceMethod(ABC):
    """Abstract base class for confidence computation methods"""

    def __init__(
        self, decision_cfg: DecisionConfig, hyperparameters: Hyperparameters
    ) -> None:
        self.decision_cfg = decision_cfg
        self.hyperparameters = hyperparameters

    @abstractmethod
    def compute_decision_confidence(
        self,
        chosen_action: int,
        runnerup_action: int,
        reaction_time: float,
        accumulators: Accumulators,
    ) -> float:
        """Compute decision confidence"""


class GapConfidence(ConfidenceMethod):
    def compute_decision_confidence(
        self,
        chosen_action: int,
        runnerup_action: int,
        reaction_time: float,
        accumulators: Accumulators,
    ) -> float:
        """Compute decision confidence using normalized distance between winner/runner-up slowest accumulators"""

        min_evidence = accumulators.min_evidence(
            actions=[chosen_action, runnerup_action]
        )
        return (min_evidence[0] - min_evidence[1]) / (self.hyperparameters.theta + 1e-6)


class BayesianConfidence(ConfidenceMethod):
    def compute_decision_confidence(
        self,
        chosen_action: int,
        runnerup_action: int,
        reaction_time: float,
        accumulators: Accumulators,
    ) -> float:
        """
        Compute decision confidence using posterior probability that the winning action's drift rate truly exceeds the runner-up's.
        Inspired by (Kepecs2008).

        Each accumulator x_k follows dx_k = v_k*dt + noise, noise ~ N(0, noise_std^2)
        per simulation step (see choose_action). After `reaction_time` steps:
            x_k ~ N(v_k * reaction_time * dt, reaction_time * noise_std^2)
        The difference between the chosen and runner-up slowest accumulators
        is then approximately Gaussian, and confidence is the posterior
        probability that this difference has positive mean, i.e. that the
        winning accumulator's drift genuinely exceeded the runner-up's:

            c = Phi( (v_i - v_j) * t* / (sigma * sqrt(2*t*)) )

        with t* = reaction_time * dt and sigma^2 = noise_std^2 / dt (the
        per-unit-time noise variance implied by the discretization above).
        """

        chosen_idx = accumulators.argmin_accumulator(action=chosen_action)
        runnerup_idx = accumulators.argmin_accumulator(action=runnerup_action)

        v_diff = (
            accumulators.drift_rates[chosen_idx]
            - accumulators.drift_rates[runnerup_idx]
        )

        # Elapsed simulated time and standard deviation of the accumulator
        # difference, matching the discretization used in choose_action
        # (noise variance accrues per step, drift accrues per unit of dt).
        elapsed_time = reaction_time * self.decision_cfg.dt
        std_diff = self.decision_cfg.noise_std * math.sqrt(2.0 * reaction_time)

        z = (v_diff * elapsed_time) / (std_diff + 1e-12)
        return self._normal_cdf(z)

    @staticmethod
    def _normal_cdf(z: float) -> float:
        """Standard normal CDF, computed via the error function (no scipy dependency)."""

        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
