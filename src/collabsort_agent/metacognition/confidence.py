"""
Confidence omputation algorithms.
"""

import math
from abc import ABC, abstractmethod

from collabsort_agent.decision.accumulators import Accumulators
from collabsort_agent.decision.decision import DecisionConfig
from collabsort_agent.metacognition import Hyperparameters, MetaConfig


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


class CalibrationMethod(ABC):
    """
    Abstract base class for outcome-based confidence recalibration methods.

    A ConfidenceMethod's output is a claim about the decision process itself
    ("how much should this particular decision be trusted"). A calibration
    method compares that claim, after the fact, to an observed outcome, and
    nudges a bias term so that the confidence value the agent actually acts
    on stays well-calibrated over time:

        confidence_error = outcome - confidence
        bias <- bias + calibration_rate * confidence_error
        calibrated_confidence = clip(confidence + bias, 0, 1)
    """

    def __init__(self, config: MetaConfig) -> None:
        self.config = config

        # Learned calibration bias b_c, added to raw/smoothed confidence
        # before it is reported to the metacognitive controller.
        self.bias: float = 0.0

    @abstractmethod
    def compute_outcome(self, td_error: float) -> float:
        """
        Compute the outcome signal o in {0, 1} used to recalibrate
        confidence, from the reward-prediction error (TD-error) observed
        for the transition that resulted from the decision.
        """

    def calibrate(self, confidence: float) -> float:
        """Apply the current calibration bias to a raw/smoothed confidence value (Eq 6f)."""

        return min(1.0, max(0.0, confidence + self.bias))

    def update(self, confidence: float, outcome: float) -> None:
        """Update the calibration bias from a confidence prediction error (Eq 6d-6e)."""

        confidence_error = outcome - confidence
        self.bias += self.config.calibration_rate * confidence_error


class TDErrorCalibration(CalibrationMethod):
    """
    Outcome-based calibration using the sign of the reward-prediction error
    (TD-error) as the outcome signal:

        o_t = 1[delta_t >= 0]

    i.e. whether the decision's consequence was at least as good as the
    agent's own prior expectation (no negative surprise), where
    delta_t = R_t + gamma * max_a' Q(S', a') - Q(S, A) is the TD-error
    computed by the Learning module for its Q-update.

    This reuses a signal already computed by the Learning module, so it
    requires no extra bookkeeping, and it is available on every transition
    (no sparse-reward gaps), unlike outcome definitions based on raw reward.
    """

    def compute_outcome(self, td_error: float) -> float:
        return 1.0 if td_error >= 0 else 0.0
