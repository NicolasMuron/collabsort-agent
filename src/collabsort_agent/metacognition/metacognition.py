"""
Common definitions for metacognition algorithms.
"""

from dataclasses import dataclass
from typing import Literal

from collabsort_agent.decision.decision import DecisionConfig
from collabsort_agent.learning.learning import LearningConfig


@dataclass
class Config:
    """Metacognition configuration"""

    # Method used to compute decision confidence:
    # - "gap": normalized distance between winner/runner-up slowest accumulators.
    # - "bayesian": posterior probability that the winning action's drift truly exceeds the runner-up's.
    confidence_method: Literal["gap", "bayesian"] = "bayesian"

    # Desired confidence level [0..1].
    # The meaningful range/scale of this parameter depends on confidence_method:
    # - "bayesian": confidence is a calibrated posterior probability,
    #   so 0.5 = chance level and e.g. 0.75 is a reasonable target.
    # - "gap": confidence is an uncalibrated geometric measure whose scale
    #   depends on noise_std/theta; a lower target (e.g. 0.4) is appropriate.
    confidence_target: float = 0.75

    # Exponential moving average decay for smoothing confidence
    ema_decay: float = 0.1

    # Step size for learning rate adjustment
    alpha_rate: float = 0.05

    # Step size for decision threshold adjustment
    theta_rate: float = 0.05


@dataclass
class Hyperparameters:
    """Hyperparameters adjusted via metacognition"""

    def __init__(
        self, decision_cfg: DecisionConfig, learning_cfg: LearningConfig
    ) -> None:

        # Learning rate
        self.alpha: float = learning_cfg.alpha_start

        # Decision threshold
        self.theta: float = decision_cfg.theta_start
