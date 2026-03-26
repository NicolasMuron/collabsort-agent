"""
Definitions for metacognition algorithms.
"""

from dataclasses import dataclass

from collabsort_agent.decision import Config as DecisionConfig
from collabsort_agent.learning import Config as LearningConfig


@dataclass
class Config:
    """Metacognition configuration"""

    # Step size for learning rate adjustment
    lr_rate: float = 0.05

    # Step size for decision threshold adjustment
    threshold_rate: float = 0.05

    # Desired confidence level [0..1]
    confidence_target: float = 0.4

    # Exponential moving average decay for smoothing confidence
    ema_decay: float = 0.9


class MetaController:
    """Metacognitive controller for adjusting hyperparameters."""

    def __init__(
        self, config: Config, learning_cfg: LearningConfig, decision_cfg: DecisionConfig
    ) -> None:
        self.config = config
        self.learning_cfg = learning_cfg
        self.decision_cfg = decision_cfg

        self.learning_rate = learning_cfg.lr_start
        self.decision_threshold = decision_cfg.threshold_start

    def update_hyperparameters(self, confidence: float, reaction_time: float) -> None:
        """Update decision and learning hyperparameters based on decision metrics"""

        # Smooth confidence with EMA to avoid reacting to single-step noise
        self.confidence_ema: float = (
            self.config.ema_decay * self.confidence_ema
            + (1.0 - self.config.ema_decay) * confidence
        )

        error = self.confidence_ema - self.config.confidence_target

        # Decision threshold: shrink when over-confident (faster decisions),
        # grow when under-confident (more deliberation)
        self.decision_threshold -= self.config.threshold_rate * error
        self.decision_threshold = float(
            max(
                self.decision_cfg.threshold_min,
                min(self.decision_cfg.threshold_max, self.decision_threshold),
            )
        )

        # Learning rate: raise when under-confident (more plastic),
        # lower when over-confident (more stable)
        self.learning_rate += self.config.lr_rate * (-error)
        self.learning_rate = float(
            max(
                self.learning_cfg.lr_min,
                min(self.learning_cfg.lr_max, self.learning_rate),
            )
        )
