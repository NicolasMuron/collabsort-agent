"""
Cognitive control algorithms.
"""

from collabsort_agent.decision import DecisionConfig
from collabsort_agent.learning import LearningConfig
from collabsort_agent.metacognition import Hyperparameters, MetaConfig


class MetaController:
    """Metacognitive controller for adjusting hyperparameters."""

    def __init__(
        self,
        config: MetaConfig,
        learning_cfg: LearningConfig,
        decision_cfg: DecisionConfig,
        hyperparameters: Hyperparameters,
    ) -> None:
        self.config = config
        self.learning_cfg = learning_cfg
        self.decision_cfg = decision_cfg

        self.hyperparameters = hyperparameters

    def update_hyperparameters(self, confidence: float, reaction_time: float) -> None:
        """Update decision and learning hyperparameters based on decision metrics"""

        error = confidence - self.config.confidence_target

        # Decision threshold: shrink when over-confident (faster decisions),
        # grow when under-confident (more deliberation)
        self.hyperparameters.theta -= self.config.theta_rate * error
        self.hyperparameters.theta = float(
            max(
                self.decision_cfg.theta_min,
                min(self.decision_cfg.theta_max, self.hyperparameters.theta),
            )
        )

        # Learning rate: raise when under-confident (more plastic),
        # lower when over-confident (more stable)
        self.hyperparameters.alpha += self.config.alpha_rate * (-error)
        self.hyperparameters.alpha = float(
            max(
                self.learning_cfg.alpha_min,
                min(self.learning_cfg.alpha_max, self.hyperparameters.alpha),
            )
        )
