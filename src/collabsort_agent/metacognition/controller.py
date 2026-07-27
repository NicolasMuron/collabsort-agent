"""
Cognitive control algorithms.
"""

from statistics import mean

from torch.utils.tensorboard import SummaryWriter

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
        self.confidence_ema = config.confidence_target  # warm-start at target

        self.confidences: list[float] = []

    def update_hyperparameters(self, confidence: float, reaction_time: float) -> None:
        """Update decision and learning hyperparameters based on decision metrics"""

        # Smooth confidence with EMA to avoid reacting to single-step noise
        self.confidence_ema: float = (
            self.config.ema_decay * self.confidence_ema
            + (1.0 - self.config.ema_decay) * confidence
        )
        self.confidences.append(self.confidence_ema)

        error = self.confidence_ema - self.config.confidence_target

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

    def log_episode(self, logger: SummaryWriter, episode: int) -> None:
        if self.confidences:
            logger.add_scalar(
                tag="metacognition/mean_confidence",
                scalar_value=mean(self.confidences),
                global_step=episode,
            )

            # Reset episode data
            self.confidences.clear()
