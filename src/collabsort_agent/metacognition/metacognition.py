"""
Definitions for metacognition algorithms.
"""

from dataclasses import dataclass
from statistics import mean

from torch.utils.tensorboard import SummaryWriter

from collabsort_agent.decision import Config as DecisionConfig
from collabsort_agent.learning import Config as LearningConfig


@dataclass
class Config:
    """Metacognition configuration"""

    # Step size for learning rate adjustment
    alpha_rate: float = 0.05

    # Step size for decision threshold adjustment
    theta_rate: float = 0.05

    # Desired confidence level [0..1].
    # The meaningful range/scale of this target depends on DecisionConfig.confidence_method:
    # - "bayesian": confidence is a calibrated posterior probability,
    #   so 0.5 = chance level and e.g. 0.75 is a reasonable target.
    # - "gap": confidence is an uncalibrated geometric measure whose scale
    #   depends on noise_std/theta; a lower target (e.g. 0.4) is appropriate.
    confidence_target: float = 0.75

    # Exponential moving average decay for smoothing confidence
    ema_decay: float = 0.1


class MetaController:
    """Metacognitive controller for adjusting hyperparameters."""

    def __init__(
        self, config: Config, learning_cfg: LearningConfig, decision_cfg: DecisionConfig
    ) -> None:
        self.config = config
        self.learning_cfg = learning_cfg
        self.decision_cfg = decision_cfg

        self.alpha = learning_cfg.alpha_start
        self.theta = decision_cfg.theta_start
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
        self.theta -= self.config.theta_rate * error
        self.theta = float(
            max(
                self.decision_cfg.theta_min,
                min(self.decision_cfg.theta_max, self.theta),
            )
        )

        # Learning rate: raise when under-confident (more plastic),
        # lower when over-confident (more stable)
        self.alpha += self.config.alpha_rate * (-error)
        self.alpha = float(
            max(
                self.learning_cfg.alpha_min,
                min(self.learning_cfg.alpha_max, self.alpha),
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
