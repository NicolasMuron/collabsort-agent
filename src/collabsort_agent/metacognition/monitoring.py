"""
Metacognitive monitoring algorithms.
"""

from statistics import mean

from torch.utils.tensorboard import SummaryWriter

from collabsort_agent.decision.accumulators import Accumulators
from collabsort_agent.metacognition import MetaConfig
from collabsort_agent.metacognition.confidence import ConfidenceMethod


class MetaMonitoring:
    """Metacognitive monitoring"""

    def __init__(self, config: MetaConfig, confidence_method: ConfidenceMethod) -> None:

        self.config = config

        # Method for computing decision confidence
        self.confidence_method = confidence_method

        # Last smoothed confidence value
        self.confidence_ema = config.confidence_target  # warm-start at target

        # List of confidence values for this episode. Used for logging.
        self.confidences: list[float] = []

    def compute_decision_confidence(
        self,
        chosen_action: int,
        runnerup_action: int,
        reaction_time: float,
        accumulators: Accumulators,
    ) -> float:
        """Compute confidence associated with a decision, smoothed via EMA"""

        confidence = self.confidence_method.compute_decision_confidence(
            chosen_action=chosen_action,
            runnerup_action=runnerup_action,
            reaction_time=reaction_time,
            accumulators=accumulators,
        )

        # Smooth confidence with EMA to avoid reacting to single-step noise
        self.confidence_ema = (
            self.config.ema_decay * self.confidence_ema
            + (1.0 - self.config.ema_decay) * confidence
        )
        self.confidences.append(self.confidence_ema)

        return self.confidence_ema

    def log_episode(self, logger: SummaryWriter, episode: int) -> None:
        """Log information after an episode"""

        if self.confidences:
            logger.add_scalar(
                tag="metacognition/mean_confidence",
                scalar_value=mean(self.confidences),
                global_step=episode,
            )

            # Reset episode data
            self.confidences.clear()
