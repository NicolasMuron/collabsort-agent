"""
Metacognitive monitoring algorithms.
"""

from statistics import mean

from torch.utils.tensorboard import SummaryWriter

from collabsort_agent.decision.accumulators import Accumulators
from collabsort_agent.metacognition import MetaConfig
from collabsort_agent.metacognition.confidence import (
    CalibrationMethod,
    ConfidenceMethod,
)


class MetaMonitoring:
    """Metacognitive monitoring"""

    def __init__(
        self,
        config: MetaConfig,
        confidence_method: ConfidenceMethod,
        calibration_method: CalibrationMethod | None = None,
    ) -> None:

        self.config = config

        # Method for computing decision confidence
        self.confidence_method = confidence_method

        # Method for outcome-based confidence recalibration. Optional: if
        # None, confidence is reported as-is (no recalibration).
        self.calibration_method = calibration_method

        # Last smoothed confidence value
        self.smoothed_confidence = config.confidence_target  # warm-start at target

        # List of confidence values for this episode. Used for logging.
        self.confidences: list[float] = []

        # List of calibration bias values for this episode. Used for logging.
        self.calibration_biases: list[float] = []

        # Confidence value most recently reported to the metacognitive
        # controller, cached so that update_calibration() can compute a
        # confidence prediction error once the outcome for that specific
        # decision becomes available (one environment step later).
        self._last_confidence: float | None = None

    def compute_decision_confidence(
        self,
        chosen_action: int,
        runnerup_action: int,
        reaction_time: float,
        accumulators: Accumulators,
    ) -> float:
        """Compute confidence associated with a decision, smoothed via EMA and calibrated"""

        confidence = self.confidence_method.compute_decision_confidence(
            chosen_action=chosen_action,
            runnerup_action=runnerup_action,
            reaction_time=reaction_time,
            accumulators=accumulators,
        )

        # Smooth confidence with EMA to avoid reacting to single-step noise
        self.smoothed_confidence = (
            self.config.ema_decay * self.smoothed_confidence
            + (1.0 - self.config.ema_decay) * confidence
        )
        confidence = self.smoothed_confidence

        # Apply outcome-based calibration bias, if configured
        if self.calibration_method is not None:
            confidence = self.calibration_method.calibrate(confidence=confidence)

        self.confidences.append(confidence)
        self._last_confidence = confidence

        return confidence

    def update_calibration(self, td_error: float) -> None:
        """
        Recalibrate confidence using the outcome of the transition that
        resulted from the most recently reported decision confidence
        (Eq 6d-6e). No-op if no calibration method is configured, or if no
        confidence has been reported yet.
        """

        if self.calibration_method is None or self._last_confidence is None:
            return

        outcome = self.calibration_method.compute_outcome(td_error=td_error)
        self.calibration_method.update(
            confidence=self._last_confidence, outcome=outcome
        )
        self.calibration_biases.append(self.calibration_method.bias)

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

        if self.calibration_biases:
            logger.add_scalar(
                tag="metacognition/calibration_bias",
                scalar_value=mean(self.calibration_biases),
                global_step=episode,
            )

            # Reset episode data
            self.calibration_biases.clear()
