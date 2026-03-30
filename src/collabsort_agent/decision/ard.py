"""
Advantage Racing Diffusion definitions.
"""

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from collabsort_agent.decision import Config as DecisionConfig
from collabsort_agent.decision import Deliberator
from collabsort_agent.learning import ActionValueEstimator
from collabsort_agent.metacognition import MetaController


def softmax(x: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""

    e = np.exp(x - x.max())
    return e / e.sum()


class AARD(Deliberator):
    def __init__(
        self,
        config: DecisionConfig,
        estimator: ActionValueEstimator,
        meta_ctrl: MetaController,
    ) -> None:
        super().__init__(config=config, estimator=estimator)

        self.meta_ctrl = meta_ctrl

    def choose_action(
        self,
        state: np.ndarray,
        training_step: int,
        rng: np.random.Generator,
    ) -> int:
        """Choose the action to perform"""

        action_values = self.estimator.get_action_values(state=state)

        # Advantage drift rates: each accumulator drifts at Q_i - mean(Q)
        drifts = action_values - action_values.mean()

        # Win probabilities ~ softmax(drift / sigma²)
        # (exact for a race of Wiener processes started at 0, absorbing at theta)
        logits = drifts / (self.config.noise_std**2)
        win_probs = softmax(logits)

        # Expected RT proxy ~ theta / sum(|drifts|)  (larger drifts → faster)
        total_drift = np.abs(drifts).sum() + 1e-8
        expected_rt = self.meta_ctrl.decision_threshold / total_drift

        # Action selection
        # action = int(np.argmax(win_probs))
        action = int(rng.choice(len(win_probs), p=win_probs))

        # Confidence = gap between the two highest win probabilities
        sorted_probs = np.sort(win_probs)[::-1]
        confidence = (
            float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0
        )

        self.meta_ctrl.update_hyperparameters(
            confidence=confidence, reaction_time=expected_rt
        )

        return action

    def log_episode(self, logger: SummaryWriter, episode: int) -> None:
        """Log information after an episode"""

        logger.add_scalar(
            tag="decision/accumulation_threshold",
            scalar_value=self.meta_ctrl.decision_threshold,
            global_step=episode,
        )
        self.meta_ctrl.log_episode(logger=logger, episode=episode)

    def save_state(self, dir: str) -> None:
        """Save the deliberator state to disk"""

        # TODO save state for ARD

    def load_state(self, dir: str) -> None:
        """Load a previously saved deliberator state from disk"""

        # TODO load state for ARD
