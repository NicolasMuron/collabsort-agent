"""
Unit tests for metacognition algorithms.
"""

from typing import cast

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from collabsort_agent.decision import DecisionConfig
from collabsort_agent.decision.accumulators import Accumulators
from collabsort_agent.learning import LearningConfig
from collabsort_agent.metacognition import Hyperparameters, MetaConfig
from collabsort_agent.metacognition.confidence import BayesianConfidence, GapConfidence
from collabsort_agent.metacognition.controller import MetaController


class LoggerStub:
    """Minimal SummaryWriter stub that records add_scalar() calls instead of writing to disk"""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []

    def add_scalar(self, tag: str, scalar_value: float, global_step: int) -> None:
        self.scalars.append((tag, scalar_value, global_step))

    def as_summary_writer(self) -> SummaryWriter:
        """Present this stub as a SummaryWriter for type-checking purposes"""

        return cast(SummaryWriter, self)


def _make_accumulators(
    drift_rates: list[float], evidence: list[float] | None = None
) -> Accumulators:
    """
    Build a 2-action Accumulators instance (one advantage accumulator per
    action) with fixed drift rates/evidence, for direct testing of confidence
    computation without running a full evidence-accumulation race.
    """

    accumulators = Accumulators(n_actions=2)
    accumulators.drift_rates = np.array(drift_rates)
    if evidence is not None:
        accumulators.evidence = np.array(evidence)
    return accumulators


class TestGapConfidence:
    def test_confidence_gap(self) -> None:
        """Gap confidence is the normalized distance between the winner/runner-up accumulators"""

        decision_cfg = DecisionConfig(theta_start=1.0)
        hyperparameters = Hyperparameters(
            decision_cfg=decision_cfg, learning_cfg=LearningConfig()
        )
        gap = GapConfidence(decision_cfg=decision_cfg, hyperparameters=hyperparameters)

        # Action 1 (chosen) has evidence 1.0, action 0 (runner-up) has evidence 0.7
        accumulators = _make_accumulators(drift_rates=[0.0, 0.0], evidence=[0.7, 1.0])

        confidence = gap.compute_decision_confidence(
            chosen_action=1,
            runnerup_action=0,
            reaction_time=10.0,
            accumulators=accumulators,
        )

        assert abs(confidence - 0.3) < 1e-4

    def test_confidence_gap_scales_with_theta(self) -> None:
        """A higher decision threshold yields a lower confidence for the same evidence gap"""

        decision_cfg = DecisionConfig(theta_start=2.0)
        hyperparameters = Hyperparameters(
            decision_cfg=decision_cfg, learning_cfg=LearningConfig()
        )
        gap = GapConfidence(decision_cfg=decision_cfg, hyperparameters=hyperparameters)

        accumulators = _make_accumulators(drift_rates=[0.0, 0.0], evidence=[0.7, 1.0])

        confidence = gap.compute_decision_confidence(
            chosen_action=1,
            runnerup_action=0,
            reaction_time=10.0,
            accumulators=accumulators,
        )

        assert abs(confidence - 0.15) < 1e-4


class TestBayesianConfidence:
    @staticmethod
    def _drift_rate_diff(decision_cfg: DecisionConfig, q_diff: float) -> float:
        """
        Drift-rate difference between the winning and runner-up advantage
        accumulators of a 2-action decision, per v(i,j) = w_d*(Qi-Qj) +
        w_s*(Qi+Qj) + V0. The w_s*(Qi+Qj) and V0 terms are identical for both
        accumulators of a 2-action decision and cancel out in the difference.
        """

        return 2.0 * decision_cfg.w_d * q_diff

    def test_confidence_bayesian_monotonic_with_clarity(self) -> None:
        """
        Bayesian (signal-detection) confidence is a genuine posterior
        probability in [0, 1] and should increase monotonically with the
        clarity of the winning action's advantage over the runner-up.
        """

        decision_cfg = DecisionConfig()
        hyperparameters = Hyperparameters(
            decision_cfg=decision_cfg, learning_cfg=LearningConfig()
        )
        bayesian = BayesianConfidence(
            decision_cfg=decision_cfg, hyperparameters=hyperparameters
        )

        reaction_time = 30.0
        # Q1 - Q0, ordered from most to least clear-cut winner
        q_diffs = [1.0, 0.5, 0.1]

        confidences = []
        for q_diff in q_diffs:
            v_diff = self._drift_rate_diff(decision_cfg, q_diff)
            accumulators = _make_accumulators(drift_rates=[-v_diff / 2, v_diff / 2])

            confidence = bayesian.compute_decision_confidence(
                chosen_action=1,
                runnerup_action=0,
                reaction_time=reaction_time,
                accumulators=accumulators,
            )

            # Always a valid probability
            assert 0.0 <= confidence <= 1.0
            confidences.append(confidence)

        # Confidence should decrease as the decision gets less clear-cut
        assert confidences[0] > confidences[1] > confidences[2]

        # A near-unambiguous decision (Q1 - Q0 = 1.0) should be near-certain
        assert confidences[0] > 0.95

        # The most ambiguous case here (Q1 - Q0 = 0.1) should still reflect a
        # real, if more modest, edge over chance level (0.5)
        assert 0.5 < confidences[2] < 0.95

    def test_confidence_bayesian_symmetric_ties_are_chance_level(self) -> None:
        """
        With identical drift rates for the winning and runner-up
        accumulators (no true evidence favoring either), confidence should
        sit at chance level (0.5).
        """

        decision_cfg = DecisionConfig()
        hyperparameters = Hyperparameters(
            decision_cfg=decision_cfg, learning_cfg=LearningConfig()
        )
        bayesian = BayesianConfidence(
            decision_cfg=decision_cfg, hyperparameters=hyperparameters
        )

        accumulators = _make_accumulators(drift_rates=[0.15, 0.15])

        confidence = bayesian.compute_decision_confidence(
            chosen_action=1,
            runnerup_action=0,
            reaction_time=30.0,
            accumulators=accumulators,
        )

        assert abs(confidence - 0.5) < 1e-6


class TestMetaController:
    def test_confidence_ema_warm_starts_at_target(self) -> None:
        """The confidence EMA should warm-start at the configured target"""

        meta_ctrl = self._make_controller(config=MetaConfig(confidence_target=0.6))

        assert meta_ctrl.confidence_ema == 0.6

    def test_update_hyperparameters_smooths_confidence_with_ema(self) -> None:
        """Confidence should be smoothed via EMA rather than used raw"""

        config = MetaConfig(confidence_target=0.75, ema_decay=0.5)
        meta_ctrl = self._make_controller(config=config)

        meta_ctrl.update_hyperparameters(confidence=0.0, reaction_time=0.0)
        # ema = 0.5*0.75 + 0.5*0.0 = 0.375
        assert abs(meta_ctrl.confidence_ema - 0.375) < 1e-9

        meta_ctrl.update_hyperparameters(confidence=1.0, reaction_time=0.0)
        # ema = 0.5*0.375 + 0.5*1.0 = 0.6875
        assert abs(meta_ctrl.confidence_ema - 0.6875) < 1e-9

        # Each smoothed value should be recorded for later episode logging
        assert meta_ctrl.confidences == [0.375, 0.6875]

    def test_underconfidence_grows_theta_and_raises_alpha(self) -> None:
        """
        Below-target confidence should widen the decision threshold (more
        deliberation) and raise the learning rate (more plastic).
        """

        decision_cfg = DecisionConfig(theta_start=1.0)
        learning_cfg = LearningConfig(alpha_start=0.1)
        meta_ctrl = self._make_controller(
            config=MetaConfig(confidence_target=0.75, ema_decay=0.0),
            decision_cfg=decision_cfg,
            learning_cfg=learning_cfg,
        )

        meta_ctrl.update_hyperparameters(confidence=0.0, reaction_time=0.0)

        assert meta_ctrl.hyperparameters.theta > decision_cfg.theta_start
        assert meta_ctrl.hyperparameters.alpha > learning_cfg.alpha_start

    def test_overconfidence_shrinks_theta_and_lowers_alpha(self) -> None:
        """
        Above-target confidence should narrow the decision threshold (faster
        decisions) and lower the learning rate (more stable).
        """

        decision_cfg = DecisionConfig(theta_start=1.0)
        learning_cfg = LearningConfig(alpha_start=0.1)
        meta_ctrl = self._make_controller(
            config=MetaConfig(confidence_target=0.75, ema_decay=0.0),
            decision_cfg=decision_cfg,
            learning_cfg=learning_cfg,
        )

        meta_ctrl.update_hyperparameters(confidence=1.0, reaction_time=0.0)

        assert meta_ctrl.hyperparameters.theta < decision_cfg.theta_start
        assert meta_ctrl.hyperparameters.alpha < learning_cfg.alpha_start

    def test_theta_is_clamped_to_bounds(self) -> None:
        """The decision threshold should never leave [theta_min, theta_max]"""

        decision_cfg = DecisionConfig(theta_start=1.0, theta_min=0.2, theta_max=3.0)

        # Strong over-confidence with a large rate should overshoot theta_min
        low_meta_ctrl = self._make_controller(
            config=MetaConfig(confidence_target=0.75, ema_decay=0.0, theta_rate=10.0),
            decision_cfg=decision_cfg,
        )
        low_meta_ctrl.update_hyperparameters(confidence=1.0, reaction_time=0.0)
        assert low_meta_ctrl.hyperparameters.theta == decision_cfg.theta_min

        # Strong under-confidence with a large rate should overshoot theta_max
        high_meta_ctrl = self._make_controller(
            config=MetaConfig(confidence_target=0.75, ema_decay=0.0, theta_rate=10.0),
            decision_cfg=decision_cfg,
        )
        high_meta_ctrl.update_hyperparameters(confidence=0.0, reaction_time=0.0)
        assert high_meta_ctrl.hyperparameters.theta == decision_cfg.theta_max

    def test_alpha_is_clamped_to_bounds(self) -> None:
        """The learning rate should never leave [alpha_min, alpha_max]"""

        learning_cfg = LearningConfig(alpha_start=0.1, alpha_min=0.01, alpha_max=0.5)

        # Strong over-confidence with a large rate should overshoot alpha_min
        low_meta_ctrl = self._make_controller(
            config=MetaConfig(confidence_target=0.75, ema_decay=0.0, alpha_rate=10.0),
            learning_cfg=learning_cfg,
        )
        low_meta_ctrl.update_hyperparameters(confidence=1.0, reaction_time=0.0)
        assert low_meta_ctrl.hyperparameters.alpha == learning_cfg.alpha_min

        # Strong under-confidence with a large rate should overshoot alpha_max
        high_meta_ctrl = self._make_controller(
            config=MetaConfig(confidence_target=0.75, ema_decay=0.0, alpha_rate=10.0),
            learning_cfg=learning_cfg,
        )
        high_meta_ctrl.update_hyperparameters(confidence=0.0, reaction_time=0.0)
        assert high_meta_ctrl.hyperparameters.alpha == learning_cfg.alpha_max

    def test_log_episode_logs_mean_confidence_and_resets(self) -> None:
        """log_episode() should log the mean of the episode's confidence values and clear them"""

        meta_ctrl = self._make_controller(config=MetaConfig(ema_decay=0.0))
        meta_ctrl.update_hyperparameters(confidence=0.4, reaction_time=0.0)
        meta_ctrl.update_hyperparameters(confidence=0.8, reaction_time=0.0)

        logger = LoggerStub()
        meta_ctrl.log_episode(logger=logger.as_summary_writer(), episode=3)

        assert len(logger.scalars) == 1
        tag, mean_confidence, episode = logger.scalars[0]
        assert tag == "metacognition/mean_confidence"
        assert abs(mean_confidence - 0.6) < 1e-9
        assert episode == 3
        # Episode data should be reset after logging
        assert meta_ctrl.confidences == []

    def test_log_episode_is_noop_when_no_confidences_recorded(self) -> None:
        """log_episode() should not log anything if no decision was made this episode"""

        meta_ctrl = self._make_controller()

        logger = LoggerStub()
        meta_ctrl.log_episode(logger=logger.as_summary_writer(), episode=0)

        assert logger.scalars == []

    def _make_controller(
        self,
        config: MetaConfig | None = None,
        decision_cfg: DecisionConfig | None = None,
        learning_cfg: LearningConfig | None = None,
    ) -> MetaController:
        decision_cfg = decision_cfg or DecisionConfig()
        learning_cfg = learning_cfg or LearningConfig()
        return MetaController(
            config=config or MetaConfig(),
            learning_cfg=learning_cfg,
            decision_cfg=decision_cfg,
            hyperparameters=Hyperparameters(
                decision_cfg=decision_cfg, learning_cfg=learning_cfg
            ),
        )
