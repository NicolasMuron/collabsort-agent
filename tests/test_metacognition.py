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
from collabsort_agent.metacognition.confidence import (
    BayesianConfidence,
    GapConfidence,
    TDErrorCalibration,
)
from collabsort_agent.metacognition.controller import MetaController
from collabsort_agent.metacognition.monitoring import MetaMonitoring


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


class TestMetaMonitoring:
    def test_confidence_ema_warm_starts_at_target(self) -> None:
        """The confidence EMA should warm-start at the configured target"""

        meta_monitoring = self._make_monitoring(
            config=MetaConfig(confidence_target=0.6)
        )

        assert meta_monitoring.smoothed_confidence == 0.6

    def test_compute_decision_confidence_smooths_with_ema(self) -> None:
        """Confidence should be smoothed via EMA rather than used raw"""

        config = MetaConfig(confidence_target=0.75, ema_decay=0.5)
        meta_monitoring = self._make_monitoring(config=config)

        # Gap confidence with theta=1.0: (min_evidence[chosen] - min_evidence[runnerup]) / theta
        zero_confidence = _make_accumulators(
            drift_rates=[0.0, 0.0], evidence=[0.5, 0.5]
        )
        full_confidence = _make_accumulators(
            drift_rates=[0.0, 0.0], evidence=[0.0, 1.0]
        )

        ema = meta_monitoring.compute_decision_confidence(
            chosen_action=1,
            runnerup_action=0,
            reaction_time=10.0,
            accumulators=zero_confidence,
        )
        # ema = 0.5*0.75 + 0.5*0.0 = 0.375
        assert abs(ema - 0.375) < 1e-9
        assert abs(meta_monitoring.smoothed_confidence - 0.375) < 1e-9

        ema = meta_monitoring.compute_decision_confidence(
            chosen_action=1,
            runnerup_action=0,
            reaction_time=10.0,
            accumulators=full_confidence,
        )
        # ema = 0.5*0.375 + 0.5*1.0 = 0.6875
        assert abs(ema - 0.6875) < 1e-6
        assert abs(meta_monitoring.smoothed_confidence - 0.6875) < 1e-6

    def test_compute_decision_confidence_records_value(self) -> None:
        """Each computed confidence should be appended right after computation"""

        meta_monitoring = self._make_monitoring()
        accumulators = _make_accumulators(drift_rates=[0.0, 0.0], evidence=[0.7, 1.0])

        meta_monitoring.compute_decision_confidence(
            chosen_action=1,
            runnerup_action=0,
            reaction_time=10.0,
            accumulators=accumulators,
        )
        assert len(meta_monitoring.confidences) == 1

        meta_monitoring.compute_decision_confidence(
            chosen_action=1,
            runnerup_action=0,
            reaction_time=10.0,
            accumulators=accumulators,
        )
        assert len(meta_monitoring.confidences) == 2

    def test_log_episode_logs_mean_confidence_and_resets(self) -> None:
        """log_episode() should log the mean of the episode's confidence values and clear them"""

        meta_monitoring = self._make_monitoring()
        accumulators = _make_accumulators(drift_rates=[0.0, 0.0], evidence=[0.7, 1.0])

        meta_monitoring.compute_decision_confidence(
            chosen_action=1,
            runnerup_action=0,
            reaction_time=10.0,
            accumulators=accumulators,
        )
        meta_monitoring.compute_decision_confidence(
            chosen_action=1,
            runnerup_action=0,
            reaction_time=10.0,
            accumulators=accumulators,
        )
        expected_mean_confidence = sum(meta_monitoring.confidences) / 2

        logger = LoggerStub()
        meta_monitoring.log_episode(logger=logger.as_summary_writer(), episode=3)

        assert len(logger.scalars) == 1
        tag, mean_confidence, episode = logger.scalars[0]
        assert tag == "metacognition/mean_confidence"
        assert abs(mean_confidence - expected_mean_confidence) < 1e-9
        assert episode == 3
        # Episode data should be reset after logging
        assert meta_monitoring.confidences == []

    def test_log_episode_is_noop_when_no_confidences_recorded(self) -> None:
        """log_episode() should not log anything if no decision was made this episode"""

        meta_monitoring = self._make_monitoring()

        logger = LoggerStub()
        meta_monitoring.log_episode(logger=logger.as_summary_writer(), episode=0)

        assert logger.scalars == []

    def _make_monitoring(self, config: MetaConfig | None = None) -> MetaMonitoring:
        decision_cfg = DecisionConfig(theta_start=1.0)
        hyperparameters = Hyperparameters(
            decision_cfg=decision_cfg, learning_cfg=LearningConfig()
        )
        return MetaMonitoring(
            config=config or MetaConfig(),
            confidence_method=GapConfidence(
                decision_cfg=decision_cfg, hyperparameters=hyperparameters
            ),
        )


class TestTDErrorCalibrationMethod:
    def test_compute_outcome_is_one_for_nonnegative_td_error(self) -> None:
        """A non-negative TD-error (no negative surprise) should count as a success outcome"""

        calibration = TDErrorCalibration(config=MetaConfig())

        assert calibration.compute_outcome(td_error=0.5) == 1.0
        assert calibration.compute_outcome(td_error=0.0) == 1.0

    def test_compute_outcome_is_zero_for_negative_td_error(self) -> None:
        """A negative TD-error (negative surprise) should count as a failure outcome"""

        calibration = TDErrorCalibration(config=MetaConfig())

        assert calibration.compute_outcome(td_error=-0.1) == 0.0

    def test_bias_starts_at_zero(self) -> None:
        """The calibration bias should start at zero (no correction applied initially)"""

        calibration = TDErrorCalibration(config=MetaConfig())

        assert calibration.bias == 0.0
        assert calibration.calibrate(confidence=0.7) == 0.7

    def test_update_raises_bias_when_outcome_exceeds_confidence(self) -> None:
        """
        An outcome better than what confidence predicted (underconfidence)
        should raise the calibration bias (Eq 6d-6e).
        """

        calibration = TDErrorCalibration(config=MetaConfig(calibration_rate=0.1))

        calibration.update(confidence=0.4, outcome=1.0)

        # bias <- 0.0 + 0.1 * (1.0 - 0.4) = 0.06
        assert abs(calibration.bias - 0.06) < 1e-9

    def test_update_lowers_bias_when_outcome_is_below_confidence(self) -> None:
        """
        An outcome worse than what confidence predicted (overconfidence)
        should lower the calibration bias (Eq 6d-6e).
        """

        calibration = TDErrorCalibration(config=MetaConfig(calibration_rate=0.1))

        calibration.update(confidence=0.9, outcome=0.0)

        # bias <- 0.0 + 0.1 * (0.0 - 0.9) = -0.09
        assert abs(calibration.bias - (-0.09)) < 1e-9

    def test_calibrate_clips_to_valid_probability_range(self) -> None:
        """Calibrated confidence should always stay within [0, 1]"""

        calibration = TDErrorCalibration(config=MetaConfig())

        calibration.bias = 0.5
        assert calibration.calibrate(confidence=0.9) == 1.0

        calibration.bias = -0.5
        assert calibration.calibrate(confidence=0.2) == 0.0

    def test_repeated_success_outcomes_converge_bias_toward_underconfidence_gap(
        self,
    ) -> None:
        """
        Repeatedly recalibrating a fixed raw confidence against a fixed,
        better-than-predicted outcome should converge the *calibrated*
        confidence toward that outcome (closed calibrate/update loop,
        asymptotic convergence of Eq 6e-6f).
        """

        calibration = TDErrorCalibration(config=MetaConfig(calibration_rate=0.2))

        raw_confidence = 0.5
        outcome = 1.0
        calibrated_confidence = raw_confidence
        for _ in range(200):
            calibrated_confidence = calibration.calibrate(confidence=raw_confidence)
            calibration.update(confidence=calibrated_confidence, outcome=outcome)

        # The calibrated confidence the agent would act on should converge
        # close to the outcome it's being calibrated against.
        assert abs(calibrated_confidence - outcome) < 1e-3


class TestMetaMonitoringCalibration:
    def test_no_calibration_method_leaves_confidence_unchanged(self) -> None:
        """Without a calibration method, reported confidence should equal the EMA'd value"""

        meta_monitoring = self._make_monitoring(calibration_method=None)
        accumulators = _make_accumulators(drift_rates=[0.0, 0.0], evidence=[0.7, 1.0])

        confidence = meta_monitoring.compute_decision_confidence(
            chosen_action=1,
            runnerup_action=0,
            reaction_time=10.0,
            accumulators=accumulators,
        )

        assert confidence == meta_monitoring.smoothed_confidence

    def test_update_calibration_is_noop_without_calibration_method(self) -> None:
        """update_calibration() should be a no-op if no calibration method is configured"""

        meta_monitoring = self._make_monitoring(calibration_method=None)
        accumulators = _make_accumulators(drift_rates=[0.0, 0.0], evidence=[0.7, 1.0])

        meta_monitoring.compute_decision_confidence(
            chosen_action=1,
            runnerup_action=0,
            reaction_time=10.0,
            accumulators=accumulators,
        )
        meta_monitoring.update_calibration(td_error=-1.0)  # Should not raise

        assert meta_monitoring.calibration_biases == []

    def test_update_calibration_is_noop_before_any_decision(self) -> None:
        """update_calibration() should be a no-op if no confidence has been reported yet"""

        calibration_method = TDErrorCalibration(config=MetaConfig())
        meta_monitoring = self._make_monitoring(calibration_method=calibration_method)

        meta_monitoring.update_calibration(td_error=1.0)

        assert calibration_method.bias == 0.0
        assert meta_monitoring.calibration_biases == []

    def test_compute_decision_confidence_applies_calibration_bias(self) -> None:
        """A non-zero calibration bias should shift the reported confidence"""

        calibration_method = TDErrorCalibration(config=MetaConfig())
        calibration_method.bias = 0.2
        meta_monitoring = self._make_monitoring(calibration_method=calibration_method)
        accumulators = _make_accumulators(drift_rates=[0.0, 0.0], evidence=[0.7, 1.0])

        confidence = meta_monitoring.compute_decision_confidence(
            chosen_action=1,
            runnerup_action=0,
            reaction_time=10.0,
            accumulators=accumulators,
        )

        assert abs(confidence - (meta_monitoring.smoothed_confidence + 0.2)) < 1e-9

    def test_update_calibration_uses_last_reported_confidence(self) -> None:
        """
        update_calibration() should recalibrate against the confidence value
        that was actually reported for the most recent decision (i.e. after
        EMA smoothing and any prior calibration), not the raw confidence.
        """

        calibration_method = TDErrorCalibration(config=MetaConfig(calibration_rate=0.1))
        meta_monitoring = self._make_monitoring(
            config=MetaConfig(ema_decay=0.0, calibration_rate=0.1),
            calibration_method=calibration_method,
        )
        # Zero confidence (chosen == runner-up evidence, gap = 0)
        accumulators = _make_accumulators(drift_rates=[0.0, 0.0], evidence=[0.5, 0.5])

        reported_confidence = meta_monitoring.compute_decision_confidence(
            chosen_action=1,
            runnerup_action=0,
            reaction_time=10.0,
            accumulators=accumulators,
        )
        assert reported_confidence == 0.0

        # Positive TD-error -> outcome = 1.0 -> confidence_error = 1.0 - 0.0
        meta_monitoring.update_calibration(td_error=0.5)

        # bias <- 0.0 + 0.1 * (1.0 - 0.0) = 0.1
        assert abs(calibration_method.bias - 0.1) < 1e-9
        assert meta_monitoring.calibration_biases == [calibration_method.bias]

    def test_log_episode_logs_mean_calibration_bias_and_resets(self) -> None:
        """log_episode() should log the mean calibration bias for the episode and clear it"""

        calibration_method = TDErrorCalibration(config=MetaConfig(calibration_rate=0.1))
        meta_monitoring = self._make_monitoring(
            config=MetaConfig(ema_decay=0.0),
            calibration_method=calibration_method,
        )
        accumulators = _make_accumulators(drift_rates=[0.0, 0.0], evidence=[0.5, 0.5])

        meta_monitoring.compute_decision_confidence(
            chosen_action=1,
            runnerup_action=0,
            reaction_time=10.0,
            accumulators=accumulators,
        )
        meta_monitoring.update_calibration(td_error=0.5)
        meta_monitoring.compute_decision_confidence(
            chosen_action=1,
            runnerup_action=0,
            reaction_time=10.0,
            accumulators=accumulators,
        )
        meta_monitoring.update_calibration(td_error=-0.5)

        expected_mean_bias = sum(meta_monitoring.calibration_biases) / 2

        logger = LoggerStub()
        meta_monitoring.log_episode(logger=logger.as_summary_writer(), episode=5)

        tags = {tag for tag, _, _ in logger.scalars}
        assert "metacognition/calibration_bias" in tags

        bias_entry = next(
            entry
            for entry in logger.scalars
            if entry[0] == "metacognition/calibration_bias"
        )
        assert abs(bias_entry[1] - expected_mean_bias) < 1e-9
        assert bias_entry[2] == 5
        assert meta_monitoring.calibration_biases == []

    def _make_monitoring(
        self,
        config: MetaConfig | None = None,
        calibration_method: TDErrorCalibration | None = None,
    ) -> MetaMonitoring:
        decision_cfg = DecisionConfig(theta_start=1.0)
        hyperparameters = Hyperparameters(
            decision_cfg=decision_cfg, learning_cfg=LearningConfig()
        )
        return MetaMonitoring(
            config=config or MetaConfig(),
            confidence_method=GapConfidence(
                decision_cfg=decision_cfg, hyperparameters=hyperparameters
            ),
            calibration_method=calibration_method,
        )


class TestMetaController:
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
