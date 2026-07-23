"""
Unit tests for metacognition algorithms.
"""

import numpy as np

from collabsort_agent.decision import Config as DecisionConfig
from collabsort_agent.decision.accumulators import Accumulators
from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.metacognition import Hyperparameters
from collabsort_agent.metacognition.confidence import BayesianConfidence, GapConfidence


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
