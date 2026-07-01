"""
Unit tests for exploration decay algorithms.
"""

import numpy as np
from numpy.testing import assert_almost_equal

from collabsort_agent.decision import Config as DecisionConfig
from collabsort_agent.decision.epsilon_greedy import EpsilonGreedy
from collabsort_agent.decision.exploration_decay import (
    ExponentialExplorationDecay,
    LinearExplorationDecay,
)
from collabsort_agent.learning import ActionValueEstimator
from collabsort_agent.learning import Config as LearningConfig


class EstimatorStub(ActionValueEstimator):
    """Minimal estimator stub used by the decision tests."""

    def __init__(self) -> None:
        super().__init__(config=LearningConfig(), n_actions=3)

    def get_action_values(self, state: np.ndarray) -> np.ndarray:
        return np.zeros(3, dtype=np.float32)

    def update_action_values(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> None:
        return None

    def save_state(self, dir: str) -> None:
        return None

    def load_state(self, dir: str) -> None:
        return None


def test_linear_explo_decay() -> None:
    """Test linear exploration decay"""

    config = DecisionConfig(decay_span=0.6)
    total_steps = 1000

    lin_decay = LinearExplorationDecay(config=config, total_steps=total_steps)

    # Assert epsilon value at beginning of decay
    assert lin_decay.get_epsilon(training_step=0) == config.epsilon_start

    # Assert epsilon value at middle of decay
    assert (
        lin_decay.get_epsilon(training_step=int(total_steps * config.decay_span) // 2)
        == config.epsilon_min + (config.epsilon_start - config.epsilon_min) / 2
    )

    # Assert epsilon value at end of decay
    assert_almost_equal(
        lin_decay.get_epsilon(training_step=int(total_steps * config.decay_span)),
        config.epsilon_min,
    )

    # Assert epsilon value at end of training
    assert_almost_equal(
        lin_decay.get_epsilon(training_step=total_steps), config.epsilon_min
    )


def test_exponential_explo_decay() -> None:
    """Test exponential exploration decay"""

    config = DecisionConfig()
    total_steps = 1000

    exp_decay = ExponentialExplorationDecay(config=config, total_steps=total_steps)

    # Assert epsilon value at beginning of decay
    assert exp_decay.get_epsilon(training_step=0) == config.epsilon_start

    # Assert epsilon value at end of decay
    assert_almost_equal(
        exp_decay.get_epsilon(training_step=int(total_steps * config.decay_span)),
        config.epsilon_min,
        decimal=2,
    )

    # Assert epsilon value at end of training
    assert_almost_equal(
        exp_decay.get_epsilon(training_step=total_steps), config.epsilon_min, decimal=2
    )


def test_epsilon_greedy_keeps_decay_by_default() -> None:
    """By default, the deliberator should not reset exploration on a new phase."""

    config = DecisionConfig(epsilon_start=0.9, epsilon_min=0.1, decay_span=0.5)
    deliberator = EpsilonGreedy(
        config=config,
        estimator=EstimatorStub(),
        exploration_decay=LinearExplorationDecay(config=config, total_steps=100),
        rng=np.random.default_rng(0),
    )

    deliberator.epsilon = 0.2
    deliberator.reset_for_phase(phase_steps=50)

    assert deliberator.epsilon == 0.2


def test_epsilon_greedy_resets_for_new_phase() -> None:
    """A new phase should restart exploration from epsilon_start when enabled."""

    config = DecisionConfig(
        epsilon_start=0.9,
        epsilon_min=0.1,
        decay_span=0.5,
        reset_exploration_per_phase=True,
    )
    deliberator = EpsilonGreedy(
        config=config,
        estimator=EstimatorStub(),
        exploration_decay=LinearExplorationDecay(config=config, total_steps=100),
        rng=np.random.default_rng(0),
    )

    deliberator.epsilon = 0.2
    deliberator.reset_for_phase(phase_steps=50)

    assert deliberator.epsilon == config.epsilon_start
    assert deliberator.exploration_decay.decay_steps == 25
