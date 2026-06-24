"""
Unit tests for exploration decay algorithms.
"""

from numpy.testing import assert_almost_equal

from collabsort_agent.decision import Config as DecisionConfig
from collabsort_agent.decision.exploration_decay import (
    ExponentialExplorationDecay,
    LinearExplorationDecay,
)


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
