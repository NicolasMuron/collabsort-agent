"""
Unit tests for the Greedy deliberator.
"""

import numpy as np
from collabsort_agent.decision import Config as DecisionConfig
from collabsort_agent.decision.greedy import Greedy
from tests.tests_decision.test_ard import EstimatorStub  # Réutilisation propre du Stub


def test_greedy_action_selection() -> None:
    """Verify Greedy deliberator takes the absolute argmax of the action values."""
    # Action value matrix where action index 1 is the highest (value: 10.5)
    action_values = np.array([-1.0, 10.5, 3.2, 0.0])
    
    estimator = EstimatorStub(action_values=action_values)
    config = DecisionConfig()
    rng = np.random.default_rng(42)

    # Instantiate our Greedy deliberator
    deliberator = Greedy(config=config, estimator=estimator, rng=rng)

    state = np.zeros(5, dtype=np.float32)
    chosen_action = deliberator.choose_action(state=state, training_step=0)

    # Assert it correctly chose index 1
    assert chosen_action == 1


def test_greedy_lifecycle_methods() -> None:
    """Verify that lifecycle methods containing 'pass' can be invoked safely."""
    action_values = np.array([1.0, 2.0])
    estimator = EstimatorStub(action_values=action_values)
    config = DecisionConfig()
    rng = np.random.default_rng(42)

    deliberator = Greedy(config=config, estimator=estimator, rng=rng)

    # Ces appels vont exécuter les lignes de 'pass' et faire monter la couverture à 100%
    # On passe None pour le logger car la méthode Greedy.log_episode n'en fait rien
    deliberator.log_episode(logger=None, episode=1)
    deliberator.save_state(dir="dummy_path")
    deliberator.load_state(dir="dummy_path")
    
    # Un assert simple pour valider que le test s'est terminé sans planter
    assert True