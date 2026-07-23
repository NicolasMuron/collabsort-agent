"""
Metacognitive monitoring algorithms.
"""

from collabsort_agent.decision.ard import Accumulators
from collabsort_agent.metacognition.confidence import ConfidenceMethod


class MetaMonitoring:
    """Metacognitive monitoring"""

    def __init__(self, confidence_method: ConfidenceMethod) -> None:

        self.confidence_method = confidence_method

    def compute_decision_confidence(
        self,
        chosen_action: int,
        runnerup_action: int,
        reaction_time: float,
        accumulators: Accumulators,
    ) -> float:
        """Compute confidence associated with a decision"""

        return self.confidence_method.compute_decision_confidence(
            chosen_action=chosen_action,
            runnerup_action=runnerup_action,
            reaction_time=reaction_time,
            accumulators=accumulators,
        )
