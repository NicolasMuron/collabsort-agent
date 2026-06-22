import numpy as np
from collabsort_agent.decision import Deliberator

class Greedy(Deliberator):
    """Sélection purement gloutonne. L'exploration est gérée en interne par NoisyNet."""
    def choose_action(self, state: np.ndarray, training_step: int) -> int:
        action_values = self.estimator.get_action_values(state=state)
        return int(np.argmax(action_values).item())

    def log_episode(self, logger, episode: int) -> None:
        pass
    def save_state(self, dir: str) -> None:
        pass
    def load_state(self, dir: str) -> None:
        pass