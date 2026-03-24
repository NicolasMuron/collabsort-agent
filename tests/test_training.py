"""
Unit tests for training.
"""

from gym_collabsort.config import Config as EnvConfig

from collabsort_agent.config import Config, load_cfg, save_cfg
from collabsort_agent.decision import Config as DecisionConfig
from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.memory import Config as MemoryConfig
from collabsort_agent.perception import Config as PerceptionConfig
from collabsort_agent.train import train


def test_random_agent() -> None:
    """Test an agent choosing random actions"""

    train(
        config=Config(
            env=EnvConfig(),
            perception=PerceptionConfig(),
            memory=MemoryConfig(),
            # epsilon = 1 => always explore randomly
            decision=DecisionConfig(epsilon_start=1, epsilon_min=1),
            learning=LearningConfig(),
            n_episodes=10,
            n_steps_episode=100,
            log_events=False,
            save_state=False,
        )
    )


def test_save_load_config(tmp_path) -> None:
    """Test saving and loading configuration from disk"""

    cfg = Config(
        env=EnvConfig(),
        perception=PerceptionConfig(),
        memory=MemoryConfig(),
        decision=DecisionConfig(),
        learning=LearningConfig(),
    )

    save_cfg(config=cfg, dir=tmp_path)
    cfg_loaded = load_cfg(dir=tmp_path)

    assert cfg_loaded == cfg
