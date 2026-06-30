"""
Unit tests for training.
"""

from gym_collabsort.config import Config as EnvConfig

from collabsort_agent.config import Config, load_cfg, save_cfg
from collabsort_agent.decision import Config as DecisionConfig
from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.learning.n_step_learning import NStepLearning
from collabsort_agent.metacognition import Config as MetaConfig, MetaController
from collabsort_agent.memory import Config as MemoryConfig
from collabsort_agent.perception import Config as PerceptionConfig
from collabsort_agent.train import _build_estimator, train


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
            meta=MetaConfig(),
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
        meta=MetaConfig(),
    )

    save_cfg(config=cfg, dir=tmp_path)
    cfg_loaded = load_cfg(dir=tmp_path)

    assert cfg_loaded == cfg


def test_n_step_learning_config_is_propagated() -> None:
    """Verify n_step is exposed from LearningConfig to NStepLearning."""

    learning_cfg = LearningConfig(algorithm="n_step", n_step=5)
    cfg = Config(
        env=EnvConfig(),
        perception=PerceptionConfig(),
        memory=MemoryConfig(),
        decision=DecisionConfig(),
        learning=learning_cfg,
        meta=MetaConfig(),
    )
    meta_ctrl = MetaController(
        config=MetaConfig(), learning_cfg=learning_cfg, decision_cfg=DecisionConfig()
    )

    estimator = _build_estimator(
        algo_name="n_step",
        config=cfg,
        n_actions=4,
        state_size=5,
        meta_ctrl=meta_ctrl,
    )

    assert isinstance(estimator, NStepLearning)
    assert estimator.n_step == 5
