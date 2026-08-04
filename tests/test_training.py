"""
Unit tests for training.
"""

import gymnasium as gym
from gym_collabsort.config import Config as EnvConfig

from collabsort_agent.config import Config, load_cfg, save_cfg
from collabsort_agent.decision import DecisionConfig
from collabsort_agent.learning import LearningConfig
from collabsort_agent.memory import MemoryConfig
from collabsort_agent.metacognition import MetaConfig
from collabsort_agent.perception import PerceptionConfig
from collabsort_agent.train import create_agent, train


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


def test_train_with_pretrained_state(tmp_path) -> None:
    """Test training resumes from a pretrained agent state."""

    cfg = Config(
        env=EnvConfig(),
        perception=PerceptionConfig(),
        memory=MemoryConfig(),
        decision=DecisionConfig(epsilon_start=1, epsilon_min=1),
        learning=LearningConfig(),
        meta=MetaConfig(),
        n_episodes=1,
        n_steps_episode=10,
        log_events=False,
        save_state=False,
    )

    env = gym.make("CollabSort-v0", config=cfg.env)
    agent = create_agent(
        config=cfg, sample_obs=env.observation_space.sample(), rng=env.np_random
    )
    pretrained_dir = tmp_path / "pretrained"
    agent.save_state(dir=str(pretrained_dir))
    env.close()

    train(config=cfg, pretrained_state_dir=str(pretrained_dir))


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
