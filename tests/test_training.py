"""
Unit tests for training and curriculum learning.
"""

import json

import gymnasium as gym
from gym_collabsort.config import Config as EnvConfig

from collabsort_agent.common import create_agent
from collabsort_agent.config import Config, load_cfg, save_cfg
from collabsort_agent.decision import DecisionConfig
from collabsort_agent.learning import LearningConfig
from collabsort_agent.memory import MemoryConfig
from collabsort_agent.metacognition import MetaConfig
from collabsort_agent.perception import PerceptionConfig
from collabsort_agent.train import CurriculumPhase, load_phases, train


def test_compute_total_training_steps() -> None:
    """The total curriculum length should be the sum of all phase steps."""

    phases = [
        CurriculumPhase(name="phase-1", n_episodes=2, env_config=EnvConfig()),
        CurriculumPhase(name="phase-2", n_episodes=3, env_config=EnvConfig()),
    ]
    n_steps_episode = 10

    total_steps = sum(p.n_episodes * n_steps_episode for p in phases)
    assert total_steps == 50


def test_random_agent() -> None:
    """Test a standard training loop (single default phase)."""

    cfg = Config(
        env=EnvConfig(),
        perception=PerceptionConfig(),
        memory=MemoryConfig(),
        decision=DecisionConfig(epsilon_start=1, epsilon_min=1),
        learning=LearningConfig(),
        meta=MetaConfig(),
        n_episodes=2,
        n_steps_episode=50,
        log_events=False,
        save_state=False,
    )

    phases = load_phases(base_config=cfg, json_path=None)
    assert len(phases) == 1

    train(base_config=cfg, phases=phases)


def test_train_curriculum(tmp_path) -> None:
    """Test multi-phase curriculum training with a JSON config."""

    json_path = tmp_path / "dummy_curriculum.json"
    dummy_phases = [
        {
            "name": "Phase 1 - Easy",
            "n_episodes": 2,
            "env_overrides": {"robot_enabled": False, "active_treadmills": ["upper"]},
        },
        {
            "name": "Phase 2 - Hard",
            "n_episodes": 2,
            "env_overrides": {
                "robot_enabled": True,
                "reward_noise_std": 0.5,
                "active_treadmills": ["upper", "lower"],
            },
        },
    ]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dummy_phases, f)

    cfg = Config(
        env=EnvConfig(),
        perception=PerceptionConfig(),
        memory=MemoryConfig(),
        decision=DecisionConfig(epsilon_start=1, epsilon_min=1),
        learning=LearningConfig(),
        meta=MetaConfig(),
        n_episodes=2,
        n_steps_episode=50,
        log_events=False,
        save_state=False,
    )

    phases = load_phases(base_config=cfg, json_path=str(json_path))

    assert len(phases) == 2
    assert phases[0].env_config.robot_enabled is False
    assert phases[1].env_config.robot_enabled is True
    assert phases[1].env_config.reward_noise_std == 0.5

    train(base_config=cfg, phases=phases)


def test_train_from_pretrained(tmp_path) -> None:
    """Test resuming/fine-tuning training from a pretrained agent state."""

    cfg = Config(
        env=EnvConfig(),
        perception=PerceptionConfig(),
        memory=MemoryConfig(),
        decision=DecisionConfig(epsilon_start=1, epsilon_min=1),
        learning=LearningConfig(),
        meta=MetaConfig(),
        n_episodes=2,
        n_steps_episode=20,
        log_events=False,
        save_state=False,
    )

    phases = load_phases(base_config=cfg, json_path=None)

    # 1. Sauvegarde d'un état d'agent initial dans un dossier temporaire
    pretrained_dir = str(tmp_path / "pretrained_agent")
    env = gym.make("CollabSort-v0", config=cfg.env)
    agent = create_agent(
        config=cfg, sample_obs=env.observation_space.sample(), rng=env.np_random
    )
    agent.save_state(dir=pretrained_dir)
    env.close()

    # 2. Entraînement à partir de l'état pré-entraîné
    train(
        base_config=cfg,
        phases=phases,
        pretrained_state_dir=pretrained_dir,
    )


def test_save_load_config(tmp_path) -> None:
    """Test saving and loading configuration from disk."""

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
