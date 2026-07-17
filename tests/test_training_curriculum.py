"""
Unit tests for curriculum training.
"""

import json

from gym_collabsort.config import Config as EnvConfig

from collabsort_agent.config import Config
from collabsort_agent.decision import Config as DecisionConfig
from collabsort_agent.learning import Config as LearningConfig
from collabsort_agent.memory import Config as MemoryConfig
from collabsort_agent.metacognition import Config as MetaConfig
from collabsort_agent.perception import Config as PerceptionConfig
from collabsort_agent.train_curriculum import (
    CurriculumPhase,
    compute_total_training_steps,
    load_curriculum_from_json,
    train_curriculum,
)


def test_compute_total_training_steps() -> None:
    """The total curriculum length should be the sum of all phase steps."""

    phases = [
        CurriculumPhase(name="phase-1", n_episodes=2, env_config=EnvConfig()),
        CurriculumPhase(name="phase-2", n_episodes=3, env_config=EnvConfig()),
    ]

    assert compute_total_training_steps(phases=phases, n_steps_episode=10) == 50


def test_train_curriculum(tmp_path) -> None:
    """Test curriculum training loop with a dummy JSON config"""

    # 1. Create a dummy curriculum json file in a temporary folder
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

    # 2. Create a base configuration (random agent for speed)
    cfg = Config(
        env=EnvConfig(),
        perception=PerceptionConfig(),
        memory=MemoryConfig(),
        decision=DecisionConfig(
            epsilon_start=1, epsilon_min=1
        ),  # Always explore randomly
        learning=LearningConfig(),
        meta=MetaConfig(),
        n_episodes=2,
        n_steps_episode=50,
        log_events=False,
        save_state=False,
    )

    # 3. Load curriculum and run training
    phases = load_curriculum_from_json(base_config=cfg, json_path=str(json_path))

    # Assert phases were correctly loaded
    assert len(phases) == 2
    assert phases[0].env_config.robot_enabled is False
    assert phases[1].env_config.robot_enabled is True
    assert phases[1].env_config.reward_noise_std == 0.5

    # Run the curriculum training loop (if it crashes, the test fails)
    train_curriculum(base_config=cfg, phases=phases)
