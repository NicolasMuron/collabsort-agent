"""
Optimal reward computation via Dynamic Programming.

At the end of each training episode, the agent takes the exact trajectory
(robot positions + objects that appeared) recorded during that episode and
computes the OPTIMAL cumulative reward an agent could have achieved.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from collabsort_agent.config import Config


@dataclass
class StepRecord:
    """State of the environment at one time step, from the agent's perspective."""

    robot_row: int
    pickable: list[tuple[int, float]] = field(default_factory=list)


@dataclass
class EpisodeTrajectory:
    """Full recorded trajectory of one episode."""

    steps: list[StepRecord] = field(default_factory=list)

    def record(
        self,
        obs: dict,
        active_agent_rewards: np.ndarray,
        arm_base_col: int,
    ) -> None:
        """Append one step record from the current observation."""
        robot_row = int(obs["robot"][0])

        pickable: list[tuple[int, float]] = []
        for obj in obs["moving_objects"]:
            if int(obj["coords"][1]) == arm_base_col:
                row = int(obj["coords"][0])
                value = float(active_agent_rewards[obj["color"], obj["shape"]])
                pickable.append((row, value))

        self.steps.append(StepRecord(robot_row=robot_row, pickable=pickable))


def compute_optimal_reward(
    trajectory: EpisodeTrajectory, config: Config
) -> tuple[float, list[int]]:
    """
    Compute the maximum cumulative reward achievable on the given episode
    trajectory using backward Dynamic Programming, and the optimal action sequence.
    """
    env_cfg = config.env
    n_rows: int = env_cfg.n_rows
    base_row: int = n_rows
    T: int = len(trajectory.steps)

    if T == 0:
        return 0.0, []

    max_k = n_rows - 1

    # V[(r, k)]
    V: dict[tuple[int, int], float] = {
        (r, k): 0.0 for r in range(1, n_rows + 1) for k in range(max_k + 1)
    }

    step_rew: float = float(env_cfg.step_reward)
    move_pen: float = float(env_cfg.movement_penalty)
    coll_pen: float = float(env_cfg.collision_penalty)
    fail_pen: float = float(env_cfg.failed_action_penalty)

    def _collision(agent_row: int, robot_row: int) -> bool:
        return agent_row == robot_row

    pi: dict[int, dict[tuple[int, int], int]] = {}

    for t in range(T - 1, -1, -1):
        record = trajectory.steps[t]
        robot_row = record.robot_row
        pickable_map: dict[int, float] = dict(record.pickable)

        new_V: dict[tuple[int, int], float] = {}

        pi[t] = {}
        for r in range(1, n_rows + 1):
            for k in range(max_k + 1):
                if k > 0:
                    next_r = min(r + 1, base_row)
                    next_k = k - 1
                    rew = step_rew
                    if _collision(next_r, robot_row):
                        rew += coll_pen
                    new_V[(r, k)] = rew + V[(next_r, next_k)]
                else:
                    options: list[tuple[float, int]] = []

                    # NONE (0)
                    rew_none = step_rew
                    if _collision(r, robot_row):
                        rew_none += coll_pen
                    options.append((rew_none + V[(r, 0)], 0))

                    # UP (1)
                    if r > 1:
                        nr_up = r - 1
                        rew_up = step_rew + move_pen
                        if _collision(nr_up, robot_row):
                            rew_up += coll_pen
                        options.append((rew_up + V[(nr_up, 0)], 1))

                    # DOWN (2)
                    if r < n_rows:
                        nr_down = r + 1
                        rew_down = step_rew + move_pen
                        if _collision(nr_down, robot_row):
                            rew_down += coll_pen
                        options.append((rew_down + V[(nr_down, 0)], 2))

                    # PICK (3)
                    if r in pickable_map:
                        rew_pick = step_rew + pickable_map[r]
                        back_k = base_row - r
                        options.append((rew_pick + V[(r, back_k)], 3))
                    else:
                        rew_fail = step_rew + fail_pen
                        options.append((rew_fail + V[(r, 0)], 3))

                    best_val, best_act = max(options, key=lambda x: x[0])
                    new_V[(r, k)] = best_val
                    pi[t][(r, k)] = best_act

        V = new_V

    # Forward pass to extract optimal actions
    opt_actions: list[int] = []
    curr_r = base_row
    curr_k = 0

    for t in range(T):
        if curr_k > 0:
            curr_r = min(curr_r + 1, base_row)
            curr_k -= 1
            opt_actions.append(0)  # Agent is moving back, action has no effect (NONE)
        else:
            act = pi[t][(curr_r, curr_k)]
            opt_actions.append(act)
            if act == 1:
                curr_r -= 1
            elif act == 2:
                curr_r += 1
            elif act == 3:
                pickable_map_t = dict(trajectory.steps[t].pickable)
                if curr_r in pickable_map_t:
                    curr_k = base_row - curr_r

    return V[(base_row, 0)], opt_actions
