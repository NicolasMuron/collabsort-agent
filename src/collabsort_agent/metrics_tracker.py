"""
Metrics tracker for generating and logging training heatmaps.
"""

from collections import defaultdict
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from gym_collabsort.config import Action
from torch.utils.tensorboard import SummaryWriter


def _log_heatmap(
    logger: SummaryWriter,
    episode: int,
    data_mat: np.ndarray,
    tag: str,
    title: str,
    xlabel: str,
    ylabel: str,
    colorbar_label: str,
    cmap: str = "viridis",
    origin: Literal["upper", "lower"] | None = None,
    figsize: tuple[int, int] = (6, 3),
    yticks_labels: list[str] | None = None,
    xticks_labels: list[str] | None = None,
) -> None:
    """Log a generic heatmap to TensorBoard"""
    fig, ax = plt.subplots(figsize=figsize)
    if origin is not None:
        im = ax.imshow(data_mat, aspect="auto", cmap=cmap, origin=origin)
    else:
        im = ax.imshow(data_mat, aspect="auto", cmap=cmap)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if yticks_labels is not None:
        ax.set_yticks(range(len(yticks_labels)))
        ax.set_yticklabels(yticks_labels)

    if xticks_labels is not None:
        ax.set_xticks(range(len(xticks_labels)))
        ax.set_xticklabels(xticks_labels, rotation=45, ha="right")

    fig.colorbar(im, ax=ax, label=colorbar_label)
    fig.tight_layout()
    logger.add_figure(tag, fig, global_step=episode)
    plt.close(fig)


class HeatmapTracker:
    """Tracks and logs detailed spatial and interaction heatmaps over episodes."""

    def __init__(self, log_freq: int = 20):
        self.log_freq = log_freq
        self.action_counts_history: list[np.ndarray] = []
        self.picked_values_history: list[np.ndarray] = []
        self.visitation_history: list[np.ndarray] = []
        self.collision_history: list[np.ndarray] = []
        self.spatial_action_counts: dict[int, dict[int, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def update(
        self,
        action_history: list[int],
        picked_values: list[float],
        episode_visitation: np.ndarray,
        episode_collisions: np.ndarray,
        spatial_actions: dict[int, dict[int, int]],
        n_actions: int,
    ) -> None:
        """Update tracker histories with data from a single episode."""
        # 1. Action distribution
        if len(action_history) > 0:
            counts = np.bincount(
                np.array(action_history, dtype=np.int32), minlength=n_actions
            )
            self.action_counts_history.append(counts)

        # 2. Picked values
        if len(picked_values) > 0:
            counts = np.bincount(np.array(picked_values, dtype=np.int32), minlength=9)
        else:
            counts = np.zeros(9, dtype=np.int32)
        self.picked_values_history.append(counts)

        # 3. Spatial stats
        # episode_visitation and episode_collisions are expected to include the row zero pad (size n_rows + 1)
        # We store rows 1 to N
        self.visitation_history.append(episode_visitation[1:])
        self.collision_history.append(episode_collisions[1:])

        # 4. Spatial actions
        # Merge dicts
        for r, actions_dict in spatial_actions.items():
            for a, c in actions_dict.items():
                self.spatial_action_counts[r][a] += c

    def log_heatmaps(
        self, logger: SummaryWriter | None, episode: int, n_actions: int
    ) -> None:
        """Log all heatmaps periodically."""
        if logger is None or episode == 0 or episode % self.log_freq != 0:
            return

        # 1. Picked Objects Values Heatmap
        if len(self.picked_values_history) > 0:
            mat_vals = np.vstack(self.picked_values_history)
            _log_heatmap(
                logger=logger,
                episode=episode,
                data_mat=mat_vals.T,
                tag="training/picked_objects_heatmap",
                title="Picked Objects Values Heatmap",
                xlabel="episode",
                ylabel="object reward value",
                colorbar_label="count",
                origin="lower",
                yticks_labels=[str(v) for v in range(9)],
            )

        # 2. Action distribution heatmap
        if len(self.action_counts_history) > 0:
            mat = np.vstack(self.action_counts_history)
            labels = []
            for i in range(mat.shape[1]):
                try:
                    labels.append(Action(i).name)
                except Exception:
                    labels.append(f"action_{i}")
            _log_heatmap(
                logger=logger,
                episode=episode,
                data_mat=mat.T,
                tag="actions/heatmap",
                title="Action distribution heatmap",
                xlabel="episode",
                ylabel="action",
                colorbar_label="count",
                yticks_labels=labels,
            )

        # 3. Visitation per episode
        if len(self.visitation_history) > 0:
            mat_vis = np.vstack(self.visitation_history)
            _log_heatmap(
                logger=logger,
                episode=episode,
                data_mat=mat_vis.T,
                tag="spatial/visitation_episodes",
                title="Visitation per episode",
                xlabel="episode",
                ylabel="agent row (Y)",
                colorbar_label="visits",
                yticks_labels=[str(r) for r in range(1, mat_vis.shape[1] + 1)],
            )

        # 4. Collisions per episode
        if len(self.collision_history) > 0:
            mat_col = np.vstack(self.collision_history)
            if np.sum(mat_col) > 0:
                _log_heatmap(
                    logger=logger,
                    episode=episode,
                    data_mat=mat_col.T,
                    tag="spatial/collisions_episodes",
                    title="Collisions per episode",
                    xlabel="episode",
                    ylabel="agent row (Y)",
                    colorbar_label="collisions",
                    cmap="Reds",
                    yticks_labels=[str(r) for r in range(1, mat_col.shape[1] + 1)],
                )

        # 5. Spatial Action Frequencies
        if len(self.spatial_action_counts) > 0:
            rows = sorted(list(self.spatial_action_counts.keys()))
            mat = np.zeros((len(rows), n_actions))
            for i, r in enumerate(rows):
                row_total = sum(self.spatial_action_counts[r].values())
                for a, c in self.spatial_action_counts[r].items():
                    if a < n_actions:
                        mat[i, a] = c / row_total if row_total > 0 else 0

            labels = []
            for i in range(n_actions):
                try:
                    labels.append(Action(i).name)
                except Exception:
                    labels.append(f"action_{i}")

            _log_heatmap(
                logger=logger,
                episode=episode,
                data_mat=mat,
                tag="spatial/action_frequencies",
                title="Action frequencies per row",
                xlabel="action",
                ylabel="agent row (Y)",
                colorbar_label="probability",
                figsize=(6, 4),
                yticks_labels=[str(r) for r in rows],
                xticks_labels=labels,
            )
