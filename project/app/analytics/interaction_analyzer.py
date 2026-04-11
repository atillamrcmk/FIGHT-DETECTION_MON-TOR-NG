from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

from app.utils.math_utils import clamp, l2, normalize_0_100, pairwise_dist


@dataclass
class InteractionFeatures:
    interaction_score: float
    clustering_score: float
    active_interactions: int


class InteractionAnalyzer:
    """
    interaction_score:
      - if two people get closer quickly (recent distance drop), score increases
    clustering_score:
      - if multiple people are in a small area, score increases
    """

    def __init__(self, interaction_distance_px: float = 120.0, clustering_distance_px: float = 110.0):
        self.interaction_distance_px = float(interaction_distance_px)
        self.clustering_distance_px = float(clustering_distance_px)

        # Tunables (MVP)
        self._close_fast_drop_px = 45.0
        self._interaction_max = 120.0
        self._cluster_max = 120.0

    def compute(
        self,
        centers_now: List[Tuple[float, float]],
        centers_prev: List[Tuple[float, float]],
    ) -> InteractionFeatures:
        n = len(centers_now)
        if n < 2:
            return InteractionFeatures(interaction_score=0.0, clustering_score=0.0, active_interactions=0)

        d_now = pairwise_dist(centers_now)
        d_prev = pairwise_dist(centers_prev) if len(centers_prev) == n else None

        # clustering: count pairs within clustering_distance
        close_pairs = 0
        for i in range(n):
            for j in range(i + 1, n):
                if float(d_now[i, j]) < self.clustering_distance_px:
                    close_pairs += 1
        clustering_score = normalize_0_100(float(close_pairs) * 30.0, self._cluster_max)

        # interaction: distance drop among already-close-ish pairs
        active = 0
        interaction_raw = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                dn = float(d_now[i, j])
                if dn > self.interaction_distance_px:
                    continue
                if d_prev is None:
                    continue
                dp = float(d_prev[i, j])
                drop = max(0.0, dp - dn)
                if drop > self._close_fast_drop_px:
                    active += 1
                    interaction_raw += drop

        interaction_score = normalize_0_100(interaction_raw, self._interaction_max)
        return InteractionFeatures(
            interaction_score=interaction_score,
            clustering_score=clustering_score,
            active_interactions=active,
        )

    def compute_tracked(
        self,
        centers_now_by_id: Dict[int, Tuple[float, float]],
        centers_prev_by_id: Dict[int, Tuple[float, float]],
    ) -> InteractionFeatures:
        """
        Interaction scoring with track-id stability:
        - clustering uses all current centers
        - interaction uses only ids present in both frames to compute distance drops
        This avoids the "list length mismatch => no interaction" pitfall in variable person-count scenes.
        """
        centers_now = list(centers_now_by_id.values())
        if len(centers_now) < 2:
            return InteractionFeatures(interaction_score=0.0, clustering_score=0.0, active_interactions=0)

        # clustering: count pairs within clustering_distance (current frame only)
        d_now_all = pairwise_dist(centers_now)
        close_pairs = 0
        for i in range(len(centers_now)):
            for j in range(i + 1, len(centers_now)):
                if float(d_now_all[i, j]) < self.clustering_distance_px:
                    close_pairs += 1
        clustering_score = normalize_0_100(float(close_pairs) * 30.0, self._cluster_max)

        # interaction: compute only on common ids
        common_ids = sorted(set(centers_now_by_id.keys()) & set(centers_prev_by_id.keys()))
        if len(common_ids) < 2:
            return InteractionFeatures(interaction_score=0.0, clustering_score=clustering_score, active_interactions=0)

        cn = [centers_now_by_id[i] for i in common_ids]
        cp = [centers_prev_by_id[i] for i in common_ids]
        d_now = pairwise_dist(cn)
        d_prev = pairwise_dist(cp)

        active = 0
        interaction_raw = 0.0
        n = len(common_ids)
        for i in range(n):
            for j in range(i + 1, n):
                dn = float(d_now[i, j])
                if dn > self.interaction_distance_px:
                    continue
                dp = float(d_prev[i, j])
                drop = max(0.0, dp - dn)
                if drop > self._close_fast_drop_px:
                    active += 1
                    interaction_raw += drop

        interaction_score = normalize_0_100(interaction_raw, self._interaction_max)
        return InteractionFeatures(
            interaction_score=interaction_score,
            clustering_score=clustering_score,
            active_interactions=active,
        )

