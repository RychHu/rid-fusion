"""Time-window and position-gated association for RID observations."""

from __future__ import annotations

import math
from .models import FusionConfig, ObservationGroup, RIDObservation
from .training.metrics import horizontal_error_m


def _distance_m(a: RIDObservation, b: RIDObservation) -> float | None:
    if None in (a.lat_deg, a.lon_deg, b.lat_deg, b.lon_deg):
        return None
    horizontal = horizontal_error_m(a.lat_deg, a.lon_deg, b.lat_deg, b.lon_deg)
    if a.alt_m is None or b.alt_m is None:
        return horizontal
    return math.hypot(horizontal, a.alt_m - b.alt_m)


def associate_observations(
    observations: list[RIDObservation], config: FusionConfig | None = None
) -> list[ObservationGroup]:
    """Associate observations by target identity and a bounded time window.

    Identity remains the primary key. Position gating does not silently discard
    contradictory reports; it marks the group as an identity conflict so the
    evidence remains auditable.
    """
    config = config or FusionConfig()
    buckets: dict[tuple[str, int], list[RIDObservation]] = {}
    for observation in sorted(
        observations, key=lambda item: (item.source_timestamp_utc, item.observation_id)
    ):
        bucket = int(round(observation.source_timestamp_utc / config.association_time_window_s))
        buckets.setdefault((observation.uas_id, bucket), []).append(observation)

    groups: list[ObservationGroup] = []
    for (track_key, _), items in buckets.items():
        times = [item.source_timestamp_utc for item in items]
        group = ObservationGroup(
            track_key=track_key,
            window_start_utc=min(times),
            window_end_utc=max(times),
            observations=items,
        )
        reference = items[0]
        for item in items:
            distance = _distance_m(reference, item)
            if distance is None:
                group.association_scores[item.observation_id] = 0.5
                continue
            var_a = reference.horizontal_variance_m2 or config.max_position_diff_m**2
            var_b = item.horizontal_variance_m2 or config.max_position_diff_m**2
            gate = max(
                config.max_position_diff_m,
                config.association_gate_sigma * math.sqrt(var_a + var_b),
            )
            group.association_scores[item.observation_id] = max(0.0, 1.0 - distance / gate)
            if distance > gate:
                group.identity_conflict = True
                group.conflict_reasons.append(
                    f"{reference.observation_id} 与 {item.observation_id} 相距 {distance:.1f}m，超过门限 {gate:.1f}m"
                )
        groups.append(group)
    return sorted(groups, key=lambda group: (group.window_start_utc, group.track_key))
