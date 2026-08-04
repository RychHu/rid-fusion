"""Comparison baselines and truth-based evaluation."""

from __future__ import annotations

from statistics import mean
from .classical_fusion import fuse_group_covariance_weighted
from .models import FusionConfig, FusedState, ObservationGroup
from .training.metrics import horizontal_error_m, rmse


def fuse_group_simple_average(
    group: ObservationGroup, config: FusionConfig | None = None
) -> FusedState:
    config = config or FusionConfig()

    def average(name: str):
        values = [getattr(o, name) for o in group.observations if getattr(o, name) is not None]
        return mean(values) if values else None

    return FusedState(
        track_key=group.track_key,
        timestamp_utc=group.window_end_utc,
        lat_deg=average("lat_deg"), lon_deg=average("lon_deg"), alt_m=average("alt_m"),
        vx_ms=average("vx_ms"), vy_ms=average("vy_ms"), vz_ms=average("vz_ms"),
        used_observation_ids=[o.observation_id for o in group.observations],
        source_protocols=sorted({o.protocol for o in group.observations}, key=lambda p: p.value),
        identity_conflict=group.identity_conflict,
        algorithm_version="simple-average/v1",
        config_digest=config.digest,
    )


def fuse_group_best_source(
    group: ObservationGroup, config: FusionConfig | None = None
) -> FusedState:
    config = config or FusionConfig()
    observation = min(
        group.observations,
        key=lambda o: o.horizontal_variance_m2 if o.horizontal_variance_m2 is not None else float("inf"),
    )
    return FusedState(
        track_key=group.track_key, timestamp_utc=group.window_end_utc,
        lat_deg=observation.lat_deg, lon_deg=observation.lon_deg, alt_m=observation.alt_m,
        vx_ms=observation.vx_ms, vy_ms=observation.vy_ms, vz_ms=observation.vz_ms,
        used_observation_ids=[observation.observation_id],
        source_protocols=[observation.protocol], identity_conflict=group.identity_conflict,
        algorithm_version="best-source/v1", config_digest=config.digest,
    )


def compare_algorithms(
    groups: list[ObservationGroup],
    truth: list[dict] | None = None,
    config: FusionConfig | None = None,
) -> dict:
    config = config or FusionConfig()
    algorithms = {
        "best_single_source": fuse_group_best_source,
        "simple_average": fuse_group_simple_average,
        "covariance_weighted": fuse_group_covariance_weighted,
    }
    truth = truth or []
    results = {}
    for name, function in algorithms.items():
        states = [function(group, config) for group in groups]
        errors = []
        for state in states:
            candidates = [p for p in truth if p.get("drone_id") == state.track_key]
            if state.lat_deg is None or state.lon_deg is None or not candidates:
                continue
            point = min(candidates, key=lambda p: abs(float(p["timestamp_utc"]) - state.timestamp_utc))
            errors.append(horizontal_error_m(state.lat_deg, state.lon_deg, point["lat_deg"], point["lon_deg"]))
        results[name] = {
            "horizontal_rmse_m": rmse(errors),
            "sample_count": len(errors),
            "state_count": len(states),
        }
    return results
