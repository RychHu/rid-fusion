"""Explainable covariance-weighted state fusion baseline."""

from __future__ import annotations

from collections import defaultdict
from .models import FusionConfig, FusedState, ObservationGroup, RIDObservation


def _weighted(
    values: list[tuple[float, float, RIDObservation]],
) -> tuple[float | None, float | None, dict[str, float]]:
    if not values:
        return None, None, {}
    weights = [1.0 / max(variance, 1e-12) for _, variance, _ in values]
    total = sum(weights)
    value = sum(item[0] * weight for item, weight in zip(values, weights)) / total
    contributions: dict[str, float] = defaultdict(float)
    for (_, _, observation), weight in zip(values, weights):
        contributions[observation.protocol.value] += weight / total
    return value, 1.0 / total, dict(contributions)


def fuse_group_covariance_weighted(
    group: ObservationGroup, config: FusionConfig | None = None
) -> FusedState:
    config = config or FusionConfig()

    def position_values(name: str):
        result = []
        for observation in group.observations:
            value = getattr(observation, name)
            if value is not None:
                variance = observation.horizontal_variance_m2 or config.max_position_diff_m**2
                result.append((value, max(variance, config.min_variance), observation))
        return result

    def velocity_values(name: str):
        result = []
        for observation in group.observations:
            value = getattr(observation, name)
            if value is not None:
                variance = observation.velocity_variance_ms2 or 4.0
                result.append((value, max(variance, config.min_variance), observation))
        return result

    lat, lat_var, lat_weights = _weighted(position_values("lat_deg"))
    lon, lon_var, lon_weights = _weighted(position_values("lon_deg"))
    altitude_values = [
        (o.alt_m, max(o.vertical_variance_m2 or 225.0, config.min_variance), o)
        for o in group.observations
        if o.alt_m is not None
    ]
    alt, alt_var, _ = _weighted(altitude_values)
    vx, vx_var, _ = _weighted(velocity_values("vx_ms"))
    vy, vy_var, _ = _weighted(velocity_values("vy_ms"))
    vz, vz_var, _ = _weighted(velocity_values("vz_ms"))
    protocol_weights = defaultdict(float)
    for source in (lat_weights, lon_weights):
        for protocol, weight in source.items():
            protocol_weights[protocol] += weight / 2.0
    total = sum(protocol_weights.values())
    if total:
        protocol_weights = defaultdict(float, {k: v / total for k, v in protocol_weights.items()})
    return FusedState(
        track_key=group.track_key,
        timestamp_utc=group.window_end_utc,
        lat_deg=lat,
        lon_deg=lon,
        alt_m=alt,
        vx_ms=vx,
        vy_ms=vy,
        vz_ms=vz,
        covariance_diag=[lat_var, lon_var, alt_var, vx_var, vy_var, vz_var],
        protocol_weights=dict(protocol_weights),
        used_observation_ids=[o.observation_id for o in group.observations],
        source_protocols=sorted({o.protocol for o in group.observations}, key=lambda p: p.value),
        identity_conflict=group.identity_conflict,
        config_digest=config.digest,
    )
