"""Top-level simulation, association, fusion and anomaly pipeline."""

from __future__ import annotations

from dataclasses import asdict
import numpy as np
from .anomaly import detect_anomalies
from .association import associate_observations
from .classical_fusion import fuse_group_covariance_weighted
from .encoders import FusionEncoder
from .models import FusionConfig, FusedState, FusedToken, ProtocolType, RIDObservation, SpatioTemporalToken
from .signals import RIDSignalSimulator, generate_drone_trajectory


class RIDFusionEngine:
    """Deterministic research pipeline used by both Python and WPF clients."""

    def __init__(
        self,
        protocols: list[ProtocolType],
        d_model: int = 128,
        seed: int = 42,
        enable_dedup: bool = True,
    ):
        self.protocols = protocols
        self.seed = seed
        self.enable_dedup = enable_dedup
        self.config = FusionConfig(d_model=d_model, seed=seed)
        self.simulator = RIDSignalSimulator(protocols, seed=seed)
        self.encoder = FusionEncoder(d_model=d_model, n_heads=self.config.n_heads, seed=seed)
        self.rng = np.random.default_rng(seed)

    def run_on_trajectory(
        self,
        drone_id: str,
        start_lat: float,
        start_lon: float,
        start_alt: float,
        duration_s: float = 60.0,
        dt_s: float = 1.0,
        speed_ms: float = 8.0,
        heading_deg: float = 45.0,
        wind_speed_ms: float = 0.0,
        precipitation_mmh: float = 0.0,
        visibility_m: float = 10000.0,
    ) -> dict:
        trajectory = generate_drone_trajectory(
            drone_id, start_lat, start_lon, start_alt, duration_s, dt_s,
            speed_ms, heading_deg, self.rng,
        )
        tokens: list[SpatioTemporalToken] = []
        for point in trajectory:
            tokens.extend(self.simulator.observe(
                drone_id=point["drone_id"], timestamp_utc=point["timestamp_utc"],
                lat_deg=point["lat_deg"], lon_deg=point["lon_deg"], alt_m=point["alt_m"],
                vx_ms=point["vx_ms"], vy_ms=point["vy_ms"], vz_ms=point["vz_ms"],
                wind_speed_ms=wind_speed_ms, precipitation_mmh=precipitation_mmh,
                visibility_m=visibility_m,
            ))
        return self._process_tokens(
            tokens, trajectory, duration_s, wind_speed_ms,
            precipitation_mmh, visibility_m,
        )

    def _process_tokens(
        self,
        all_tokens: list[SpatioTemporalToken],
        trajectory: list[dict],
        duration_s: float,
        wind_speed_ms: float = 0.0,
        precipitation_mmh: float = 0.0,
        visibility_m: float = 10000.0,
    ) -> dict:
        observations = [RIDObservation.from_token(token) for token in all_tokens]
        result = self.process_observations(observations)
        states: list[FusedState] = result["states"]

        # Keep the attention representation as an auxiliary feature. The
        # explainable physical state remains the primary output.
        by_key: dict[tuple[str, float], list[SpatioTemporalToken]] = {}
        for token in all_tokens:
            by_key.setdefault((token.drone_id, token.timestamp_utc), []).append(token)
        fused_tokens: list[FusedToken] = []
        for state in states:
            candidates = min(
                (key for key in by_key if key[0] == state.track_key),
                key=lambda key: abs(key[1] - state.timestamp_utc),
                default=None,
            )
            if candidates is None:
                continue
            tokens = by_key[candidates]
            semantic = self.encoder.fuse(
                [t for t in tokens if t.protocol == ProtocolType.WIFI_BEACON],
                [t for t in tokens if t.protocol == ProtocolType.BLE_ADVB],
                [t for t in tokens if t.protocol == ProtocolType.NR_BROADCAST],
                wind_speed=wind_speed_ms, precip=precipitation_mmh,
                visibility=visibility_m,
            )
            if semantic:
                state.semantic_embedding = semantic[0].embedding
                fused_tokens.extend(semantic)

        protocol_distribution: dict[str, int] = {}
        for observation in observations:
            name = observation.protocol.value
            protocol_distribution[name] = protocol_distribution.get(name, 0) + 1
        stats = {
            "total_raw_tokens": len(all_tokens),
            "total_fused_tokens": len(states),
            "protocol_distribution": protocol_distribution,
            "dedup_enabled": False,
            "trajectory_duration_s": duration_s,
            "trajectory_steps": len(trajectory),
            "received_observations": len(observations),
            "parsed_observations": len(observations),
            "rejected_observations": 0,
            "associated_groups": len(result["groups"]),
            "used_observations": sum(len(state.used_observation_ids) for state in states),
            "excluded_observations": sum(len(state.excluded_observations) for state in states),
            "fused_states": len(states),
            "config_digest": self.config.digest,
            "target_count": len({state.track_key for state in states}),
        }
        return {
            "raw_tokens": all_tokens,
            "observations": observations,
            "groups": result["groups"],
            "states": states,
            "fused_tokens": fused_tokens,
            "trajectory": trajectory,
            "anomalies": result["anomalies"],
            "stats": stats,
        }

    def ingest_batch(self, observations: list[RIDObservation]) -> list[FusedState]:
        return self.process_observations(observations)["states"]

    def process_observations(self, observations: list[RIDObservation]) -> dict:
        groups = associate_observations(observations, self.config)
        states = [fuse_group_covariance_weighted(group, self.config) for group in groups]
        anomalies = detect_anomalies(observations, groups)
        by_target: dict[str, int] = {}
        for event in anomalies:
            by_target[event.target_id] = by_target.get(event.target_id, 0) + 1
        for state in states:
            state.anomaly_score = float(by_target.get(state.track_key, 0))
        return {"groups": groups, "states": states, "anomalies": anomalies}

    def run_multi_target(
        self,
        targets: list[dict],
        duration_s: float = 60.0,
        dt_s: float = 1.0,
        wind_speed_ms: float = 0.0,
        precipitation_mmh: float = 0.0,
        visibility_m: float = 10000.0,
    ) -> dict:
        trajectory: list[dict] = []
        tokens: list[SpatioTemporalToken] = []
        for target in targets:
            points = generate_drone_trajectory(
                target["drone_id"], target["start_lat"], target["start_lon"],
                target["start_alt"], duration_s, dt_s,
                target.get("speed_ms", 8.0), target.get("heading_deg", 45.0), self.rng,
            )
            trajectory.extend(points)
            for point in points:
                tokens.extend(self.simulator.observe(
                    drone_id=point["drone_id"], timestamp_utc=point["timestamp_utc"],
                    lat_deg=point["lat_deg"], lon_deg=point["lon_deg"], alt_m=point["alt_m"],
                    vx_ms=point["vx_ms"], vy_ms=point["vy_ms"], vz_ms=point["vz_ms"],
                    wind_speed_ms=wind_speed_ms, precipitation_mmh=precipitation_mmh,
                    visibility_m=visibility_m,
                ))
        result = self._process_tokens(tokens, trajectory, duration_s, wind_speed_ms, precipitation_mmh, visibility_m)
        result["targets"] = targets
        return result


def compare_single_vs_fused(
    engine: RIDFusionEngine,
    drone_id: str = "DJI-2024-A17F",
    start_lat: float = 30.5728,
    start_lon: float = 104.0668,
    start_alt: float = 100.0,
    duration_s: float = 30.0,
) -> dict:
    result = engine.run_on_trajectory(drone_id, start_lat, start_lon, start_alt, duration_s)
    counts = {
        protocol.value.lower(): sum(1 for token in result["raw_tokens"] if token.protocol == protocol)
        for protocol in engine.protocols
    }
    enriched = sum(1 for token in result["fused_tokens"] if len(token.source_protocols) >= 2)
    return {
        "single_protocol": counts,
        "fused_total": len(result["fused_tokens"]),
        "fused_enriched": enriched,
        "enrichment_ratio": enriched / max(len(result["fused_tokens"]), 1),
        "trajectory": result["trajectory"],
        "fused_tokens": result["fused_tokens"],
    }
