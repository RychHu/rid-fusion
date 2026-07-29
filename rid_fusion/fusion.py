"""
fusion.py — Top-level fusion engine.

Orchestrates the full pipeline:
  1. Signal simulation
  2. Tokenization + protocol-specific encoding
  3. Cross-modal attention fusion
  4. Output: protocol-agnostic FusedToken stream
"""

from __future__ import annotations
import logging
from typing import Optional
import numpy as np
from rid_fusion.models import SpatioTemporalToken, FusedToken, ProtocolType
from rid_fusion.signals import RIDSignalSimulator, generate_drone_trajectory
from rid_fusion.tokenizer import Tokenizer, deduplicate_tokens
from rid_fusion.encoders import FusionEncoder

logger = logging.getLogger(__name__)


class RIDFusionEngine:
    """
    Main engine: simulate multi-protocol RID signals, fuse them,
    output protocol-agnostic semantic tokens.
    """

    def __init__(
        self,
        protocols: list[ProtocolType],
        d_model: int = 128,
        seed: int = 42,
        enable_dedup: bool = True,
    ):
        self.simulator = RIDSignalSimulator(protocols, seed=seed)
        self.tokenizer = Tokenizer(hidden_dim=d_model, seed=seed)
        self.encoder = FusionEncoder(d_model=d_model, seed=seed)
        self.protocols = protocols
        self.enable_dedup = enable_dedup

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
        """
        Run the full fusion pipeline on a synthetic drone trajectory.

        Returns:
            dict with keys:
              - raw_tokens: list of all SpatioTemporalToken
              - fused_tokens: list of FusedToken
              - trajectory: list of ground-truth points
              - stats: dict of fusion statistics
        """

        # 1. Generate ground-truth trajectory
        trajectory = generate_drone_trajectory(
            drone_id=drone_id,
            start_lat=start_lat,
            start_lon=start_lon,
            start_alt=start_alt,
            duration_s=duration_s,
            dt_s=dt_s,
            speed_ms=speed_ms,
            heading_deg=heading_deg,
        )

        # 2. Simulate multi-protocol RID signals
        all_tokens: list[SpatioTemporalToken] = []
        for point in trajectory:
            tokens = self.simulator.observe(
                drone_id=point["drone_id"],
                timestamp_utc=point["timestamp_utc"],
                lat_deg=point["lat_deg"],
                lon_deg=point["lon_deg"],
                alt_m=point["alt_m"],
                vx_ms=point["vx_ms"],
                vy_ms=point["vy_ms"],
                vz_ms=point["vz_ms"],
                wind_speed_ms=wind_speed_ms,
                precipitation_mmh=precipitation_mmh,
                visibility_m=visibility_m,
            )
            all_tokens.extend(tokens)

        # 3. Token deduplication (cross-protocol merge)
        if self.enable_dedup:
            before = len(all_tokens)
            all_tokens = deduplicate_tokens(all_tokens)
            after = len(all_tokens)
            logger.info("Deduplication: %d → %d tokens (%.1f%% reduction)",
                        before, after, (1 - after / max(before, 1)) * 100)

        # 4. Separate by protocol for fusion
        wifi_tokens = [t for t in all_tokens if t.protocol == ProtocolType.WIFI_BEACON]
        ble_tokens  = [t for t in all_tokens if t.protocol == ProtocolType.BLE_ADVB]
        nr_tokens   = [t for t in all_tokens if t.protocol == ProtocolType.NR_BROADCAST]

        # 5. Cross-modal fusion
        fused_tokens = self.encoder.fuse(
            wifi_tokens=wifi_tokens,
            ble_tokens=ble_tokens,
            nr_tokens=nr_tokens,
            t0=trajectory[0]["timestamp_utc"] if trajectory else 0.0,
            wind_speed=wind_speed_ms,
            precip=precipitation_mmh,
            visibility=visibility_m,
        )

        # 6. Statistics
        protocol_counts = {}
        for t in all_tokens:
            protocol_counts[t.protocol.value] = protocol_counts.get(t.protocol.value, 0) + 1

        stats = {
            "total_raw_tokens": len(all_tokens),
            "total_fused_tokens": len(fused_tokens),
            "protocol_distribution": protocol_counts,
            "dedup_enabled": self.enable_dedup,
            "trajectory_duration_s": duration_s,
            "trajectory_steps": len(trajectory),
        }

        return {
            "raw_tokens": all_tokens,
            "fused_tokens": fused_tokens,
            "trajectory": trajectory,
            "stats": stats,
        }


def compare_single_vs_fused(
    engine: RIDFusionEngine,
    drone_id: str = "DJI-2024-A17F",
    start_lat: float = 30.5728,
    start_lon: float = 104.0668,
    start_alt: float = 100.0,
    duration_s: float = 30.0,
) -> dict:
    """
    Compare single-protocol accuracy vs fused accuracy.
    Demonstrates the benefit of multi-protocol fusion.
    """
    result = engine.run_on_trajectory(
        drone_id=drone_id,
        start_lat=start_lat,
        start_lon=start_lon,
        start_alt=start_alt,
        duration_s=duration_s,
    )

    # Count tokens per protocol
    wifi_count = sum(1 for t in result["raw_tokens"] if t.protocol == ProtocolType.WIFI_BEACON)
    ble_count  = sum(1 for t in result["raw_tokens"] if t.protocol == ProtocolType.BLE_ADVB)
    nr_count   = sum(1 for t in result["raw_tokens"] if t.protocol == ProtocolType.NR_BROADCAST)

    # Fused tokens with multiple source protocols are "enriched"
    enriched = sum(1 for ft in result["fused_tokens"] if len(ft.source_protocols) >= 2)

    return {
        "single_protocol": {"wifi": wifi_count, "ble": ble_count, "nr": nr_count},
        "fused_total": len(result["fused_tokens"]),
        "fused_enriched": enriched,
        "enrichment_ratio": enriched / max(len(result["fused_tokens"]), 1),
        "trajectory": result["trajectory"],
        "fused_tokens": result["fused_tokens"],
    }
