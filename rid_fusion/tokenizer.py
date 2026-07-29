"""
tokenizer.py — Spatio-temporal tokenization for RID signals.

Converts raw SpatioTemporalToken objects into fixed-dimensional
embedding vectors ready for cross-modal attention fusion.

Steps:
  1. Normalise each field to [0, 1] or appropriate range
  2. Encode spatial fields (lat/lon/alt → sinusoidal position encoding)
  3. Encode protocol type (one-hot → learned embedding)
  4. Concatenate into a single token vector
"""

from __future__ import annotations
import logging
import math
import numpy as np
from typing import Optional
from rid_fusion.models import SpatioTemporalToken, ProtocolType

logger = logging.getLogger(__name__)


class SpatialEncoder:
    """
    Maps (lat, lon, alt) to a sinusoidal position embedding.
    This preserves spatial relationships: nearby points have similar embeddings.
    """

    def __init__(self, dim: int = 32):
        self.dim = dim

    def encode(
        self,
        lat_deg: float,
        lon_deg: float,
        alt_m: float,
    ) -> np.ndarray:
        """
        Sinusoidal encoding of 3D position.
        Uses different frequency bands for lat, lon, alt.
        """
        vec = np.zeros(self.dim)
        for i in range(0, self.dim, 6):
            freq = 10000 ** (i / self.dim)
            if i + 5 < self.dim:
                vec[i]     = math.sin(lat_deg * freq)
                vec[i + 1] = math.cos(lat_deg * freq)
                vec[i + 2] = math.sin(lon_deg * freq)
                vec[i + 3] = math.cos(lon_deg * freq)
                vec[i + 4] = math.sin(alt_m / 100.0 * freq)
                vec[i + 5] = math.cos(alt_m / 100.0 * freq)
        return vec


class TemporalEncoder:
    """Encodes timestamp and optionally a time-delta from sequence start."""

    def __init__(self, dim: int = 16):
        self.dim = dim

    def encode(self, timestamp_utc: float, t0: float = 0.0) -> np.ndarray:
        vec = np.zeros(self.dim)
        dt = timestamp_utc - t0
        for i in range(self.dim):
            freq = 10000 ** (2 * i / self.dim)
            if i % 2 == 0:
                vec[i] = math.sin(dt * freq)
            else:
                vec[i] = math.cos(dt * freq)
        return vec


class SignalEncoder:
    """
    Encodes signal quality metrics (RSSI, SNR) into a fixed vector.

    Uses sinusoidal encoding across frequency bands so that different
    RSSI/SNR combinations produce distinguishable embeddings —
    unlike a fixed-weight blend that collapses to near-constant values.
    """

    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode(self, rssi_dbm: float, snr_db: float, protocol: ProtocolType) -> np.ndarray:
        # Normalise RSSI: typical range [-120, -20] dBm
        rssi_norm = max(0.0, min(1.0, (rssi_dbm + 120) / 100.0))

        # Normalise SNR: typical range [-10, 40] dB
        snr_norm = max(0.0, min(1.0, (snr_db + 10) / 50.0))

        vec = np.zeros(self.dim)
        for i in range(self.dim):
            # Each dimension uses a different frequency scale so that
            # (rssi, snr) pairs map to distinct points in ℝ^dim
            freq = 1000.0 ** (i / max(self.dim - 1, 1))
            vec[i] = rssi_norm * math.sin(freq) + snr_norm * math.cos(freq)
        return vec


class ProtocolEncoder:
    """One-hot → learned embedding lookup for protocol type."""

    PROTOCOL_EMBEDDINGS = {
        ProtocolType.WIFI_BEACON:  np.array([1.0, 0.0, 0.0, 0.0]),
        ProtocolType.BLE_ADVB:     np.array([0.0, 1.0, 0.0, 0.0]),
        ProtocolType.NR_BROADCAST: np.array([0.0, 0.0, 1.0, 0.0]),
        ProtocolType.LORAWAN:      np.array([0.0, 0.0, 0.0, 1.0]),
        ProtocolType.ADS_B:        np.array([0.5, 0.5, 0.5, 0.5]),
    }

    @classmethod
    def encode(cls, protocol: ProtocolType) -> np.ndarray:
        return cls.PROTOCOL_EMBEDDINGS.get(protocol, np.zeros(4))


class ContextEncoder:
    """Encodes weather context into a small vector."""

    def __init__(self, dim: int = 8):
        self.dim = dim

    def encode(
        self,
        wind_speed_ms: float,
        precipitation_mmh: float,
        visibility_m: float,
    ) -> np.ndarray:
        wind_norm = min(wind_speed_ms / 30.0, 1.0)
        precip_norm = min(precipitation_mmh / 50.0, 1.0)
        vis_norm = min(visibility_m / 10000.0, 1.0)

        vec = np.zeros(self.dim)
        vec[0] = wind_norm
        vec[1] = precip_norm
        vec[2] = vis_norm
        # Fill remaining with sinusoidal encoding of the triple
        for i in range(3, self.dim):
            freq = 2 ** (i - 3)
            vec[i] = math.sin((wind_norm + precip_norm + vis_norm) * freq)
        return vec


class Tokenizer:
    """
    Full tokenization pipeline.
    Converts SpatioTemporalToken → fixed-dimension embedding vector.
    """

    def __init__(
        self,
        spatial_dim: int = 32,
        temporal_dim: int = 16,
        signal_dim: int = 8,
        protocol_dim: int = 4,
        context_dim: int = 8,
        hidden_dim: int = 128,
        seed: int = 42,
    ):
        self.spatial_enc  = SpatialEncoder(spatial_dim)
        self.temporal_enc = TemporalEncoder(temporal_dim)
        self.signal_enc   = SignalEncoder(signal_dim)
        self.context_enc  = ContextEncoder(context_dim)
        self.hidden_dim   = hidden_dim

        # Projection matrix — Xavier/Glorot initialisation (deterministic via seed)
        input_dim = spatial_dim + temporal_dim + signal_dim + protocol_dim + context_dim
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (input_dim + hidden_dim))
        self.W_proj = rng.uniform(-limit, limit, (input_dim, hidden_dim))

    @property
    def output_dim(self) -> int:
        return self.hidden_dim

    def encode(self, token: SpatioTemporalToken, t0: float = 0.0) -> np.ndarray:
        """Convert a single token to an embedding vector."""

        # 1. Encode each modality
        spatial_vec  = self.spatial_enc.encode(token.lat_deg, token.lon_deg, token.alt_m)
        temporal_vec = self.temporal_enc.encode(token.timestamp_utc, t0)
        signal_vec   = self.signal_enc.encode(token.rssi_dbm, token.snr_db, token.protocol)
        protocol_vec = ProtocolEncoder.encode(token.protocol)
        context_vec  = self.context_enc.encode(
            token.wind_speed_ms, token.precipitation_mmh, token.visibility_m
        )

        # 2. Concatenate
        concat = np.concatenate([
            spatial_vec, temporal_vec, signal_vec, protocol_vec, context_vec
        ])

        # 3. Project to hidden dimension
        embedding = concat @ self.W_proj

        # 4. Layer normalisation (simplified)
        std = np.std(embedding) + 1e-8
        embedding = embedding / std

        return embedding

    def encode_batch(
        self,
        tokens: list[SpatioTemporalToken],
        t0: float = 0.0,
    ) -> np.ndarray:
        """Encode a batch of tokens → (N, hidden_dim) array."""
        if not tokens:
            return np.zeros((0, self.hidden_dim))
        return np.stack([self.encode(t, t0) for t in tokens])


def deduplicate_tokens(
    tokens: list[SpatioTemporalToken],
    time_window_s: float = 0.5,
    max_position_diff_m: float = 50.0,
) -> list[SpatioTemporalToken]:
    """
    Cross-protocol deduplication: if the same drone_id has multiple tokens
    within time_window_s, merge them into one.

    Strategy:
      - Group by (drone_id, time_bucket)
      - Within each group, check that reported positions are consistent
        (within max_position_diff_m). If a token's position deviates
        significantly from the group median, it is flagged as anomalous
        and excluded from the merge.
      - Keep the token with the highest RSSI among consistent reporters
        as the representative.

    This step saves computation in the fusion encoder without losing
    information, and guards against faulty sensors that report RSSI
    loudly but positions incorrectly.
    """
    if not tokens:
        return []

    # Approximate metres-per-degree constants
    _lat_m_per_deg = 111320.0

    # Pre-compute longitude scaling per token (depends on latitude)
    def _lon_m_per_deg(lat: float) -> float:
        return 111320.0 * max(math.cos(math.radians(lat)), 1e-3)

    # Group by (drone_id, time_bucket)
    groups: dict[tuple[str, int], list[SpatioTemporalToken]] = {}
    for t in tokens:
        bucket = int(t.timestamp_utc / time_window_s)
        key = (t.drone_id, bucket)
        groups.setdefault(key, []).append(t)

    merged = []
    for key, group in groups.items():
        if len(group) == 1:
            merged.append(group[0])
            continue

        # ── Position consistency check ──
        # Compute median lat/lon as reference point
        lats = np.array([t.lat_deg for t in group])
        lons = np.array([t.lon_deg for t in group])
        med_lat = np.median(lats)
        med_lon = np.median(lons)
        lon_scale = _lon_m_per_deg(med_lat)

        consistent: list[SpatioTemporalToken] = []
        for t in group:
            dlat_m = abs(t.lat_deg - med_lat) * _lat_m_per_deg
            dlon_m = abs(t.lon_deg - med_lon) * lon_scale
            dist_m = math.sqrt(dlat_m * dlat_m + dlon_m * dlon_m)
            if dist_m <= max_position_diff_m:
                consistent.append(t)
            # else: token is an outlier — silently excluded

        # Fallback: if all are outliers, keep all (system may be in GPS-denied zone)
        if not consistent:
            consistent = group

        # Keep the token with highest RSSI as the representative
        best = max(consistent, key=lambda x: x.rssi_dbm)
        # Record all source protocols (including outliers for transparency)
        best.protocol_payload["dedup_sources"] = [t.protocol.value for t in group]
        outliers = [t.protocol.value for t in group if t not in consistent]
        best.protocol_payload["dedup_outliers"] = outliers
        if outliers:
            logger.debug("Dedup outliers excluded: drone=%s bucket=%d outliers=%s",
                         best.drone_id, key[1], outliers)
        merged.append(best)

    return merged
