"""
models.py — Core data structures for the RID fusion engine.

Token:       unified spatio-temporal representation of a single RID observation
Detection:   a single sensor's report of a drone at one instant
FusedResult: output of cross-modal attention fusion
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import uuid


# ═══════════════════════════════════════════════════════════
# Protocol & object classification enums
# ═══════════════════════════════════════════════════════════


class ProtocolType(str, Enum):
    WIFI_BEACON   = "WIFI_BEACON"    # ASTM F3411 Wi-Fi Beacon
    BLE_ADVB      = "BLE_ADVB"       # Bluetooth 5.0 Extended Advertising
    NR_BROADCAST  = "NR_BROADCAST"   # 4G/5G NR Sidelink Broadcast
    LORAWAN       = "LORAWAN"        # LoRaWAN uplink
    ADS_B         = "ADS_B"          # ADS-B (manned aviation, for cross-ref)


class ObjectClass(str, Enum):
    DRONE_MULTIROTOR = "DRONE_MULTIROTOR"
    DRONE_FIXED_WING = "DRONE_FIXED_WING"
    BIRD             = "BIRD"
    CLUTTER          = "CLUTTER"
    UNKNOWN          = "UNKNOWN"


# ═══════════════════════════════════════════════════════════
# Unified configuration
# ═══════════════════════════════════════════════════════════


@dataclass
class FusionConfig:
    """Centralised hyperparameter configuration for the RID fusion engine.

    All tunable parameters are collected here rather than scattered
    across module constructors.  Pass a single ``FusionConfig`` to
    :class:`RIDFusionEngine` or :class:`FusionEncoder` instead of
    individual kwargs.

    Attributes:
        d_model:         embedding dimension for all token vectors
        n_heads:         number of attention heads in cross-modal fusion
        spatial_dim:     dimension of the spatial (lat/lon/alt) encoding
        temporal_dim:    dimension of the temporal encoding
        signal_dim:      dimension of the signal-quality (RSSI/SNR) encoding
        protocol_dim:    dimension of the protocol one-hot embedding
        context_dim:     dimension of the weather context encoding
        seed:            global random seed for reproducibility
        inner_lr:        MAML inner-loop learning rate
        outer_lr:        MAML outer-loop (meta) learning rate
        n_inner_steps:   MAML inner-loop gradient steps
        time_window_s:   deduplication time window (seconds)
        max_position_diff_m:  maximum allowed position discrepancy in dedup (metres)
    """
    d_model: int = 128
    n_heads: int = 4
    spatial_dim: int = 32
    temporal_dim: int = 16
    signal_dim: int = 8
    protocol_dim: int = 4
    context_dim: int = 8
    seed: int = 42
    inner_lr: float = 0.01
    outer_lr: float = 0.001
    n_inner_steps: int = 5
    time_window_s: float = 0.5
    max_position_diff_m: float = 50.0


def new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class SpatioTemporalToken:
    """
    Unified token: every RID observation, regardless of protocol,
    is mapped to this structure before entering the fusion encoder.
    
    Fields are based on the mandatory fields in ASTM F3411-22a
    (Remote ID for Unmanned Aircraft Systems).
    """
    token_id: str = field(default_factory=new_id)
    
    # ── Core identity ──
    drone_id: str = ""                  # UAS ID (serial or session ID)
    protocol: ProtocolType = ProtocolType.WIFI_BEACON
    
    # ── Spatial ──
    lat_deg: float = 0.0                # WGS-84 latitude
    lon_deg: float = 0.0                # WGS-84 longitude
    alt_m: float = 0.0                  # altitude above takeoff, metres
    lat_error_m: Optional[float] = None
    lon_error_m: Optional[float] = None
    alt_error_m: Optional[float] = None
    
    # ── Velocity ──
    vx_ms: float = 0.0                  # east  velocity, m/s
    vy_ms: float = 0.0                  # north velocity, m/s
    vz_ms: float = 0.0                  # up    velocity, m/s
    
    # ── Signal ──
    rssi_dbm: float = -100.0            # received signal strength, dBm
    snr_db: float = 0.0                 # signal-to-noise ratio, dB
    timestamp_utc: float = 0.0          # Unix timestamp (seconds)
    
    # ── Protocol-specific payload (optional) ──
    protocol_payload: dict = field(default_factory=dict)
    
    # ── Weather context (injected by context encoder) ──
    wind_speed_ms: float = 0.0
    precipitation_mmh: float = 0.0
    visibility_m: float = 10000.0
    
    # ── GeoSOT grid (injected by spatial encoder) ──
    geosot_grid: str = ""


@dataclass
class TokenSequence:
    """Time-ordered sequence of tokens for one drone."""
    drone_id: str
    tokens: list[SpatioTemporalToken] = field(default_factory=list)
    
    @property
    def duration_s(self) -> float:
        if len(self.tokens) < 2:
            return 0.0
        return self.tokens[-1].timestamp_utc - self.tokens[0].timestamp_utc


@dataclass
class FusedToken:
    """
    Output of the cross-modal attention fusion.
    This is a protocol-agnostic semantic representation
    that can be consumed directly by a downstream LLM.
    """
    drone_id: str
    timestamp_utc: float
    embedding: list[float] = field(default_factory=list)  # ℝ^d vector
    source_protocols: list[ProtocolType] = field(default_factory=list)
    anomaly_score: float = 0.0
    risk_level: float = 0.0


@dataclass
class Detection:
    """A single sensor's report (for back-compat with classical trackers)."""
    detection_id: str = field(default_factory=new_id)
    token: SpatioTemporalToken = field(default_factory=SpatioTemporalToken)
    sensor_id: str = ""
    object_class: ObjectClass = ObjectClass.UNKNOWN
    classification_confidence: float = 0.0
