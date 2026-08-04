"""Core data models used by the RID fusion pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import hashlib
import json
import math
import uuid


class ProtocolType(str, Enum):
    WIFI_BEACON = "WIFI_BEACON"
    BLE_ADVB = "BLE_ADVB"
    NR_BROADCAST = "NR_BROADCAST"
    LORAWAN = "LORAWAN"
    ADS_B = "ADS_B"


class ObjectClass(str, Enum):
    DRONE_MULTIROTOR = "DRONE_MULTIROTOR"
    DRONE_FIXED_WING = "DRONE_FIXED_WING"
    BIRD = "BIRD"
    CLUTTER = "CLUTTER"
    UNKNOWN = "UNKNOWN"


@dataclass
class FusionConfig:
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
    association_time_window_s: float = 0.75
    association_gate_sigma: float = 4.0
    reorder_buffer_s: float = 1.0
    min_variance: float = 0.0001

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.n_heads <= 0:
            raise ValueError("d_model and n_heads must be positive")
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        for name in (
            "time_window_s",
            "max_position_diff_m",
            "association_time_window_s",
            "association_gate_sigma",
            "min_variance",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def digest(self) -> str:
        payload = json.dumps(self.__dict__, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def new_id() -> str:
    return str(uuid.uuid4())


@dataclass
class SpatioTemporalToken:
    token_id: str = field(default_factory=new_id)
    drone_id: str = ""
    protocol: ProtocolType = ProtocolType.WIFI_BEACON
    lat_deg: float = 0.0
    lon_deg: float = 0.0
    alt_m: float = 0.0
    lat_error_m: Optional[float] = None
    lon_error_m: Optional[float] = None
    alt_error_m: Optional[float] = None
    vx_ms: float = 0.0
    vy_ms: float = 0.0
    vz_ms: float = 0.0
    rssi_dbm: float = -100.0
    snr_db: float = 0.0
    timestamp_utc: float = 0.0
    protocol_payload: dict = field(default_factory=dict)
    wind_speed_ms: float = 0.0
    precipitation_mmh: float = 0.0
    visibility_m: float = 10000.0
    geosot_grid: str = ""
    field_validity: dict[str, bool] = field(default_factory=dict)
    receiver_id: str = ""
    raw_record_digest: str = ""

    def is_valid(self, field_name: str) -> bool:
        return self.field_validity.get(field_name, True)


@dataclass
class TokenSequence:
    drone_id: str
    tokens: list[SpatioTemporalToken] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        if len(self.tokens) < 2:
            return 0.0
        return self.tokens[-1].timestamp_utc - self.tokens[0].timestamp_utc


@dataclass
class FusedToken:
    drone_id: str
    timestamp_utc: float
    embedding: list[float] = field(default_factory=list)
    source_protocols: list[ProtocolType] = field(default_factory=list)
    anomaly_score: float = 0.0
    risk_level: float = 0.0


@dataclass
class Detection:
    detection_id: str = field(default_factory=new_id)
    token: SpatioTemporalToken = field(default_factory=SpatioTemporalToken)
    sensor_id: str = ""
    object_class: ObjectClass = ObjectClass.UNKNOWN
    classification_confidence: float = 0.0


@dataclass
class RIDObservation:
    observation_id: str = field(default_factory=new_id)
    uas_id: str = ""
    protocol: ProtocolType = ProtocolType.WIFI_BEACON
    transport_type: str = ""
    receiver_id: str = ""
    source_timestamp_utc: float = 0.0
    receive_timestamp_utc: float = 0.0
    lat_deg: Optional[float] = None
    lon_deg: Optional[float] = None
    alt_m: Optional[float] = None
    vx_ms: Optional[float] = None
    vy_ms: Optional[float] = None
    vz_ms: Optional[float] = None
    rssi_dbm: Optional[float] = None
    snr_db: Optional[float] = None
    horizontal_variance_m2: Optional[float] = None
    vertical_variance_m2: Optional[float] = None
    velocity_variance_ms2: Optional[float] = None
    altitude_reference: str = "WGS84_ELLIPSOID"
    valid_fields: dict[str, bool] = field(default_factory=dict)
    parse_issues: list[str] = field(default_factory=list)
    source_digest: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_token(cls, token: SpatioTemporalToken) -> "RIDObservation":
        def valid(name: str) -> bool:
            return token.field_validity.get(name, True)

        horizontal_error = token.lat_error_m or token.lon_error_m
        velocity_error = token.protocol_payload.get("vel_error_std_ms")
        return cls(
            observation_id=token.token_id,
            uas_id=token.drone_id,
            protocol=token.protocol,
            transport_type=token.protocol.value,
            receiver_id=token.receiver_id,
            source_timestamp_utc=token.timestamp_utc,
            receive_timestamp_utc=token.timestamp_utc,
            lat_deg=token.lat_deg if valid("lat") else None,
            lon_deg=token.lon_deg if valid("lon") else None,
            alt_m=token.alt_m if valid("alt") else None,
            vx_ms=token.vx_ms if valid("vx") else None,
            vy_ms=token.vy_ms if valid("vy") else None,
            vz_ms=token.vz_ms if valid("vz") else None,
            rssi_dbm=token.rssi_dbm if valid("rssi") else None,
            snr_db=token.snr_db if valid("snr") else None,
            horizontal_variance_m2=(horizontal_error or 10.0) ** 2,
            vertical_variance_m2=(token.alt_error_m or 15.0) ** 2,
            velocity_variance_ms2=(velocity_error or 1.0) ** 2,
            valid_fields=dict(token.field_validity),
            source_digest=token.raw_record_digest,
            metadata={"protocol_payload": token.protocol_payload},
        )


@dataclass
class ObservationGroup:
    group_id: str = field(default_factory=new_id)
    track_key: str = ""
    window_start_utc: float = 0.0
    window_end_utc: float = 0.0
    observations: list[RIDObservation] = field(default_factory=list)
    association_scores: dict[str, float] = field(default_factory=dict)
    identity_conflict: bool = False
    conflict_reasons: list[str] = field(default_factory=list)
    association_method_version: str = "gated-window/v1"


@dataclass
class FusedState:
    track_key: str
    timestamp_utc: float
    lat_deg: Optional[float] = None
    lon_deg: Optional[float] = None
    alt_m: Optional[float] = None
    vx_ms: Optional[float] = None
    vy_ms: Optional[float] = None
    vz_ms: Optional[float] = None
    covariance_diag: list[Optional[float]] = field(default_factory=list)
    protocol_weights: dict[str, float] = field(default_factory=dict)
    used_observation_ids: list[str] = field(default_factory=list)
    excluded_observations: dict[str, str] = field(default_factory=dict)
    source_protocols: list[ProtocolType] = field(default_factory=list)
    identity_conflict: bool = False
    anomaly_score: float = 0.0
    algorithm_version: str = "covariance-weighted/v1"
    config_digest: str = ""
    semantic_embedding: list[float] = field(default_factory=list)

    @property
    def horizontal_std_m(self) -> Optional[float]:
        if len(self.covariance_diag) < 2:
            return None
        x, y = self.covariance_diag[:2]
        if x is None or y is None:
            return None
        return math.sqrt(max(0.0, x + y))
