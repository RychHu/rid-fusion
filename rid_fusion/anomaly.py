"""Explainable anomaly checks for associated RID observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from .models import ObservationGroup, RIDObservation
from .training.metrics import horizontal_error_m


@dataclass
class AnomalyEvent:
    code: str
    severity: str
    target_id: str
    timestamp_utc: float
    message: str
    observation_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def detect_anomalies(
    observations: list[RIDObservation],
    groups: list[ObservationGroup] | None = None,
    max_speed_ms: float = 80.0,
    max_time_regression_s: float = 0.01,
) -> list[AnomalyEvent]:
    events: list[AnomalyEvent] = []
    for group in groups or []:
        if group.identity_conflict:
            events.append(AnomalyEvent(
                "IDENTITY_CONFLICT", "high", group.track_key, group.window_end_utc,
                "同一目标ID在相近时间报告了空间上不相容的位置。",
                [o.observation_id for o in group.observations],
            ))

    by_target: dict[str, list[RIDObservation]] = {}
    seen_keys: set[tuple] = set()
    for observation in observations:
        by_target.setdefault(observation.uas_id, []).append(observation)
        replay_key = (observation.uas_id, observation.source_timestamp_utc, observation.source_digest)
        if observation.source_digest and replay_key in seen_keys:
            events.append(AnomalyEvent(
                "POSSIBLE_REPLAY", "medium", observation.uas_id,
                observation.source_timestamp_utc,
                "检测到相同时间戳和相同来源摘要的重复观测。",
                [observation.observation_id],
            ))
        seen_keys.add(replay_key)
        components = [v for v in (observation.vx_ms, observation.vy_ms, observation.vz_ms) if v is not None]
        speed = math.sqrt(sum(v * v for v in components))
        if speed > max_speed_ms:
            events.append(AnomalyEvent(
                "EXCESSIVE_REPORTED_SPEED", "high", observation.uas_id,
                observation.source_timestamp_utc,
                f"报告速度{speed:.1f}m/s超过配置阈值{max_speed_ms:.1f}m/s。",
                [observation.observation_id],
            ))

    for target, items in by_target.items():
        ordered = sorted(items, key=lambda o: (o.receive_timestamp_utc, o.observation_id))
        for previous, current in zip(ordered, ordered[1:]):
            if current.source_timestamp_utc < previous.source_timestamp_utc - max_time_regression_s:
                events.append(AnomalyEvent(
                    "TIMESTAMP_REGRESSION", "medium", target,
                    current.source_timestamp_utc, "源时间戳相对接收顺序发生倒退。",
                    [previous.observation_id, current.observation_id],
                ))
            dt = current.source_timestamp_utc - previous.source_timestamp_utc
            if dt <= 0 or None in (previous.lat_deg, previous.lon_deg, current.lat_deg, current.lon_deg):
                continue
            implied = horizontal_error_m(previous.lat_deg, previous.lon_deg, current.lat_deg, current.lon_deg) / dt
            if implied > max_speed_ms * 1.5:
                events.append(AnomalyEvent(
                    "POSITION_JUMP", "high", target, current.source_timestamp_utc,
                    f"相邻位置推算速度{implied:.1f}m/s，超过跳变阈值。",
                    [previous.observation_id, current.observation_id],
                ))
    rank = {"high": 0, "medium": 1, "low": 2}
    return sorted(events, key=lambda event: (rank[event.severity], event.timestamp_utc, event.code))
