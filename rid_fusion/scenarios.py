"""Validated built-in scenarios used by the CLI and desktop UI."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field


@dataclass
class TargetScenario:
    drone_id: str
    start_lat: float
    start_lon: float
    start_alt: float
    speed_ms: float = 8.0
    heading_deg: float = 45.0

    def validate(self) -> None:
        if not self.drone_id or len(self.drone_id) > 64:
            raise ValueError("drone_id must contain 1-64 characters")
        if not -90 <= self.start_lat <= 90 or not -180 <= self.start_lon <= 180:
            raise ValueError("invalid latitude or longitude")
        if not -500 <= self.start_alt <= 10_000 or not 0 <= self.speed_ms <= 150:
            raise ValueError("invalid altitude or speed")
        if not 0 <= self.heading_deg <= 360:
            raise ValueError("invalid heading")


@dataclass
class SimulationScenario:
    name: str = "自定义场景"
    duration_s: float = 30.0
    dt_s: float = 1.0
    protocols: str = "wifi_ble_nr"
    seed: int = 42
    wind_speed_ms: float = 0.0
    precipitation_mmh: float = 0.0
    visibility_m: float = 10000.0
    targets: list[TargetScenario] = field(default_factory=list)

    def validate(self) -> None:
        if not 1 <= self.duration_s <= 3600 or not 0.1 <= self.dt_s <= 60:
            raise ValueError("invalid duration or sample interval")
        if not 0 <= self.seed <= 4_294_967_295:
            raise ValueError("invalid seed")
        if not self.targets:
            raise ValueError("scenario must contain at least one target")
        for target in self.targets:
            target.validate()

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS = {
    "chengdu_basic": SimulationScenario(
        "成都三来源基础场景", targets=[TargetScenario("UAS-CD-001", 30.5728, 104.0668, 120.0)]
    ),
    "shenzhen_fast": SimulationScenario(
        "深圳快速飞行场景", 45, targets=[TargetScenario("UAS-SZ-001", 22.5431, 114.0579, 100.0, 18.0, 90.0)]
    ),
    "crossing_targets": SimulationScenario(
        "双目标交汇场景", 60, targets=[
            TargetScenario("UAS-A", 30.5724, 104.0664, 110.0, 10.0, 45.0),
            TargetScenario("UAS-B", 30.576, 104.07, 125.0, 10.0, 225.0),
        ]
    ),
    "single_source": SimulationScenario(
        "Wi-Fi单来源场景", protocols="wifi",
        targets=[TargetScenario("UAS-WIFI-001", 30.5728, 104.0668, 80.0, 6.0, 20.0)],
    ),
    "poor_visibility": SimulationScenario(
        "低能见度多来源场景", wind_speed_ms=12.0, precipitation_mmh=20.0,
        visibility_m=800.0,
        targets=[TargetScenario("UAS-WX-001", 30.5728, 104.0668, 100.0, 7.0, 120.0)],
    ),
}


def get_preset(key: str) -> SimulationScenario:
    if key not in PRESETS:
        raise KeyError(f"unknown preset: {key}")
    return deepcopy(PRESETS[key])


def list_presets() -> list[dict]:
    return [{"key": key, **scenario.to_dict()} for key, scenario in PRESETS.items()]
