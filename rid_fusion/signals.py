"""
signals.py — Multi-protocol RID signal simulator.

Generates synthetic Remote ID signals for:
  - ASTM F3411 Wi-Fi Beacon (802.11)
  - Bluetooth 5.0 BLE Extended Advertising
  - 4G/5G NR Sidelink Broadcast

Each protocol has its own noise profile, detection probability,
field availability, and error characteristics — mirroring real-world
deployment heterogeneity.
"""

from __future__ import annotations
import logging
import math
import numpy as np
from typing import Optional
from rid_fusion.models import SpatioTemporalToken, ProtocolType

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# Protocol-specific noise profiles
# ═══════════════════════════════════════════════════════════

PROTOCOL_PROFILES = {
    ProtocolType.WIFI_BEACON: {
        "range_m": 1000,
        "pos_error_std_m": 3.0,          # GPS-level accuracy
        "alt_error_std_m": 5.0,
        "vel_error_std_ms": 0.3,
        "rssi_range_dbm": (-70, -30),
        "snr_range_db": (10, 30),
        "detection_prob": 0.92,
        "has_drone_id": True,
        "has_altitude": True,
        "has_velocity": True,
        "fields_available": ["drone_id", "lat", "lon", "alt", "vx", "vy", "vz", "rssi"],
    },
    ProtocolType.BLE_ADVB: {
        "range_m": 300,
        "pos_error_std_m": 5.0,          # BLE positioning less accurate
        "alt_error_std_m": 8.0,
        "vel_error_std_ms": 0.5,
        "rssi_range_dbm": (-85, -45),
        "snr_range_db": (5, 20),
        "detection_prob": 0.85,          # BLE more likely to miss
        "has_drone_id": True,
        "has_altitude": True,
        "has_velocity": False,           # BLE ADVB typically no velocity
        "fields_available": ["drone_id", "lat", "lon", "alt", "rssi"],
    },
    ProtocolType.NR_BROADCAST: {
        "range_m": 3000,
        "pos_error_std_m": 2.0,          # 5G positioning very accurate
        "alt_error_std_m": 3.0,
        "vel_error_std_ms": 0.2,
        "rssi_range_dbm": (-90, -50),
        "snr_range_db": (8, 25),
        "detection_prob": 0.95,
        "has_drone_id": True,
        "has_altitude": True,
        "has_velocity": True,
        "fields_available": ["drone_id", "lat", "lon", "alt", "vx", "vy", "vz", "rssi", "snr"],
    },
    ProtocolType.LORAWAN: {
        "range_m": 5000,
        "pos_error_std_m": 20.0,         # LoRaWAN TDOA coarse
        "alt_error_std_m": 25.0,
        "vel_error_std_ms": 1.0,
        "rssi_range_dbm": (-120, -80),
        "snr_range_db": (-10, 10),
        "detection_prob": 0.70,
        "has_drone_id": True,
        "has_altitude": False,
        "has_velocity": False,
        "fields_available": ["drone_id", "lat", "lon", "rssi"],
    },
}


# ═══════════════════════════════════════════════════════════
# Signal generator
# ═══════════════════════════════════════════════════════════

class RIDSignalSimulator:
    """
    Generates synthetic RID tokens for a drone flying a given trajectory.

    Each call to .observe() produces one token per active protocol,
    with protocol-specific noise and detection failures.
    """

    def __init__(
        self,
        protocols: list[ProtocolType],
        seed: int = 42,
    ):
        self.protocols = protocols
        self.rng = np.random.default_rng(seed)

    def observe(
        self,
        drone_id: str,
        timestamp_utc: float,
        lat_deg: float,
        lon_deg: float,
        alt_m: float,
        vx_ms: float = 0.0,
        vy_ms: float = 0.0,
        vz_ms: float = 0.0,
        geosot_grid: str = "",
        wind_speed_ms: float = 0.0,
        precipitation_mmh: float = 0.0,
        visibility_m: float = 10000.0,
    ) -> list[SpatioTemporalToken]:
        """
        Generate tokens for all active protocols at one timestep.

        Returns:
            list of SpatioTemporalToken (one per protocol that detected the drone)
        """
        tokens: list[SpatioTemporalToken] = []

        for proto in self.protocols:
            profile = PROTOCOL_PROFILES[proto]

            # Detection check
            if self.rng.random() > profile["detection_prob"]:
                logger.debug("Missed detection: drone=%s protocol=%s ts=%.1f",
                             drone_id, proto.value, timestamp_utc)
                continue  # missed detection

            token = SpatioTemporalToken(
                drone_id=drone_id,
                protocol=proto,
                timestamp_utc=timestamp_utc,
            )

            # Apply protocol-specific noise
            # Latitude: 1° ≈ 111,320 m (nearly constant)
            # Longitude: 1° ≈ 111,320 m × cos(latitude) — correction is essential
            #            at mid-to-high latitudes (error reaches ~29% at 45°N)
            cos_lat = math.cos(math.radians(lat_deg))
            lon_m_per_deg = 111320.0 * max(cos_lat, 1e-3)

            if "lat" in profile["fields_available"]:
                token.lat_deg = lat_deg + self.rng.normal(0, profile["pos_error_std_m"] / 111320)
                token.lat_error_m = profile["pos_error_std_m"]
            if "lon" in profile["fields_available"]:
                token.lon_deg = lon_deg + self.rng.normal(0, profile["pos_error_std_m"] / lon_m_per_deg)
                token.lon_error_m = profile["pos_error_std_m"]
            if "alt" in profile["fields_available"]:
                token.alt_m = alt_m + self.rng.normal(0, profile["alt_error_std_m"])
                token.alt_error_m = profile["alt_error_std_m"]
            if "vx" in profile["fields_available"]:
                token.vx_ms = vx_ms + self.rng.normal(0, profile["vel_error_std_ms"])
            if "vy" in profile["fields_available"]:
                token.vy_ms = vy_ms + self.rng.normal(0, profile["vel_error_std_ms"])
            if "vz" in profile["fields_available"]:
                token.vz_ms = vz_ms + self.rng.normal(0, profile["vel_error_std_ms"])

            # RSSI drops with range (simplified path-loss)
            rssi_base = self.rng.uniform(*profile["rssi_range_dbm"])
            token.rssi_dbm = rssi_base
            token.snr_db = self.rng.uniform(*profile["snr_range_db"])

            # Context
            token.geosot_grid = geosot_grid
            token.wind_speed_ms = wind_speed_ms
            token.precipitation_mmh = precipitation_mmh
            token.visibility_m = visibility_m

            # Protocol-specific payload
            token.protocol_payload = {
                "protocol": proto.value,
                "profile": profile,
                "vel_error_std_ms": profile["vel_error_std_ms"],
            }
            token.field_validity = {
                "lat": "lat" in profile["fields_available"],
                "lon": "lon" in profile["fields_available"],
                "alt": "alt" in profile["fields_available"],
                "vx": "vx" in profile["fields_available"],
                "vy": "vy" in profile["fields_available"],
                "vz": "vz" in profile["fields_available"],
                "rssi": "rssi" in profile["fields_available"],
                "snr": "snr" in profile["fields_available"],
            }
            token.receiver_id = f"SIM-{proto.value}"

            tokens.append(token)

        return tokens


def generate_drone_trajectory(
    drone_id: str,
    start_lat: float,
    start_lon: float,
    start_alt: float,
    duration_s: float = 120.0,
    dt_s: float = 1.0,
    speed_ms: float = 8.0,
    heading_deg: float = 45.0,
    rng: Optional[np.random.Generator] = None,
) -> list[dict]:
    """
    Generate a straight-line trajectory with optional random perturbations.
    Returns a list of dicts with lat, lon, alt, vx, vy, vz at each timestep.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    heading_rad = math.radians(heading_deg)
    vx = speed_ms * math.sin(heading_rad)
    vy = speed_ms * math.cos(heading_rad)

    steps = int(duration_s / dt_s)
    points = []

    for i in range(steps):
        t = i * dt_s
        # Longitude degree length varies with latitude: cos(lat) correction
        cos_lat = math.cos(math.radians(start_lat))
        lon_m_per_deg = 111320.0 * max(cos_lat, 1e-3)

        # Random perturbation
        delta_lat = rng.normal(0, 0.5) / 111320  # ~0.5 m jitter
        delta_lon = rng.normal(0, 0.5) / lon_m_per_deg
        delta_alt = rng.normal(0, 1.0)

        points.append({
            "drone_id": drone_id,
            "timestamp_utc": t,
            "lat_deg": start_lat + vx * t / 111320 + delta_lat,
            "lon_deg": start_lon + vy * t / lon_m_per_deg + delta_lon,
            "alt_m": start_alt + delta_alt,
            "vx_ms": vx + rng.normal(0, 0.1),
            "vy_ms": vy + rng.normal(0, 0.1),
            "vz_ms": rng.normal(0, 0.05),
        })

    return points


def generate_complex_trajectory(
    drone_id: str,
    waypoints: list[dict],
    dt_s: float = 1.0,
    hover_duration_s: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> list[dict]:
    """
    Generate a multi-segment trajectory with turns, hover, and variable speed.

    Unlike ``generate_drone_trajectory`` which produces a single straight line,
    this function stitches together multiple waypoints, supporting realistic
    drone flight patterns: take-off, cruise, loiter (hover), and descent.

    Each waypoint dict may contain:
        ``lat``, ``lon``, ``alt`` — target position (required)
        ``speed_ms`` — cruise speed between this and the next waypoint
        ``hover_s`` — hover duration at this waypoint before moving on

    Args:
        drone_id:        identifier for the drone
        waypoints:       list of dicts with 'lat', 'lon', 'alt' and optional
                         'speed_ms' (default 8.0), 'hover_s' (default 0)
        dt_s:            time step in seconds
        hover_duration_s: global hover at every waypoint if not overridden
        rng:             NumPy random generator for perturbation

    Returns:
        list of dicts with lat, lon, alt, vx, vy, vz at each timestep
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if len(waypoints) < 2:
        raise ValueError("At least 2 waypoints are required for a trajectory")

    points: list[dict] = []
    t_global = 0.0

    for i in range(len(waypoints) - 1):
        wp_a = waypoints[i]
        wp_b = waypoints[i + 1]

        lat_a, lon_a, alt_a = wp_a["lat"], wp_a["lon"], wp_a["alt"]
        lat_b, lon_b, alt_b = wp_b["lat"], wp_b["lon"], wp_b["alt"]
        speed = wp_b.get("speed_ms", wp_a.get("speed_ms", 8.0))

        # Distance in metres (approximate flat-earth at mid-latitudes)
        cos_lat = math.cos(math.radians((lat_a + lat_b) / 2))
        dlat_m = (lat_b - lat_a) * 111320.0
        dlon_m = (lon_b - lon_a) * 111320.0 * cos_lat
        dalt_m = alt_b - alt_a
        dist_m = math.sqrt(dlat_m * dlat_m + dlon_m * dlon_m + dalt_m * dalt_m)

        if dist_m < 1e-3:
            # Hover-in-place segment
            hover_s = wp_a.get("hover_s", hover_duration_s)
            n_steps = max(1, int(hover_s / dt_s))
            for _ in range(n_steps):
                points.append({
                    "drone_id": drone_id,
                    "timestamp_utc": t_global,
                    "lat_deg": lat_a + rng.normal(0, 0.3) / 111320,
                    "lon_deg": lon_a + rng.normal(0, 0.3) / (111320 * max(cos_lat, 1e-3)),
                    "alt_m": alt_a + rng.normal(0, 0.3),
                    "vx_ms": rng.normal(0, 0.05),
                    "vy_ms": rng.normal(0, 0.05),
                    "vz_ms": rng.normal(0, 0.05),
                })
                t_global += dt_s
            continue

        # Direction unit vector
        u_lat = dlat_m / dist_m
        u_lon = dlon_m / dist_m
        u_alt = dalt_m / dist_m

        # Flight duration at cruise speed
        travel_s = dist_m / speed
        n_steps = max(1, int(travel_s / dt_s))
        actual_dt = travel_s / n_steps

        for step in range(n_steps):
            frac = step / max(n_steps - 1, 1)
            cur_lat = lat_a + dlat_m * frac / 111320.0
            cur_lon = lon_a + dlon_m * frac / (111320.0 * max(cos_lat, 1e-3))
            cur_alt = alt_a + dalt_m * frac

            # Jitter
            cur_lat += rng.normal(0, 0.5) / 111320.0
            cur_lon += rng.normal(0, 0.5) / (111320.0 * max(cos_lat, 1e-3))
            cur_alt += rng.normal(0, 1.0)

            vx = speed * u_lon + rng.normal(0, 0.1)
            vy = speed * u_lat + rng.normal(0, 0.1)
            vz = speed * u_alt + rng.normal(0, 0.05)

            points.append({
                "drone_id": drone_id,
                "timestamp_utc": t_global,
                "lat_deg": cur_lat,
                "lon_deg": cur_lon,
                "alt_m": cur_alt,
                "vx_ms": vx,
                "vy_ms": vy,
                "vz_ms": vz,
            })
            t_global += actual_dt

        # Hover at arrival waypoint
        hover_s = wp_b.get("hover_s", 0.0) or hover_duration_s
        if hover_s > 0:
            n_hover = max(1, int(hover_s / dt_s))
            for _ in range(n_hover):
                points.append({
                    "drone_id": drone_id,
                    "timestamp_utc": t_global,
                    "lat_deg": lat_b + rng.normal(0, 0.3) / 111320.0,
                    "lon_deg": lon_b + rng.normal(0, 0.3) / (111320.0 * max(cos_lat, 1e-3)),
                    "alt_m": alt_b + rng.normal(0, 0.3),
                    "vx_ms": rng.normal(0, 0.05),
                    "vy_ms": rng.normal(0, 0.05),
                    "vz_ms": rng.normal(0, 0.05),
                })
                t_global += dt_s

    return points
