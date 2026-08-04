"""CSV, JSON and JSONL import with field aliases and row-level issues."""

from __future__ import annotations

from dataclasses import dataclass, field
import csv
import hashlib
import json
from pathlib import Path
from ..models import ProtocolType, RIDObservation


PROTOCOL_ALIASES = {
    "WIFI": ProtocolType.WIFI_BEACON,
    "WIFI_BEACON": ProtocolType.WIFI_BEACON,
    "BLE": ProtocolType.BLE_ADVB,
    "BLE_ADVB": ProtocolType.BLE_ADVB,
    "NR": ProtocolType.NR_BROADCAST,
    "NR_BROADCAST": ProtocolType.NR_BROADCAST,
    "LORA": ProtocolType.LORAWAN,
    "LORAWAN": ProtocolType.LORAWAN,
    "ADS_B": ProtocolType.ADS_B,
    "ADS-B": ProtocolType.ADS_B,
}


@dataclass
class ImportResult:
    path: str
    observations: list[RIDObservation] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)
    total_rows: int = 0

    @property
    def accepted_rows(self) -> int:
        return len(self.observations)

    @property
    def rejected_rows(self) -> int:
        return self.total_rows - self.accepted_rows

    def summary(self) -> dict:
        return {
            "path": self.path,
            "total_rows": self.total_rows,
            "accepted_rows": self.accepted_rows,
            "rejected_rows": self.rejected_rows,
            "target_count": len({o.uas_id for o in self.observations}),
            "protocol_distribution": _counts(o.protocol.value for o in self.observations),
            "time_start": min((o.source_timestamp_utc for o in self.observations), default=None),
            "time_end": max((o.source_timestamp_utc for o in self.observations), default=None),
            "issues": self.issues,
        }


def _counts(values) -> dict:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _optional_float(value):
    if value is None or str(value).strip().lower() in {"", "none", "null", "nan"}:
        return None
    return float(value)


def _first(row: dict, *names):
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    return None


def _row_to_observation(row: dict, row_number: int, source_digest: str) -> RIDObservation:
    uas_id = str(_first(row, "uas_id", "drone_id", "target_id") or "").strip()
    if not uas_id:
        raise ValueError("missing uas_id/drone_id")
    protocol_text = str(_first(row, "protocol", "protocol_type", "source") or "").strip().upper()
    if protocol_text not in PROTOCOL_ALIASES:
        raise ValueError(f"unsupported protocol: {protocol_text or '<empty>'}")
    timestamp = _optional_float(_first(row, "timestamp_utc", "source_timestamp_utc", "timestamp"))
    lat = _optional_float(_first(row, "lat_deg", "lat", "latitude"))
    lon = _optional_float(_first(row, "lon_deg", "lon", "longitude"))
    if timestamp is None or lat is None or lon is None:
        raise ValueError("timestamp_utc, lat_deg and lon_deg are required")
    if not -90 <= lat <= 90 or not -180 <= lon <= 180:
        raise ValueError("latitude or longitude out of range")

    horizontal_variance = _optional_float(_first(row, "horizontal_variance_m2", "position_variance_m2"))
    vertical_variance = _optional_float(_first(row, "vertical_variance_m2", "altitude_variance_m2"))
    velocity_variance = _optional_float(_first(row, "velocity_variance_ms2"))
    for name, value in (("horizontal_variance_m2", horizontal_variance), ("vertical_variance_m2", vertical_variance), ("velocity_variance_ms2", velocity_variance)):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")

    canonical = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
    digest = hashlib.sha256((source_digest + canonical).encode("utf-8")).hexdigest()
    receive_time = _optional_float(_first(row, "receive_timestamp_utc"))
    return RIDObservation(
        observation_id=str(_first(row, "observation_id") or f"row-{row_number}-{digest[:12]}"),
        uas_id=uas_id,
        protocol=PROTOCOL_ALIASES[protocol_text],
        transport_type=str(_first(row, "transport_type") or protocol_text),
        receiver_id=str(_first(row, "receiver_id") or ""),
        source_timestamp_utc=timestamp,
        receive_timestamp_utc=timestamp if receive_time is None else receive_time,
        lat_deg=lat, lon_deg=lon,
        alt_m=_optional_float(_first(row, "alt_m", "alt", "altitude")),
        vx_ms=_optional_float(_first(row, "vx_ms")),
        vy_ms=_optional_float(_first(row, "vy_ms")),
        vz_ms=_optional_float(_first(row, "vz_ms")),
        rssi_dbm=_optional_float(_first(row, "rssi_dbm", "rssi")),
        snr_db=_optional_float(_first(row, "snr_db", "snr")),
        horizontal_variance_m2=horizontal_variance or 100.0,
        vertical_variance_m2=vertical_variance or 225.0,
        velocity_variance_ms2=velocity_variance or 4.0,
        valid_fields={name: _first(row, name) not in (None, "") for name in ("alt_m", "vx_ms", "vy_ms", "vz_ms", "rssi_dbm", "snr_db")},
        source_digest=digest,
        metadata={"row_number": row_number},
    )


def _read_rows(path: Path) -> list[dict]:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))
    if suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, dict):
            data = data.get("observations", data.get("rows"))
        if not isinstance(data, list):
            raise ValueError("JSON must contain an observation array")
        return data
    raise ValueError("only CSV, JSON and JSONL files are supported")


def load_observation_file(path: str | Path) -> ImportResult:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    rows = _read_rows(source)
    file_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    result = ImportResult(str(source), total_rows=len(rows))
    for number, row in enumerate(rows, start=1):
        try:
            if not isinstance(row, dict):
                raise ValueError("row is not an object")
            result.observations.append(_row_to_observation(row, number, file_digest))
        except Exception as exc:
            result.issues.append({"row": number, "message": str(exc)})
    return result
