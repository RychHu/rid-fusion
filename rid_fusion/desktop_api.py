"""JSON command API consumed by the native WPF desktop shell."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
import sys
from .comparison import compare_algorithms
from .fusion import RIDFusionEngine
from .importers import load_observation_file
from .meta_learner import simulate_meta_learning_demo
from .models import FusedState, ProtocolType
from .replay import ReplaySession
from .reporting import export_csv, export_json, export_markdown
from .scenarios import PRESETS, get_preset, list_presets


PROTOCOL_OPTIONS = {
    "wifi_ble": [ProtocolType.WIFI_BEACON, ProtocolType.BLE_ADVB],
    "wifi_ble_nr": [ProtocolType.WIFI_BEACON, ProtocolType.BLE_ADVB, ProtocolType.NR_BROADCAST],
    "wifi": [ProtocolType.WIFI_BEACON],
    "all": [ProtocolType.WIFI_BEACON, ProtocolType.BLE_ADVB, ProtocolType.NR_BROADCAST, ProtocolType.LORAWAN],
}


def validate_fusion_args(args) -> None:
    checks = (
        ("lat", -90, 90), ("lon", -180, 180), ("alt", -500, 10_000),
        ("duration", 1, 3600), ("dt", 0.1, 60), ("speed", 0, 150),
        ("heading", 0, 360), ("wind", 0, 100), ("precipitation", 0, 500),
        ("visibility", 1, 100_000),
    )
    if not args.drone or len(args.drone) > 64:
        raise ValueError("目标ID必须为1-64个字符")
    for name, low, high in checks:
        value = getattr(args, name)
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{name} must be between {low} and {high}")
    if not 0 <= args.seed <= 4_294_967_295:
        raise ValueError("seed must be between 0 and 4294967295")


def _state_payload(state: FusedState) -> dict:
    return {
        "track_key": state.track_key,
        "timestamp": state.timestamp_utc,
        "lat": state.lat_deg,
        "lon": state.lon_deg,
        "alt": state.alt_m,
        "vx": state.vx_ms,
        "vy": state.vy_ms,
        "vz": state.vz_ms,
        "uncertainty_m": state.horizontal_std_m,
        "protocol_weights": state.protocol_weights,
        "evidence_count": len(state.used_observation_ids),
        "identity_conflict": state.identity_conflict,
    }


def _result_payload(result: dict) -> dict:
    states = result.get("states", [])
    payload = {
        "version": "0.4.0",
        "stats": result.get("stats", {}),
        "average_horizontal_std_m": (
            sum(s.horizontal_std_m for s in states if s.horizontal_std_m is not None)
            / max(sum(s.horizontal_std_m is not None for s in states), 1)
        ),
        "latest_protocol_weights": states[-1].protocol_weights if states else {},
        "latest_evidence_count": len(states[-1].used_observation_ids) if states else 0,
        "trajectory": [
            {
                "drone_id": p["drone_id"], "timestamp": p["timestamp_utc"],
                "lat": p["lat_deg"], "lon": p["lon_deg"], "alt": p["alt_m"],
            }
            for p in result.get("trajectory", [])
        ],
        "states": [_state_payload(state) for state in states],
        "anomalies": [event.to_dict() if hasattr(event, "to_dict") else event for event in result.get("anomalies", [])],
    }
    if "targets" in result:
        payload["targets"] = result["targets"]
    return payload


def _engine(args) -> RIDFusionEngine:
    return RIDFusionEngine(PROTOCOL_OPTIONS[args.protocols], seed=args.seed, enable_dedup=False)


def fusion(args) -> dict:
    validate_fusion_args(args)
    result = _engine(args).run_on_trajectory(
        args.drone, args.lat, args.lon, args.alt, args.duration, args.dt,
        args.speed, args.heading, args.wind, args.precipitation, args.visibility,
    )
    return _result_payload(result)


def presets(args) -> dict:
    if not args.key:
        return {"presets": list_presets()}
    scenario = get_preset(args.key)
    scenario.validate()
    engine = RIDFusionEngine(PROTOCOL_OPTIONS[scenario.protocols], seed=scenario.seed, enable_dedup=False)
    targets = [asdict(target) for target in scenario.targets]
    result = engine.run_multi_target(
        targets, scenario.duration_s, scenario.dt_s, scenario.wind_speed_ms,
        scenario.precipitation_mmh, scenario.visibility_m,
    )
    payload = _result_payload(result)
    payload["scenario"] = scenario.to_dict()
    return payload


def multi_fusion(args) -> dict:
    validate_fusion_args(args)
    if not 1 <= args.count <= 20:
        raise ValueError("count must be between 1 and 20")
    if not 0 <= args.spacing_m <= 100_000:
        raise ValueError("spacing-m must be between 0 and 100000")
    targets = []
    for index in range(args.count):
        angle = 2 * math.pi * index / max(args.count, 1)
        north = args.spacing_m * math.sin(angle)
        east = args.spacing_m * math.cos(angle)
        targets.append({
            "drone_id": f"{args.drone}-{index + 1:02d}",
            "start_lat": args.lat + north / 111_320.0,
            "start_lon": args.lon + east / (111_320.0 * max(math.cos(math.radians(args.lat)), 1e-3)),
            "start_alt": args.alt + index * args.altitude_step_m,
            "speed_ms": args.speed,
            "heading_deg": (args.heading + index * args.heading_step_deg) % 360,
        })
    result = _engine(args).run_multi_target(
        targets, args.duration, args.dt, args.wind, args.precipitation, args.visibility,
    )
    return _result_payload(result)


def file_analysis(args) -> dict:
    imported = load_observation_file(args.path)
    engine = RIDFusionEngine(list(ProtocolType), seed=42, enable_dedup=False)
    processed = engine.process_observations(imported.observations)
    replay = ReplaySession(imported.observations, args.bucket)
    return {
        "import": imported.summary(),
        "replay": replay.summary(),
        "frames": [
            {
                "index": frame.index, "timestamp": frame.timestamp_utc,
                "observation_count": len(frame.observations),
                "targets": sorted({o.uas_id for o in frame.observations}),
                "protocols": sorted({o.protocol.value for o in frame.observations}),
            }
            for frame in replay.frames
        ],
        "state_count": len(processed["states"]),
        "anomalies": [event.to_dict() for event in processed["anomalies"]],
        "states": [_state_payload(state) for state in processed["states"]],
    }


def comparison(args) -> dict:
    validate_fusion_args(args)
    result = _engine(args).run_on_trajectory(
        args.drone, args.lat, args.lon, args.alt, args.duration, args.dt,
        args.speed, args.heading, args.wind, args.precipitation, args.visibility,
    )
    return {"comparison": compare_algorithms(result["groups"], result["trajectory"])}


def export_report(args) -> dict:
    validate_fusion_args(args)
    result = _engine(args).run_on_trajectory(
        args.drone, args.lat, args.lon, args.alt, args.duration, args.dt,
        args.speed, args.heading, args.wind, args.precipitation, args.visibility,
    )
    compare = compare_algorithms(result["groups"], result["trajectory"])
    anomalies = [event.to_dict() for event in result["anomalies"]]
    metadata = {"version": "0.4.0", "drone": args.drone, "seed": args.seed, "protocols": args.protocols, "config_digest": result["stats"]["config_digest"]}
    if args.format == "json":
        output = export_json(args.output, result["states"], metadata, anomalies, compare)
    elif args.format == "csv":
        output = export_csv(args.output, result["states"])
    else:
        output = export_markdown(args.output, result["states"], metadata, anomalies, compare)
    return {"path": output, "format": args.format, "state_count": len(result["states"])}


def adaptation(args) -> dict:
    return simulate_meta_learning_demo(args.seed)


def selftest(_args) -> dict:
    checks = []
    try:
        for key in PRESETS:
            get_preset(key).validate()
        checks.append("✓ 场景模板校验通过")
        engine = RIDFusionEngine(PROTOCOL_OPTIONS["wifi_ble_nr"], seed=7, enable_dedup=False)
        single = engine.run_on_trajectory("SELFTEST", 30.57, 104.07, 100, 2)
        assert single["states"]
        checks.append("✓ 单目标融合通过")
        multi = engine.run_multi_target([
            {"drone_id": "A", "start_lat": 30.57, "start_lon": 104.07, "start_alt": 100},
            {"drone_id": "B", "start_lat": 30.58, "start_lon": 104.08, "start_alt": 110},
        ], 2)
        assert len({state.track_key for state in multi["states"]}) == 2
        checks.append("✓ 多目标融合通过")
        assert "covariance_weighted" in compare_algorithms(single["groups"], single["trajectory"])
        checks.append("✓ 算法对比通过")
        checks.append("✓ Python源码运行环境通过")
    except Exception as exc:
        return {"ok": False, "output": "\n".join(checks + [f"✗ {exc}"]), "suites": 1, "tests": len(checks) + 1}
    return {"ok": True, "output": "\n".join(checks), "suites": 1, "tests": len(checks)}


def _add_base(parser) -> None:
    parser.add_argument("--drone", default="UAS-A17F")
    parser.add_argument("--lat", type=float, default=30.5728)
    parser.add_argument("--lon", type=float, default=104.0668)
    parser.add_argument("--alt", type=float, default=120.0)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--protocols", choices=PROTOCOL_OPTIONS, default="wifi_ble_nr")


def _add_advanced(parser) -> None:
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=8.0)
    parser.add_argument("--heading", type=float, default=45.0)
    parser.add_argument("--wind", type=float, default=0.0)
    parser.add_argument("--precipitation", type=float, default=0.0)
    parser.add_argument("--visibility", type=float, default=10000.0)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rid-fusion")
    commands = parser.add_subparsers(dest="command", required=True)
    for name, function in (("fusion", fusion), ("compare", comparison), ("export", export_report)):
        sub = commands.add_parser(name)
        _add_base(sub); _add_advanced(sub); sub.set_defaults(func=function)
    sub = commands.add_parser("multi")
    _add_base(sub); _add_advanced(sub)
    sub.add_argument("--count", type=int, default=3)
    sub.add_argument("--spacing-m", type=float, default=80.0)
    sub.add_argument("--altitude-step-m", type=float, default=5.0)
    sub.add_argument("--heading-step-deg", type=float, default=30.0)
    sub.set_defaults(func=multi_fusion)
    sub = commands.add_parser("import")
    sub.add_argument("--path", required=True); sub.add_argument("--bucket", type=float, default=1.0)
    sub.set_defaults(func=file_analysis)
    sub = commands.add_parser("presets")
    sub.add_argument("--key", choices=PRESETS); sub.add_argument("--describe", action="store_true")
    sub.set_defaults(func=presets)
    sub = commands.add_parser("adaptation")
    sub.add_argument("--seed", type=int, default=42); sub.set_defaults(func=adaptation)
    commands.add_parser("selftest").set_defaults(func=selftest)
    export_parser = commands.choices["export"]
    export_parser.add_argument("--output", required=True)
    export_parser.add_argument("--format", choices=("json", "csv", "md"), default="md")
    return parser


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        args = build_parser().parse_args()
        print(json.dumps({"ok": True, "data": args.func(args)}, ensure_ascii=False, default=str))
    except Exception as exc:
        print(json.dumps({"ok": False, "error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
