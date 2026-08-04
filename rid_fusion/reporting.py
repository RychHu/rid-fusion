"""Export fused evidence as JSON, CSV or a readable Markdown report."""

from __future__ import annotations

from dataclasses import asdict
import csv
import json
from pathlib import Path
from .models import FusedState


def _state_dict(state: FusedState) -> dict:
    data = asdict(state)
    data["source_protocols"] = [protocol.value for protocol in state.source_protocols]
    data["horizontal_std_m"] = state.horizontal_std_m
    return data


def export_json(
    path: str | Path,
    states: list[FusedState],
    metadata: dict,
    anomalies: list[dict] | None = None,
    comparison: dict | None = None,
) -> str:
    path = str(path)
    Path(path).write_text(json.dumps({
        "metadata": metadata,
        "states": [_state_dict(state) for state in states],
        "anomalies": anomalies or [],
        "comparison": comparison,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def export_csv(path: str | Path, states: list[FusedState]) -> str:
    path = str(path)
    columns = ["track_key", "timestamp_utc", "lat_deg", "lon_deg", "alt_m", "vx_ms", "vy_ms", "vz_ms", "horizontal_std_m", "source_protocols", "evidence_count", "identity_conflict"]
    with open(path, "w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for state in states:
            writer.writerow({
                "track_key": state.track_key, "timestamp_utc": state.timestamp_utc,
                "lat_deg": state.lat_deg, "lon_deg": state.lon_deg, "alt_m": state.alt_m,
                "vx_ms": state.vx_ms, "vy_ms": state.vy_ms, "vz_ms": state.vz_ms,
                "horizontal_std_m": state.horizontal_std_m,
                "source_protocols": ",".join(p.value for p in state.source_protocols),
                "evidence_count": len(state.used_observation_ids),
                "identity_conflict": state.identity_conflict,
            })
    return path


def export_markdown(
    path: str | Path,
    states: list[FusedState],
    metadata: dict,
    anomalies: list[dict] | None = None,
    comparison: dict | None = None,
) -> str:
    lines = ["# RID Fusion 分析报告", "", "## 实验信息", ""]
    lines.extend(f"- {key}: {value}" for key, value in metadata.items())
    lines += ["", "## 融合状态", "", "|目标|时间|纬度|经度|高度(m)|水平不确定度(m)|证据数|", "|---|---:|---:|---:|---:|---:|---:|"]
    for state in states:
        lines.append(f"|{state.track_key}|{state.timestamp_utc:.3f}|{state.lat_deg}|{state.lon_deg}|{state.alt_m}|{state.horizontal_std_m}|{len(state.used_observation_ids)}|")
    lines += ["", "## 异常", ""]
    lines.extend(f"- [{item.get('severity')}] {item.get('message')}" for item in anomalies or [])
    if not anomalies:
        lines.append("- 未检测到已定义规则覆盖的异常。")
    if comparison:
        lines += ["", "## 算法对比", "", "```json", json.dumps(comparison, ensure_ascii=False, indent=2), "```"]
    path = str(path)
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
