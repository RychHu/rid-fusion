"""Regression tests for the v0.4 source pipeline and desktop API."""

from pathlib import Path
from types import SimpleNamespace

from rid_fusion.desktop_api import file_analysis, fusion, multi_fusion, selftest


def base_args(**updates):
    values = dict(
        drone="TEST", lat=30.5728, lon=104.0668, alt=120.0,
        duration=2.0, seed=123, protocols="all", dt=1.0,
        speed=8.0, heading=45.0, wind=0.0, precipitation=0.0,
        visibility=10000.0, count=3, spacing_m=80.0,
        altitude_step_m=5.0, heading_step_deg=30.0,
    )
    values.update(updates)
    return SimpleNamespace(**values)


def test_single_target_pipeline_is_reproducible():
    first = fusion(base_args())
    second = fusion(base_args())
    assert first["stats"] == second["stats"]
    assert first["states"] == second["states"]
    assert first["stats"]["target_count"] == 1
    assert first["stats"]["fused_states"] == 2


def test_multi_target_keeps_tracks_separate():
    result = multi_fusion(base_args(seed=321))
    assert result["stats"]["target_count"] == 3
    assert len({state["track_key"] for state in result["states"]}) == 3


def test_sample_import_accepts_all_rows():
    path = Path(__file__).parents[1] / "sample_data" / "observations_example.csv"
    result = file_analysis(SimpleNamespace(path=str(path), bucket=1.0))
    assert result["import"]["accepted_rows"] == 9
    assert result["import"]["rejected_rows"] == 0
    assert result["state_count"] == 3


def test_source_selftest_passes():
    result = selftest(None)
    assert result["ok"] is True
    assert result["tests"] == 5
