"""
Tests for rid-fusion core modules.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from rid_fusion.models import SpatioTemporalToken, ProtocolType, FusedToken
from rid_fusion.tokenizer import Tokenizer, deduplicate_tokens
from rid_fusion.signals import RIDSignalSimulator, generate_drone_trajectory


def test_token_creation():
    t = SpatioTemporalToken(
        drone_id="DJI-TEST-001",
        protocol=ProtocolType.WIFI_BEACON,
        lat_deg=30.5,
        lon_deg=104.0,
        alt_m=100.0,
        rssi_dbm=-50,
        timestamp_utc=1.0,
    )
    assert t.drone_id == "DJI-TEST-001"
    assert t.protocol == ProtocolType.WIFI_BEACON
    assert t.token_id != ""
    print("  ✓ test_token_creation")


def test_tokenizer():
    tok = Tokenizer(hidden_dim=128)
    t = SpatioTemporalToken(
        drone_id="DJI-TEST-001",
        protocol=ProtocolType.WIFI_BEACON,
        lat_deg=30.5,
        lon_deg=104.0,
        alt_m=100.0,
        rssi_dbm=-50,
        snr_db=20,
        timestamp_utc=1.0,
    )
    emb = tok.encode(t, t0=0.0)
    assert emb.shape == (128,)
    assert not np.all(emb == 0)
    print("  ✓ test_tokenizer (shape=%s)" % str(emb.shape))


def test_tokenizer_batch():
    tok = Tokenizer(hidden_dim=128)
    tokens = [
        SpatioTemporalToken(
            drone_id="DJI-TEST-001",
            protocol=ProtocolType.WIFI_BEACON,
            lat_deg=30.5, lon_deg=104.0, alt_m=100.0,
            rssi_dbm=-50, snr_db=20, timestamp_utc=float(i),
        )
        for i in range(10)
    ]
    embs = tok.encode_batch(tokens)
    assert embs.shape == (10, 128)
    print("  ✓ test_tokenizer_batch (shape=%s)" % str(embs.shape))


def test_deduplication():
    t1 = SpatioTemporalToken(
        drone_id="DJI-A", protocol=ProtocolType.WIFI_BEACON,
        lat_deg=30.5, lon_deg=104.0, alt_m=100.0,
        rssi_dbm=-50, timestamp_utc=1.0,
    )
    t2 = SpatioTemporalToken(
        drone_id="DJI-A", protocol=ProtocolType.BLE_ADVB,
        lat_deg=30.5, lon_deg=104.0, alt_m=100.0,
        rssi_dbm=-60, timestamp_utc=1.0,  # same drone, same time, different protocol
    )
    t3 = SpatioTemporalToken(
        drone_id="DJI-B", protocol=ProtocolType.WIFI_BEACON,
        lat_deg=30.6, lon_deg=104.1, alt_m=80.0,
        rssi_dbm=-70, timestamp_utc=1.0,  # different drone
    )

    deduped = deduplicate_tokens([t1, t2, t3], time_window_s=0.5)
    # DJI-A has 2 tokens within the window → merged to 1
    # DJI-B has 1 token → kept
    assert len(deduped) == 2  # DJI-A merged + DJI-B
    print("  ✓ test_deduplication (%d tokens → %d)" % (3, len(deduped)))


def test_signal_simulator():
    sim = RIDSignalSimulator([ProtocolType.WIFI_BEACON, ProtocolType.BLE_ADVB], seed=42)
    tokens = sim.observe(
        "DJI-TEST-001", timestamp_utc=1.0,
        lat_deg=30.5, lon_deg=104.0, alt_m=100.0,
    )
    assert len(tokens) >= 0  # may miss detection
    if tokens:
        assert tokens[0].drone_id == "DJI-TEST-001"
    print("  ✓ test_signal_simulator (%d tokens generated)" % len(tokens))


def test_trajectory_generation():
    traj = generate_drone_trajectory(
        "DJI-TEST", 30.5, 104.0, 100.0, duration_s=10.0, dt_s=1.0,
    )
    assert len(traj) == 10
    assert traj[0]["drone_id"] == "DJI-TEST"
    print("  ✓ test_trajectory_generation (%d points)" % len(traj))


def test_fusion_engine():
    from rid_fusion.fusion import RIDFusionEngine
    engine = RIDFusionEngine(
        protocols=[ProtocolType.WIFI_BEACON, ProtocolType.BLE_ADVB],
        d_model=128, seed=42,
    )
    result = engine.run_on_trajectory(
        "DJI-TEST", 30.5, 104.0, 100.0, duration_s=10.0,
    )
    assert "raw_tokens" in result
    assert "fused_tokens" in result
    assert "stats" in result
    assert result["stats"]["total_raw_tokens"] >= 0
    print("  ✓ test_fusion_engine (raw=%d, fused=%d)" % (
        result["stats"]["total_raw_tokens"],
        result["stats"]["total_fused_tokens"],
    ))


def test_meta_learner():
    from rid_fusion.meta_learner import simulate_meta_learning_demo
    result = simulate_meta_learning_demo(seed=42)
    assert "meta_train_losses" in result
    assert "adapted_loss" in result
    assert "improvement_factor" in result
    assert result["improvement_factor"] > 0
    print("  ✓ test_meta_learner (improvement: %.1fx)" % result["improvement_factor"])


# ═══════════════════════════════════════════════════════════
# Boundary / edge-case tests
# ═══════════════════════════════════════════════════════════

def test_tokenizer_empty_batch():
    """Empty token list should return (0, hidden_dim) array."""
    tok = Tokenizer(hidden_dim=128)
    embs = tok.encode_batch([])
    assert embs.shape == (0, 128)
    print("  ✓ test_tokenizer_empty_batch")


def test_deduplication_empty():
    """Empty input should return empty list."""
    result = deduplicate_tokens([])
    assert result == []
    print("  ✓ test_deduplication_empty")


def test_deduplication_single():
    """Single token passes through unchanged."""
    t = SpatioTemporalToken(
        drone_id="DJI-A", protocol=ProtocolType.WIFI_BEACON,
        lat_deg=30.5, lon_deg=104.0, alt_m=100.0,
        rssi_dbm=-50, timestamp_utc=1.0,
    )
    result = deduplicate_tokens([t])
    assert len(result) == 1
    assert result[0].drone_id == "DJI-A"
    print("  ✓ test_deduplication_single")


def test_signal_simulator_single_protocol():
    """Single-protocol simulator should work correctly."""
    sim = RIDSignalSimulator([ProtocolType.NR_BROADCAST], seed=42)
    tokens = sim.observe("DJI-TEST", 1.0, 30.5, 104.0, 100.0, vx_ms=5.0, vy_ms=3.0)
    for t in tokens:
        assert t.protocol == ProtocolType.NR_BROADCAST
        assert t.vx_ms != 0  # NR has velocity
    print("  ✓ test_signal_simulator_single_protocol (%d tokens)" % len(tokens))


def test_signal_simulator_zero_velocity():
    """Zero velocity input should produce approximately zero-velocity tokens."""
    sim = RIDSignalSimulator([ProtocolType.WIFI_BEACON], seed=42)
    tokens = sim.observe("DJI-TEST", 1.0, 30.5, 104.0, 100.0,
                         vx_ms=0.0, vy_ms=0.0, vz_ms=0.0)
    if tokens:
        # Velocity should be within noise bounds (±3σ)
        assert abs(tokens[0].vx_ms) < 1.5
        assert abs(tokens[0].vy_ms) < 1.5
    print("  ✓ test_signal_simulator_zero_velocity")


def test_signal_simulator_extreme_noise():
    """LoRaWAN has very high position error std — should still produce valid tokens."""
    sim = RIDSignalSimulator([ProtocolType.LORAWAN], seed=42)
    tokens = sim.observe("DJI-TEST", 1.0, 30.5, 104.0, 100.0)
    if tokens:
        t = tokens[0]
        assert -90 <= t.lat_deg <= 90
        assert -180 <= t.lon_deg <= 180
        assert t.lat_error_m is not None
    print("  ✓ test_signal_simulator_extreme_noise (%d tokens)" % len(tokens))


def test_fusion_single_protocol():
    """Fusion should work with only one protocol (no cross-modal needed)."""
    from rid_fusion.fusion import RIDFusionEngine
    engine = RIDFusionEngine(
        protocols=[ProtocolType.WIFI_BEACON],
        d_model=128, seed=42,
    )
    result = engine.run_on_trajectory(
        "DJI-TEST", 30.5, 104.0, 100.0, duration_s=5.0,
    )
    assert result["stats"]["total_raw_tokens"] >= 0
    # Each FusedToken should have a unique timestamp
    if result["fused_tokens"]:
        timestamps = [ft.timestamp_utc for ft in result["fused_tokens"]]
        assert len(timestamps) == len(set(timestamps)), \
            "FusedTokens should have unique timestamps"
    print("  ✓ test_fusion_single_protocol (raw=%d, fused=%d)" % (
        result["stats"]["total_raw_tokens"],
        result["stats"]["total_fused_tokens"],
    ))


def test_complex_trajectory():
    """Multi-segment trajectory with hover."""
    from rid_fusion.signals import generate_complex_trajectory
    waypoints = [
        {"lat": 30.57, "lon": 104.07, "alt": 0.0, "speed_ms": 5.0},
        {"lat": 30.58, "lon": 104.08, "alt": 50.0, "hover_s": 3.0},
        {"lat": 30.59, "lon": 104.09, "alt": 100.0, "speed_ms": 8.0},
    ]
    traj = generate_complex_trajectory("DJI-TEST", waypoints, dt_s=1.0)
    assert len(traj) > 0
    # Verify hover points have near-zero velocity
    hover_pts = [p for p in traj if abs(p["vx_ms"]) < 0.3 and abs(p["vy_ms"]) < 0.3]
    assert len(hover_pts) >= 2, "Should have hover points with near-zero velocity"
    print("  ✓ test_complex_trajectory (%d points, %d hover)" % (len(traj), len(hover_pts)))


def test_deduplication_position_outlier():
    """Dedup should exclude a token whose position deviates >50m from the group."""
    t1 = SpatioTemporalToken(
        drone_id="DJI-A", protocol=ProtocolType.WIFI_BEACON,
        lat_deg=30.5000, lon_deg=104.0000, alt_m=100.0,
        rssi_dbm=-50, timestamp_utc=1.0,
    )
    t2 = SpatioTemporalToken(
        drone_id="DJI-A", protocol=ProtocolType.BLE_ADVB,
        lat_deg=30.5000, lon_deg=104.0000, alt_m=100.0,
        rssi_dbm=-60, timestamp_utc=1.0,
    )
    t3 = SpatioTemporalToken(
        drone_id="DJI-A", protocol=ProtocolType.NR_BROADCAST,
        # Offset by ~100m north — should be excluded
        lat_deg=30.5010, lon_deg=104.0000, alt_m=100.0,
        rssi_dbm=-40, timestamp_utc=1.0,
    )
    deduped = deduplicate_tokens([t1, t2, t3], max_position_diff_m=50.0)
    assert len(deduped) == 1
    best = deduped[0]
    # The outlier (t3, NR) should be excluded despite having best RSSI
    assert best.protocol != ProtocolType.NR_BROADCAST, \
        "Position outlier should not be selected even with best RSSI"
    assert "NR_BROADCAST" in best.protocol_payload.get("dedup_outliers", [])
    print("  ✓ test_deduplication_position_outlier")


if __name__ == "__main__":
    print("Running rid-fusion tests...\n")
    tests = [
        test_token_creation,
        test_tokenizer,
        test_tokenizer_batch,
        test_deduplication,
        test_signal_simulator,
        test_trajectory_generation,
        test_fusion_engine,
        test_meta_learner,
        # Boundary / edge-case tests
        test_tokenizer_empty_batch,
        test_deduplication_empty,
        test_deduplication_single,
        test_signal_simulator_single_protocol,
        test_signal_simulator_zero_velocity,
        test_signal_simulator_extreme_noise,
        test_fusion_single_protocol,
        test_complex_trajectory,
        test_deduplication_position_outlier,
    ]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as e:
            print("  ✗ %s FAILED: %s" % (test.__name__, e))
            failures += 1
    print("\n%d/%d tests passed" % (len(tests) - failures, len(tests)))
    sys.exit(failures)
