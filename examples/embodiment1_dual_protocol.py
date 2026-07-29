"""
Embodiment 1: Dual-Protocol Fusion (Wi-Fi Beacon + BLE ADVB)
=============================================================

Demonstrates the core claim of the patent:
  - A single drone broadcasts RID via both Wi-Fi and Bluetooth simultaneously
  - Each protocol has different field availability and noise characteristics
  - Cross-modal attention fusion produces a unified token that is
    more accurate than either single-protocol token alone

Usage:
    python examples/embodiment1_dual_protocol.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rid_fusion.models import ProtocolType, SpatioTemporalToken
from rid_fusion.fusion import RIDFusionEngine, compare_single_vs_fused
from rid_fusion.tokenizer import deduplicate_tokens


def main():
    print("=" * 60)
    print("EMBODIMENT 1: Dual-Protocol RID Fusion")
    print("Chengdu Hi-Tech Zone — Wi-Fi Beacon + BLE ADVB")
    print("=" * 60)

    # ── Setup: Chengdu Hi-Tech Zone deployment ──
    engine = RIDFusionEngine(
        protocols=[ProtocolType.WIFI_BEACON, ProtocolType.BLE_ADVB],
        d_model=128,
        seed=42,
        enable_dedup=True,
    )

    # ── Scenario: DJI drone flying over Chengdu Hi-Tech Zone ──
    result = compare_single_vs_fused(
        engine,
        drone_id="DJI-2024-A17F",
        start_lat=30.5728,   # Chengdu Hi-Tech Zone
        start_lon=104.0668,
        start_alt=120.0,     # 120m altitude (typical RID reporting height)
        duration_s=30.0,
    )

    print("\n--- Single-Protocol Tokens ---")
    for proto, count in result["single_protocol"].items():
        print("  %-12s: %3d tokens" % (proto, count))

    print("\n--- Cross-Protocol Deduplication ---")
    # Re-run without dedup to compare
    engine_no_dedup = RIDFusionEngine(
        protocols=[ProtocolType.WIFI_BEACON, ProtocolType.BLE_ADVB],
        d_model=128, seed=42, enable_dedup=False,
    )
    result_no_dedup = engine_no_dedup.run_on_trajectory(
        "DJI-2024-A17F", 30.5728, 104.0668, 120.0, duration_s=30.0,
    )
    print("  Without dedup: %d raw tokens" % len(result_no_dedup["raw_tokens"]))
    print("  With dedup:    %d fused tokens" % result["fused_total"])
    reduction = (1 - result["fused_total"] / max(len(result_no_dedup["raw_tokens"]), 1)) * 100
    print("  Token reduction: %.1f%%" % reduction)

    print("\n--- Fusion Results ---")
    print("  Fused tokens:        %d" % result["fused_total"])
    print("  Multi-protocol enriched: %d (%.1f%%)" % (
        result["fused_enriched"],
        result["enrichment_ratio"] * 100,
    ))

    # Show sample fused tokens
    if result["fused_tokens"]:
        print("\n  Sample fused tokens:")
        for ft in result["fused_tokens"][:3]:
            print("    ts=%.1fs  drone=%s  sources=%s  emb_dim=%d" % (
                ft.timestamp_utc,
                ft.drone_id,
                [p.value for p in ft.source_protocols],
                len(ft.embedding),
            ))

    print("\n--- Patent Mapping ---")
    print("  This embodiment demonstrates:")
    print("  1. Multi-protocol RID signal simulation (ASTM F3411 Wi-Fi + BLE 5.0)")
    print("  2. Protocol-specific tokenization (different noise profiles)")
    print("  3. Cross-protocol deduplication (same drone_id + timestamp → merge)")
    print("  4. Cross-modal attention fusion (spatial ↔ temporal ↔ signal)")
    print("  5. Output: protocol-agnostic FusedToken (ready for LLM consumption)")
    print()

    # ── Trajectory sample ──
    print("--- Ground-Truth Trajectory Sample ---")
    for pt in result["trajectory"][:5]:
        print("  t=%.1fs  pos=(%.4f, %.4f)  alt=%.1fm  vel=(%.1f, %.1f) m/s" % (
            pt["timestamp_utc"],
            pt["lat_deg"], pt["lon_deg"],
            pt["alt_m"],
            pt["vx_ms"], pt["vy_ms"],
        ))
    print("  ... (%d total points)" % len(result["trajectory"]))


if __name__ == "__main__":
    main()
