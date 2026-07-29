"""
Embodiment 3: Token Deduplication for Cost Savings
===================================================

Demonstrates cross-protocol token deduplication:
  - Same drone broadcasting via 3 protocols simultaneously
    generates redundant tokens (same drone_id, same timestamp)
  - Deduplication merges them into a single token per time_window,
    keeping the highest-RSSI reading as the representative
  - This reduces downstream computation cost without losing
    information (all source protocols are recorded)

Usage:
    python examples/embodiment3_dedup.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rid_fusion.models import ProtocolType
from rid_fusion.signals import RIDSignalSimulator, generate_drone_trajectory
from rid_fusion.tokenizer import deduplicate_tokens, Tokenizer


def main():
    print("=" * 60)
    print("EMBODIMENT 3: Cross-Protocol Token Deduplication")
    print("3 Protocols → 1 Fused Token per Time Window")
    print("=" * 60)

    # ── Setup: 3-protocol deployment (Shenzhen) ──
    sim = RIDSignalSimulator(
        protocols=[
            ProtocolType.WIFI_BEACON,
            ProtocolType.BLE_ADVB,
            ProtocolType.NR_BROADCAST,
        ],
        seed=42,
    )

    traj = generate_drone_trajectory(
        "DJI-Shenzhen-B7F2",
        start_lat=22.5431,   # Shenzhen Nanshan
        start_lon=113.9544,
        start_alt=80.0,
        duration_s=20.0,
        dt_s=1.0,
        speed_ms=6.0,
    )

    # Generate tokens
    all_tokens = []
    for pt in traj:
        tokens = sim.observe(
            drone_id=pt["drone_id"],
            timestamp_utc=pt["timestamp_utc"],
            lat_deg=pt["lat_deg"],
            lon_deg=pt["lon_deg"],
            alt_m=pt["alt_m"],
            vx_ms=pt["vx_ms"],
            vy_ms=pt["vy_ms"],
            vz_ms=pt["vz_ms"],
        )
        all_tokens.extend(tokens)

    print("\n--- Before Deduplication ---")
    print("  Total tokens: %d" % len(all_tokens))
    proto_counts = {}
    for t in all_tokens:
        proto_counts[t.protocol.value] = proto_counts.get(t.protocol.value, 0) + 1
    for proto, count in sorted(proto_counts.items()):
        print("    %-16s: %3d" % (proto, count))

    # ── Deduplication ──
    deduped = deduplicate_tokens(all_tokens, time_window_s=0.5)

    print("\n--- After Deduplication (time_window=0.5s) ---")
    print("  Deduped tokens: %d" % len(deduped))
    dedup_proto_counts = {}
    for t in deduped:
        dedup_proto_counts[t.protocol.value] = dedup_proto_counts.get(t.protocol.value, 0) + 1
    for proto, count in sorted(dedup_proto_counts.items()):
        print("    %-16s: %3d" % (proto, count))

    reduction = (1 - len(deduped) / max(len(all_tokens), 1)) * 100
    print("\n  Token reduction: %.1f%%" % reduction)

    # ── Verify information preservation ──
    print("\n--- Information Preservation Check ---")
    with_sources = sum(1 for t in deduped if len(t.protocol_payload.get("dedup_sources", [])) >= 2)
    print("  Tokens with 2+ source protocols: %d / %d (%.1f%%)" % (
        with_sources, len(deduped),
        with_sources / max(len(deduped), 1) * 100,
    ))

    if reduction > 40:
        print("  ✓ Significant cost reduction achieved")
    else:
        print("  △ Moderate reduction (consider tuning time_window)")

    # Show example merged token
    if deduped:
        multi = [t for t in deduped if len(t.protocol_payload.get("dedup_sources", [])) >= 2]
        if multi:
            example = multi[0]
            print("\n  Example merged token:")
            print("    drone_id: %s" % example.drone_id)
            print("    ts: %.1fs" % example.timestamp_utc)
            print("    position: (%.4f, %.4f)" % (example.lat_deg, example.lon_deg))
            print("    sources:  %s" % example.protocol_payload.get("dedup_sources", []))

    print("\n--- Patent Mapping ---")
    print("  This embodiment demonstrates:")
    print("  1. Multi-protocol redundancy (3 protocols → 1 drone)")
    print("  2. Cross-protocol deduplication (time_window + drone_id matching)")
    print("  3. Token cost reduction without information loss")
    print("  4. Cost savings → lower LLM inference token cost")
    print("  5. Applicable to any number of protocols")


if __name__ == "__main__":
    main()
