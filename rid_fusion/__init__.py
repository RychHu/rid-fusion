"""
rid-fusion: Multi-Protocol Remote ID Signal Fusion Engine
==========================================================
Protocol-agnostic tokenization, cross-modal attention fusion,
and meta-learning adaptation for heterogeneous drone Remote ID signals.

Target: ASTM F3411 (Wi-Fi Beacon), Bluetooth 5.0 BLE ADVB, 4G/5G NR Broadcast
"""

__version__ = "0.2.0"

from rid_fusion.models import (
    ProtocolType,
    ObjectClass,
    SpatioTemporalToken,
    TokenSequence,
    FusedToken,
    Detection,
    FusionConfig,
)
