# rid-fusion

**Multi-Protocol Remote ID Signal Fusion Engine**

Protocol-agnostic tokenization, cross-modal attention fusion, and
meta-learning adaptation for heterogeneous drone Remote ID signals.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)]
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)]
[![Tests](https://img.shields.io/badge/tests-17%2F17-brightgreen.svg)]

---

## Overview

Modern counter-drone systems must process Remote ID signals from multiple
radio protocols simultaneously — Wi-Fi Beacon (ASTM F3411), Bluetooth 5.0
BLE Advertising, 4G/5G NR Sidelink Broadcast, LoRaWAN, and ADS-B.

Each protocol differs in:
- **Field availability** (position, velocity, altitude may or may not be present)
- **Noise characteristics** (BLE GPS is coarser than 5G NR positioning)
- **Detection probability** (LoRaWAN has lower P(d) than Wi-Fi)

**rid-fusion** solves this by:
1. **Tokenizing** every RID observation into a unified spatio-temporal token
2. **Encoding** each protocol into its own embedding space
3. **Fusing** cross-protocol tokens via multi-head cross-modal attention
4. **Adapting** to new protocols with 10-shot meta-learning (MAML)

The output is a **protocol-agnostic semantic vector** that can be
consumed directly by any downstream LLM or reasoning engine.

---

## Quick Start

```bash
# 1. Install dependencies (numpy + scipy only)
pip install -r requirements.txt

# 2. Launch the desktop GUI (recommended for demos)
python rid_fusion_gui.py

# 3. Or run via command line:
python tests/test_core.py            # 17 unit tests
python examples/embodiment1_dual_protocol.py
python examples/embodiment2_cross_city.py
python examples/embodiment3_dedup.py
```

---

## Architecture

```
┌──────────────────────────────────────────────────────┐
│  PROTOCOL-SPECIFIC SIGNALS                            │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐               │
│  │Wi-Fi    │  │BLE      │  │4G/5G NR │               │
│  │Beacon   │  │ADVB     │  │Sidelink │               │
│  └────┬────┘  └────┬────┘  └────┬────┘               │
│       │          │          │                         │
│       ▼          ▼          ▼                         │
│  ┌─────────────────────────────────────┐             │
│  │       SPATIO-TEMPORAL TOKENIZER     │             │
│  │  [τ, Geo, sig, ID, pos₃D, vel, RSSI]│            │
│  │  * Xavier-init, deterministic seed  │             │
│  └─────────────────┬───────────────────┘             │
│                    ▼                                   │
│  ┌─────────────────────────────────────┐             │
│  │   PROTOCOL-SPECIFIC ENCODERS        │             │
│  │  ┌────────┐ ┌────────┐ ┌────────┐   │             │
│  │  │WiFiEnc │ │BLEEnc  │ │NR-Enc  │   │             │
│  │  └───┬────┘ └───┬────┘ └───┬────┘   │             │
│  └──────┼───────────┼───────────┼───────┘             │
│         ▼           ▼           ▼                      │
│  ┌─────────────────────────────────────┐             │
│  │   CROSS-MODAL ATTENTION FUSION      │             │
│  │   Q(spatial) × K(temporal) × V(signal)│           │
│  │   * Per-modality learnable proj.    │             │
│  │   * Per-timestamp unique embeddings │             │
│  └─────────────────┬───────────────────┘             │
│                    ▼                                   │
│  ┌─────────────────────────────────────┐             │
│  │  PROTOCOL-AGNOSTIC FUSED TOKEN (v)  │             │
│  │  → Downstream LLM / Reasoning       │             │
│  └─────────────────────────────────────┘             │
│                                                       │
│  ┌─────────────────────────────────────┐             │
│  │  META-LEARNING ADAPTER (MAML)       │             │
│  │  New protocol · 10 shots → adapted  │             │
│  └─────────────────────────────────────┘             │
│                                                       │
│  ┌─────────────────────────────────────┐             │
│  │  FUSION CONFIG (centralised params)  │             │
│  │  d_model, n_heads, lr, seed, etc.   │             │
│  └─────────────────────────────────────┘             │
└──────────────────────────────────────────────────────┘
```

---

## Project Structure

```
rid-fusion/
├── rid_fusion/
│   ├── __init__.py          # Package metadata, public exports
│   ├── models.py            # SpatioTemporalToken, FusedToken, FusionConfig
│   ├── signals.py           # Multi-protocol RID signal simulator
│   │                          + straight-line & complex (multi-waypoint) trajectories
│   ├── tokenizer.py         # Space/time/signal/protocol tokenization
│   │                          + cross-protocol dedup w/ position sanity check
│   ├── encoders.py          # Protocol-specific encoders + cross-modal attention
│   │                          + per-modality learnable Q/K/V projections
│   ├── fusion.py            # Top-level fusion engine (RIDFusionEngine)
│   └── meta_learner.py      # MAML-based few-shot protocol adapter
├── examples/
│   ├── embodiment1_dual_protocol.py   # Wi-Fi + BLE dual fusion
│   ├── embodiment2_cross_city.py      # 10-shot adaptation to new protocol
│   └── embodiment3_dedup.py           # Token deduplication for cost saving
├── tests/
│   └── test_core.py         # 17 unit tests (8 core + 9 edge-case)
├── rid_fusion_gui.py        # Desktop GUI (tkinter, no extra deps)
├── README.md
├── CODE_REVIEW.md
├── setup.py
├── requirements.txt
└── LICENSE
```

---

## Three Embodiments (Patent-Aligned)

### Embodiment 1: Dual-Protocol Fusion
A DJI drone over Chengdu Hi-Tech Zone broadcasts RID via both
Wi-Fi Beacon and BLE ADVB. The fusion engine produces a unified token
that is more positionally accurate than either single-protocol token.

```bash
python examples/embodiment1_dual_protocol.py
```

### Embodiment 2: Cross-City Meta-Learning
A model trained on Wi-Fi + BLE (Chengdu) adapts to 4G/5G NR
(Shenzhen) with only 10 labeled samples — without retraining
the fusion engine or downstream LLM.

```bash
python examples/embodiment2_cross_city.py
```

### Embodiment 3: Token Deduplication
Three protocols generate redundant tokens for the same drone.
Cross-protocol deduplication reduces token count by ~50%
without information loss. Includes position outlier detection.

```bash
python examples/embodiment3_dedup.py
```

---

## Supported Protocols

| Protocol | Standard | Typical Range | Position Accuracy |
|:---|:---|:---|:---|
| Wi-Fi Beacon | ASTM F3411-22a | ~1 km | ±3 m |
| BLE ADVB | Bluetooth 5.0 | ~300 m | ±5 m |
| 4G/5G NR Sidelink | 3GPP Rel-17 | ~3 km | ±2 m |
| LoRaWAN | LoRa Alliance | ~5 km | ±20 m |
| ADS-B | ICAO | ~100 km | ±10 m |

*Note: Longitude error conversion now includes cos(latitude) correction
(was previously a flat 111,000 m/deg for both axes).*

---

## Key Features (v0.2.0)

| Feature | Module |
|:---|:---|
| Multi-protocol RID → unified spatio-temporal token | `tokenizer.py` |
| Protocol-specific encoding before cross-modal fusion | `encoders.py` |
| Multi-head cross-modal attention (Q[spatial] × K[temporal] × V[signal]) | `encoders.py` |
| Per-modality learnable Q/K/V projections (replaces fixed slicing) | `encoders.py` |
| Per-timestamp unique fused embeddings (was shared global vector) | `encoders.py` |
| Token deduplication with position outlier detection | `tokenizer.py` |
| 10-shot MAML adaptation (first-order, corrected LR bug) | `meta_learner.py` |
| Deterministic Xavier-initialised tokenizer (seed-reproducible) | `tokenizer.py` |
| Longitude degree cos(lat) correction for geographic accuracy | `signals.py` |
| Complex multi-waypoint trajectories with hover segments | `signals.py` |
| Centralised `FusionConfig` for all hyperparameters | `models.py` |
| Structured logging across all 5 core modules | `*` |
| 17 tests: 8 core + 9 edge-case (empty/single/extreme noise/outlier) | `tests/` |

---

## Patent Coverage

This codebase supports the following patent claims:

| Claim | Module |
|:---|:---|
| Multi-protocol RID signal → unified spatio-temporal token | `tokenizer.py` |
| Protocol-specific encoding before cross-modal fusion | `encoders.py` |
| Multi-head cross-modal attention (Q[space] × K[time] × V[signal]) | `encoders.py` |
| Per-modality learnable attention projections | `encoders.py` |
| Token deduplication with position consistency guard | `tokenizer.py` |
| 10-shot meta-learning adaptation to unknown protocols | `meta_learner.py` |
| Protocol-agnostic output (FusedToken) for LLM consumption | `fusion.py` |
| Centralised reproducible configuration system | `models.py` |

---

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Citation

If you use this code in research, please cite:

```bibtex
@software{rid_fusion_2026,
  author = {Kongyu Technologies},
  title = {rid-fusion: Multi-Protocol Remote ID Signal Fusion Engine},
  year = {2026},
  version = {0.2.0},
  url = {https://github.com/kongyu/rid-fusion},
}
```
