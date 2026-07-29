# rid-fusion

**多协议 Remote ID 信号融合引擎**

面向异构无人机 Remote ID 信号的协议无关令牌化、跨模态注意力融合与元学习自适应。

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)]
[![Version](https://img.shields.io/badge/version-0.2.0-orange.svg)]
[![Tests](https://img.shields.io/badge/tests-17%2F17-brightgreen.svg)]

---

## 概述

现代反无人机系统需要同时处理来自多种无线电协议的 Remote ID 信号——Wi-Fi Beacon（ASTM F3411）、蓝牙 5.0 BLE Advertising、4G/5G NR Sidelink Broadcast、LoRaWAN 以及 ADS-B。

不同协议的差异体现在：
- **字段可用性**（位置、速度、高度可能存在或缺失）
- **噪声特性**（BLE GPS 精度低于 5G NR 定位）
- **检测概率**（LoRaWAN 的检测概率低于 Wi-Fi）

**rid-fusion** 的解决方案：
1. **令牌化**：将每条 RID 观测数据转化为统一时空令牌
2. **编码**：将每种协议映射到各自的嵌入空间
3. **融合**：通过多头跨模态注意力机制融合跨协议令牌
4. **自适应**：利用 10-shot 元学习（MAML）适配新协议

最终输出是一个**与协议无关的语义向量**，可直接被任意下游大模型或推理引擎消费。

---

## 快速开始

```bash
# 1. 安装依赖（仅需 numpy + scipy）
pip install -r requirements.txt

# 2. 运行全部 17 项测试
python tests/test_core.py

# 3. 运行交互式演示
python examples/embodiment1_dual_protocol.py
python examples/embodiment2_cross_city.py
python examples/embodiment3_dedup.py
```

---

## 架构

```
┌──────────────────────────────────────────────────────┐
│  协议特定信号                                           │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐               │
│  │Wi-Fi    │  │BLE      │  │4G/5G NR │               │
│  │Beacon   │  │ADVB     │  │Sidelink │               │
│  └────┬────┘  └────┬────┘  └────┬────┘               │
│       │          │          │                         │
│       ▼          ▼          ▼                         │
│  ┌─────────────────────────────────────┐             │
│  │       时空令牌化器（Tokenizer）        │             │
│  │  [τ, Geo, sig, ID, pos₃D, vel, RSSI]│            │
│  │  * Xavier 初始化，确定性种子          │             │
│  └─────────────────┬───────────────────┘             │
│                    ▼                                   │
│  ┌─────────────────────────────────────┐             │
│  │   协议特定编码器（Encoders）          │             │
│  │  ┌────────┐ ┌────────┐ ┌────────┐   │             │
│  │  │WiFiEnc │ │BLEEnc  │ │NR-Enc  │   │             │
│  │  └───┬────┘ └───┬────┘ └───┬────┘   │             │
│  └──────┼───────────┼───────────┼───────┘             │
│         ▼           ▼           ▼                      │
│  ┌─────────────────────────────────────┐             │
│  │   跨模态注意力融合                    │             │
│  │   Q(空间) × K(时间) × V(信号)        │             │
│  │   * 按模态可学习投影                  │             │
│  │   * 按时间戳独立生成嵌入              │             │
│  └─────────────────┬───────────────────┘             │
│                    ▼                                   │
│  ┌─────────────────────────────────────┐             │
│  │  协议无关融合令牌（FusedToken）       │             │
│  │  → 下游大模型 / 推理引擎             │             │
│  └─────────────────────────────────────┘             │
│                                                       │
│  ┌─────────────────────────────────────┐             │
│  │  元学习适配器（MAML）                 │             │
│  │  新协议 · 10 样本 → 已适配           │             │
│  └─────────────────────────────────────┘             │
│                                                       │
│  ┌─────────────────────────────────────┐             │
│  │  融合配置（FusionConfig）             │             │
│  │  集中管理 d_model、n_heads、lr 等     │             │
│  └─────────────────────────────────────┘             │
└──────────────────────────────────────────────────────┘
```

---

## 项目结构

```
rid-fusion/
├── rid_fusion/
│   ├── __init__.py          # 包元数据，公开导出接口
│   ├── models.py            # SpatioTemporalToken、FusedToken、FusionConfig
│   ├── signals.py           # 多协议 RID 信号模拟器
│   │                          + 直线与复杂（多航点）轨迹生成
│   ├── tokenizer.py         # 空间/时间/信号/协议令牌化
│   │                          + 跨协议去重与位置一致性校验
│   ├── encoders.py          # 协议特定编码器 + 跨模态注意力
│   │                          + 按模态可学习 Q/K/V 投影
│   ├── fusion.py            # 顶层融合引擎（RIDFusionEngine）
│   └── meta_learner.py      # 基于 MAML 的小样本协议适配器
├── examples/
│   ├── embodiment1_dual_protocol.py   # Wi-Fi + BLE 双协议融合
│   ├── embodiment2_cross_city.py      # 10-shot 跨城市新协议适配
│   └── embodiment3_dedup.py           # 令牌去重降低成本
├── tests/
│   └── test_core.py         # 17 项单元测试（8 核心 + 9 边界）
├── README.md
├── README_CN.md
├── CODE_REVIEW.md
├── setup.py
├── requirements.txt
└── LICENSE
```

---

## 三个实施例（与专利对应）

### 实施例 1：双协议融合
一架大疆无人机在成都高新区上空飞行，同时通过 Wi-Fi Beacon 和 BLE ADVB 广播 RID 信号。融合引擎生成的统一令牌在定位精度上优于任一单协议令牌。

```bash
python examples/embodiment1_dual_protocol.py
```

### 实施例 2：跨城市元学习
在成都使用 Wi-Fi + BLE 训练的模型，仅需 10 个标注样本即可适配深圳的 4G/5G NR 新协议，无需重新训练融合引擎或下游大模型。

```bash
python examples/embodiment2_cross_city.py
```

### 实施例 3：令牌去重降本
三种协议为同一架无人机生成冗余令牌。跨协议去重在不丢失信息的前提下将令牌数量减少约 50%，并具备位置异常检测能力。

```bash
python examples/embodiment3_dedup.py
```

---

## 支持的协议

| 协议 | 标准 | 典型覆盖范围 | 定位精度 |
|:---|:---|:---|:---|
| Wi-Fi Beacon | ASTM F3411-22a | ~1 km | ±3 m |
| BLE ADVB | 蓝牙 5.0 | ~300 m | ±5 m |
| 4G/5G NR Sidelink | 3GPP Rel-17 | ~3 km | ±2 m |
| LoRaWAN | LoRa 联盟 | ~5 km | ±20 m |
| ADS-B | ICAO | ~100 km | ±10 m |

*注：经度误差转换已加入 cos(纬度) 修正（此前两个轴向均使用固定 111,000 m/deg）。*

---

## 核心特性（v0.2.0）

| 特性 | 所在模块 |
|:---|:---|
| 多协议 RID → 统一时空令牌 | `tokenizer.py` |
| 融合前进行协议特定编码 | `encoders.py` |
| 多头跨模态注意力（Q[空间] × K[时间] × V[信号]） | `encoders.py` |
| 按模态可学习 Q/K/V 投影（替代固定切片） | `encoders.py` |
| 按时间戳独立生成融合嵌入（原为共享全局向量） | `encoders.py` |
| 令牌去重与位置异常检测 | `tokenizer.py` |
| 10-shot MAML 自适应（一阶近似，已修复学习率 Bug） | `meta_learner.py` |
| 确定性 Xavier 初始化令牌化器（seed 可复现） | `tokenizer.py` |
| 经度 cos(纬度) 修正，确保地理精度 | `signals.py` |
| 复杂多航点轨迹与悬停段生成 | `signals.py` |
| 集中式 `FusionConfig` 管理全部超参数 | `models.py` |
| 全部 5 个核心模块接入结构化日志 | `*` |
| 17 项测试：8 核心 + 9 边界（空输入/单协议/极端噪声/异常值） | `tests/` |

---

## 专利覆盖范围

本代码库支撑以下专利权利要求：

| 权利要求 | 所在模块 |
|:---|:---|
| 多协议 RID 信号 → 统一时空令牌 | `tokenizer.py` |
| 融合前进行协议特定编码 | `encoders.py` |
| 多头跨模态注意力（Q[空间] × K[时间] × V[信号]） | `encoders.py` |
| 按模态可学习注意力投影 | `encoders.py` |
| 令牌去重与位置一致性保护 | `tokenizer.py` |
| 10-shot 元学习适配未知协议 | `meta_learner.py` |
| 协议无关输出（FusedToken）供大模型消费 | `fusion.py` |
| 集中式可复现配置系统 | `models.py` |

---

## 许可证

MIT 许可证。详见 [LICENSE](LICENSE)。

---

## 引用

如在研究中使用本代码，请引用：

```bibtex
@software{rid_fusion_2026,
  author = {空御科技},
  title = {rid-fusion: 多协议 Remote ID 信号融合引擎},
  year = {2026},
  version = {0.2.0},
  url = {https://github.com/RychHu/rid-fusion},
}
```
