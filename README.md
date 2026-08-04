# RID Fusion Studio v0.4.0

RID Fusion Studio 是一个面向无人机 Remote ID（RID）研究的可解释多来源观测关联与状态融合原型。仓库包含完整 Python 源码、Windows WPF 桌面界面、统一数据格式、场景模拟、异常检测、算法对比、报告导出和测试。

> 当前项目是研究与专利实施演示原型，不是实际无线电监听设备，也不是执法、适航或空域审批系统。

## 它解决什么问题

同一架无人机的位置、速度和高度可能由不同接收方式产生，数据会出现噪声、漏报、字段缺失和时间差。项目将这些观测转换为统一对象，然后依次执行：

1. 字段规范化与缺失值显式记录；
2. 按目标身份和时间窗进行观测关联；
3. 使用协方差加权生成可解释融合状态；
4. 保留每个结果使用的观测ID、协议贡献和不确定度；
5. 检查身份冲突、重复播放、异常速度、时间倒退和位置跳变；
6. 支持多目标场景、回放、算法对比和报告导出。

注意：Wi-Fi 和 BLE 在这里是 RID 消息承载/观测来源；NR 与 LoRa 是实验扩展来源，不应表述为通用法定 RID 标准。

## 功能

- 单目标与最多 20 目标的可复现实验；
- 5 个后端固定场景，以及桌面端扩展场景模板；
- CSV、JSON、JSONL 观测文件导入与逐帧回放；
- 协方差加权、简单平均、最佳单来源三种方法对比；
- 结构化 JSON、CSV、Markdown 报告；
- 协议特定编码、跨模态注意力语义表征；
- 少样本物理状态适配实验；
- Windows 城市搜索、当前位置与天气参数回填；
- 白色主题 WPF 原生桌面界面。

## 快速开始

需要 Python 3.9+，推荐 Python 3.12。

```powershell
git clone https://github.com/RychHu/rid-fusion.git
cd rid-fusion
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

启动桌面端：

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\desktop\Run-RIDFusion.ps1
```

也可以双击根目录的 `启动_RID_Fusion.vbs`。

第一次使用建议打开“融合实验”，保留默认参数并点击“运行融合”。

## 命令行

```powershell
# 单目标融合
python -m rid_fusion.desktop_api fusion --duration 10 --seed 42 --protocols wifi_ble_nr

# 三目标场景
python -m rid_fusion.desktop_api multi --duration 10 --count 3 --spacing-m 80

# 导入示例数据
python -m rid_fusion.desktop_api import --path sample_data/observations_example.csv --bucket 1

# 算法对比
python -m rid_fusion.desktop_api compare --duration 10

# 内置自检
python -m rid_fusion.desktop_api selftest
```

命令输出统一采用：

```json
{"ok": true, "data": {}}
```

## 输入数据

最小字段：

| 字段 | 含义 |
|---|---|
| `uas_id` | 无人机/目标ID |
| `protocol` | `WIFI`、`BLE`、`NR`、`LORA` 或规范化名称 |
| `timestamp_utc` | 观测时间，UTC 秒数 |
| `lat_deg` | 纬度，-90 到 90 |
| `lon_deg` | 经度，-180 到 180 |

高度、三轴速度、RSSI、SNR、测量方差和接收机ID均为可选字段。未知值应留空或使用 `null`，不要用 0 代替未知值。完整说明见 [`sample_data/数据格式说明.txt`](sample_data/数据格式说明.txt) 和 [`sample_data/rid_observation_schema.json`](sample_data/rid_observation_schema.json)。

## 输出解释

- `received_observations`：接收到的有效观测数；
- `associated_groups`：目标ID与时间窗形成的关联组数；
- `fused_states`：生成的融合状态数；
- `uncertainty_m`：依据输入方差计算的水平分散程度，不是真实误差保证；
- `protocol_weights`：不同来源对当前位置结果的计算贡献；
- `evidence_count`：该状态使用的原始观测数量；
- `anomalies`：触发确定性检查规则的事件，不等同于违法或攻击结论。

桌面轨迹图中，灰色虚线是模拟参考轨迹；彩色点是各目标的融合位置。导入数据没有参考真值，因此灰线只表示导入状态连线。

## 项目结构

```text
rid-fusion/
├─ rid_fusion/             Python 核心源码
│  ├─ models.py            统一观测、关联组与融合状态
│  ├─ association.py       时空关联与身份冲突标记
│  ├─ classical_fusion.py  协方差加权融合
│  ├─ anomaly.py           可解释异常检查
│  ├─ importers/           CSV/JSON/JSONL 导入
│  ├─ reporting.py         JSON/CSV/Markdown 报告
│  └─ desktop_api.py       桌面端 JSON 命令接口
├─ desktop/                WPF XAML 与 PowerShell 控制层
├─ sample_data/            示例数据与 JSON Schema
├─ examples/               Python 使用示例
├─ tests/                  核心与 v0.4 回归测试
├─ scripts/                验证和可选 EXE 构建脚本
├─ packaging/              PyInstaller 入口
└─ 启动_RID_Fusion.vbs      Windows 快捷启动入口
```

## 验证与构建

```powershell
# 完整验证
.\scripts\validate.ps1 -Python .\.venv\Scripts\python.exe

# 可选：生成便携版后端 EXE
.\scripts\build_backend.ps1 -Python .\.venv\Scripts\python.exe
```

仓库默认只提交源码，不提交 Python 运行时、DLL、`.pyd` 和编译后的 EXE。便携安装包应通过 GitHub Release 单独发布。

## 与低空监管垂类模型的关系

该项目当前主要支撑“RID 多来源数据规范化、关联、融合和统一监管对象生成”的数据底座。它还没有实现真实 ADS-B/视觉/飞行计划接入、空域规则库、向量知识库、RAG 或垂类大模型训练，因此不能单独表述为已经完成低空监管垂类大模型。

## 许可证

[MIT License](LICENSE)
