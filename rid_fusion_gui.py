#!/usr/bin/env python
"""
rid_fusion_gui.py — Desktop GUI for the RID Fusion Engine v0.2.0
=================================================================
Single-window tkinter application wrapping all rid-fusion functionality.
No extra dependencies beyond Python stdlib + numpy/scipy.

Launch:  python rid_fusion_gui.py
"""
from __future__ import annotations

import io
import sys
import os
import threading
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Dark theme colours ──
BG      = "#1a1a2e"
BG2     = "#16213e"
BG3     = "#0f3460"
FG      = "#e0e0e0"
ACCENT  = "#e94560"
ACCENT2 = "#00b4d8"
GREEN   = "#2ecc71"
YELLOW  = "#f1c40f"
RED     = "#e74c3c"
BORDER  = "#2a2a4a"
ENTRY_BG = "#0d0d1a"


class RedirectText(io.StringIO):
    """Redirects print() output to a tkinter ScrolledText widget."""

    def __init__(self, widget: scrolledtext.ScrolledText):
        super().__init__()
        self.widget = widget

    def write(self, s: str) -> int:
        self.widget.insert(tk.END, s)
        self.widget.see(tk.END)
        self.widget.update_idletasks()
        return len(s)

    def flush(self) -> None:
        pass


def _run_in_thread(target, *args, **kwargs):
    """Run target in a daemon thread so the UI stays responsive."""
    t = threading.Thread(target=target, args=args, kwargs=kwargs, daemon=True)
    t.start()


class RIDFusionGUI:
    """Main application window."""

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("RID Fusion Engine  v0.2.0  —  空御科技总体部")
        self.root.geometry("1000x720")
        self.root.minsize(800, 600)
        self.root.configure(bg=BG)

        self._setup_style()
        self._build_notebook()
        self._build_fusion_tab()
        self._build_metalearn_tab()
        self._build_dedup_tab()
        self._build_tests_tab()
        self._build_config_tab()

        # Status bar
        self.status_var = tk.StringVar(value="Ready — 选择标签页开始操作")
        status_bar = tk.Label(
            self.root, textvariable=self.status_var, anchor=tk.W,
            bg=BG3, fg=FG, font=("Consolas", 9), padx=10, pady=3,
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    # ═══════════════════════════════════════════════════════════
    # Styling
    # ═══════════════════════════════════════════════════════════

    def _setup_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")

        # Notebook
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab",
                        background=BG2, foreground=FG, padding=(16, 6),
                        font=("Microsoft YaHei UI", 10, "bold"),
                        borderwidth=0)
        style.map("TNotebook.Tab",
                  background=[("selected", BG3), ("active", BG3)],
                  foreground=[("selected", ACCENT)])

        # Frames
        style.configure("TFrame", background=BG)
        style.configure("Dark.TFrame", background=BG2)
        style.configure("Dark2.TFrame", background=BG3)

        # Labels
        style.configure("TLabel", background=BG, foreground=FG,
                        font=("Microsoft YaHei UI", 9))
        style.configure("Header.TLabel", background=BG, foreground=ACCENT2,
                        font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("Dark.TLabel", background=BG2, foreground=FG,
                        font=("Microsoft YaHei UI", 9))

        # Buttons
        style.configure("Run.TButton",
                        background=ACCENT, foreground="white",
                        font=("Microsoft YaHei UI", 10, "bold"),
                        borderwidth=0, padding=(20, 8))
        style.map("Run.TButton",
                  background=[("active", "#ff6b81"), ("disabled", "#555")])

        # Entries
        style.configure("TEntry", fieldbackground=ENTRY_BG, foreground=FG,
                        insertcolor=FG, borderwidth=1)

        # Combobox
        style.configure("TCombobox", fieldbackground=ENTRY_BG, foreground=FG,
                        arrowcolor=FG)

        # Labelframe
        style.configure("TLabelframe", background=BG, foreground=ACCENT2,
                        borderwidth=1, relief="solid")
        style.configure("TLabelframe.Label", background=BG, foreground=ACCENT2,
                        font=("Microsoft YaHei UI", 9, "bold"))

    # ═══════════════════════════════════════════════════════════
    # Notebook
    # ═══════════════════════════════════════════════════════════

    def _build_notebook(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=4, pady=(4, 0))

    # ═══════════════════════════════════════════════════════════
    # Helper: labelled entry row
    # ═══════════════════════════════════════════════════════════

    def _entry_row(self, parent: tk.Widget, label: str, default: str,
                   width: int = 10) -> tk.Entry:
        frame = ttk.Frame(parent, style="Dark.TFrame")
        frame.pack(fill=tk.X, pady=2)
        ttk.Label(frame, text=label, width=20, anchor=tk.W,
                  style="Dark.TLabel").pack(side=tk.LEFT, padx=(8, 4))
        var = tk.StringVar(value=default)
        entry = tk.Entry(frame, textvariable=var, width=width,
                         bg=ENTRY_BG, fg=FG, insertbackground=FG,
                         relief="flat", font=("Consolas", 10))
        entry.pack(side=tk.LEFT, padx=4)
        return entry

    def _section_label(self, parent: tk.Widget, text: str):
        lbl = ttk.Label(parent, text=text, style="Header.TLabel")
        lbl.pack(anchor=tk.W, padx=12, pady=(12, 4))

    # ═══════════════════════════════════════════════════════════
    # Output panel
    # ═══════════════════════════════════════════════════════════

    def _make_output(self, parent: tk.Widget, height: int = 18) -> scrolledtext.ScrolledText:
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=(4, 8))

        output = scrolledtext.ScrolledText(
            frame, wrap=tk.WORD, height=height,
            bg="#0a0a15", fg="#c0ffc0", insertbackground=FG,
            font=("Consolas", 10), relief="flat",
            borderwidth=1, highlightthickness=1,
            highlightbackground=BORDER, highlightcolor=ACCENT2,
        )
        output.pack(fill=tk.BOTH, expand=True)
        return output

    # ═══════════════════════════════════════════════════════════
    # Tab 1: Dual-Protocol Fusion
    # ═══════════════════════════════════════════════════════════

    def _build_fusion_tab(self):
        tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(tab, text="  🔗 双协议融合  ")

        # ── Inputs ──
        input_frame = ttk.Frame(tab, style="Dark.TFrame")
        input_frame.pack(fill=tk.X, padx=8, pady=(8, 0))

        self._section_label(input_frame, "▸ 场景参数")
        row1 = ttk.Frame(input_frame, style="Dark.TFrame")
        row1.pack(fill=tk.X)
        self.fus_drone_id = self._entry_row(row1, "无人机 ID", "DJI-2024-A17F")
        self.fus_lat = self._entry_row(row1, "起始纬度", "30.5728")
        self.fus_lon = self._entry_row(row1, "起始经度", "104.0668")
        self.fus_alt = self._entry_row(row1, "起始高度 (m)", "120")
        self.fus_dur = self._entry_row(row1, "飞行时长 (s)", "30")
        self.fus_seed = self._entry_row(row1, "随机种子", "42")

        row2 = ttk.Frame(input_frame, style="Dark.TFrame")
        row2.pack(fill=tk.X)
        self.fus_protos_var = tk.StringVar(value="WIFI + BLE")
        ttk.Label(row2, text="协议组合", width=20, anchor=tk.W,
                  style="Dark.TLabel").pack(side=tk.LEFT, padx=(8, 4))
        proto_combo = ttk.Combobox(row2, textvariable=self.fus_protos_var,
                                   values=["WIFI + BLE", "WIFI + BLE + NR", "WIFI only", "ALL"],
                                   state="readonly", width=14)
        proto_combo.pack(side=tk.LEFT, padx=4)

        self.fus_dedup_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(row2, text="启用去重", variable=self.fus_dedup_var,
                        style="Dark.TLabel").pack(side=tk.LEFT, padx=(20, 8))

        # Run button
        btn_frame = ttk.Frame(input_frame, style="Dark.TFrame")
        btn_frame.pack(fill=tk.X, pady=(10, 4))
        run_btn = tk.Button(btn_frame, text="▶  运行双协议融合",
                            bg=ACCENT, fg="white", font=("Microsoft YaHei UI", 10, "bold"),
                            relief="flat", padx=20, pady=6,
                            activebackground="#ff6b81", activeforeground="white",
                            cursor="hand2",
                            command=self._run_fusion)
        run_btn.pack(side=tk.LEFT, padx=12)

        # ── Output ──
        self._section_label(tab, "▸ 运行结果")
        self.fus_output = self._make_output(tab, height=18)

    def _run_fusion(self):
        self.fus_output.delete(1.0, tk.END)
        self.status_var.set("正在运行双协议融合...")
        _run_in_thread(self._run_fusion_thread)

    def _run_fusion_thread(self):
        old_stdout = sys.stdout
        sys.stdout = RedirectText(self.fus_output)
        try:
            from rid_fusion.models import ProtocolType
            from rid_fusion.fusion import RIDFusionEngine, compare_single_vs_fused

            proto_str = self.fus_protos_var.get()
            proto_map = {
                "WIFI + BLE": [ProtocolType.WIFI_BEACON, ProtocolType.BLE_ADVB],
                "WIFI + BLE + NR": [ProtocolType.WIFI_BEACON, ProtocolType.BLE_ADVB, ProtocolType.NR_BROADCAST],
                "WIFI only": [ProtocolType.WIFI_BEACON],
                "ALL": [ProtocolType.WIFI_BEACON, ProtocolType.BLE_ADVB, ProtocolType.NR_BROADCAST, ProtocolType.LORAWAN],
            }
            protocols = proto_map.get(proto_str, [ProtocolType.WIFI_BEACON, ProtocolType.BLE_ADVB])

            engine = RIDFusionEngine(
                protocols=protocols,
                d_model=128,
                seed=int(self.fus_seed.get()),
                enable_dedup=self.fus_dedup_var.get(),
            )

            print("=" * 55)
            print(f"  双协议融合 — {proto_str}  |  seed={self.fus_seed.get()}")
            print("=" * 55)

            result = compare_single_vs_fused(
                engine,
                drone_id=self.fus_drone_id.get(),
                start_lat=float(self.fus_lat.get()),
                start_lon=float(self.fus_lon.get()),
                start_alt=float(self.fus_alt.get()),
                duration_s=float(self.fus_dur.get()),
            )

            print(f"\n  单协议 Token 数:")
            for k, v in result["single_protocol"].items():
                print(f"    {k:12s}: {v:4d}")

            print(f"\n  融合结果:")
            print(f"    融合后 Token:     {result['fused_total']:4d}")
            print(f"    多协议富集 Token: {result['fused_enriched']:4d}")
            print(f"    富集比例:         {result['enrichment_ratio']*100:5.1f}%")

            print(f"\n  轨迹采样 (前 5 点):")
            for pt in result["trajectory"][:5]:
                print(f"    t={pt['timestamp_utc']:4.1f}s  "
                      f"pos=({pt['lat_deg']:.5f}, {pt['lon_deg']:.5f})  "
                      f"alt={pt['alt_m']:.1f}m  vel=({pt['vx_ms']:.1f}, {pt['vy_ms']:.1f}) m/s")

            print(f"\n  ✓ 完成 — {len(result['fused_tokens'])} 个 FusedToken 输出")

        except Exception as e:
            print(f"\n  ✗ 错误: {e}")
            import traceback; traceback.print_exc()
        finally:
            sys.stdout = old_stdout
            self.root.after(0, lambda: self.status_var.set("双协议融合完成 — Ready"))

    # ═══════════════════════════════════════════════════════════
    # Tab 2: Meta-Learning
    # ═══════════════════════════════════════════════════════════

    def _build_metalearn_tab(self):
        tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(tab, text="  🧠 元学习适配  ")

        input_frame = ttk.Frame(tab, style="Dark.TFrame")
        input_frame.pack(fill=tk.X, padx=8, pady=(8, 0))

        self._section_label(input_frame, "▸ 元学习参数")
        row = ttk.Frame(input_frame, style="Dark.TFrame")
        row.pack(fill=tk.X)
        self.meta_seed = self._entry_row(row, "随机种子", "42")
        self.meta_shots = self._entry_row(row, "适配样本数 (shots)", "10")
        self.meta_episodes = self._entry_row(row, "元训练轮数", "50")
        self.meta_inner_lr = self._entry_row(row, "内循环学习率", "0.01")
        self.meta_outer_lr = self._entry_row(row, "外循环学习率", "0.001")

        btn_frame = ttk.Frame(input_frame, style="Dark.TFrame")
        btn_frame.pack(fill=tk.X, pady=(10, 4))
        tk.Button(btn_frame, text="▶  运行元学习演示",
                  bg=ACCENT, fg="white", font=("Microsoft YaHei UI", 10, "bold"),
                  relief="flat", padx=20, pady=6,
                  activebackground="#ff6b81", activeforeground="white",
                  cursor="hand2", command=self._run_metalearn).pack(side=tk.LEFT, padx=12)

        self._section_label(tab, "▸ 运行结果")
        self.meta_output = self._make_output(tab, height=18)

    def _run_metalearn(self):
        self.meta_output.delete(1.0, tk.END)
        self.status_var.set("正在运行元学习演示...")
        _run_in_thread(self._run_metalearn_thread)

    def _run_metalearn_thread(self):
        old_stdout = sys.stdout
        sys.stdout = RedirectText(self.meta_output)
        try:
            from rid_fusion.meta_learner import simulate_meta_learning_demo

            print("=" * 55)
            print("  跨城市元学习 — Chengdu(WiFi+BLE) → Shenzhen(5G NR)")
            print("=" * 55)

            result = simulate_meta_learning_demo(seed=int(self.meta_seed.get()))

            print(f"\n  源域 (成都):")
            print(f"    Wi-Fi tokens: {result['n_source_tokens']['wifi']}")
            print(f"    BLE tokens:   {result['n_source_tokens']['ble']}")

            print(f"\n  目标域 (深圳):")
            print(f"    5G NR tokens: {result['n_target_tokens']}")
            print(f"    适配样本数:   {result['n_adaptation_shots']}")

            print(f"\n  适配结果:")
            print(f"    随机投影损失:   {result['random_projection_loss']:.6f}")
            print(f"    10-shot 适配损失: {result['adapted_loss']:.6f}")
            print(f"    提升倍数:       {result['improvement_factor']:.1f}x")
            verdict = "✓ 显著提升" if result['improvement_factor'] > 1.5 else "△ 有限提升"
            print(f"    判定:           {verdict}")

            if result["meta_train_losses"]:
                L0 = result["meta_train_losses"][0]
                LN = result["meta_train_losses"][-1]
                red = (1 - LN / L0) * 100
                print(f"\n  元训练收敛:")
                print(f"    初始 loss: {L0:.6f}")
                print(f"    最终 loss: {LN:.6f}")
                print(f"    下降:      {red:.1f}%")

        except Exception as e:
            print(f"\n  ✗ 错误: {e}")
            import traceback; traceback.print_exc()
        finally:
            sys.stdout = old_stdout
            self.root.after(0, lambda: self.status_var.set("元学习演示完成 — Ready"))

    # ═══════════════════════════════════════════════════════════
    # Tab 3: Deduplication
    # ═══════════════════════════════════════════════════════════

    def _build_dedup_tab(self):
        tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(tab, text="  📦 去重降本  ")

        input_frame = ttk.Frame(tab, style="Dark.TFrame")
        input_frame.pack(fill=tk.X, padx=8, pady=(8, 0))

        self._section_label(input_frame, "▸ 去重参数")
        row = ttk.Frame(input_frame, style="Dark.TFrame")
        row.pack(fill=tk.X)
        self.dedup_seed = self._entry_row(row, "随机种子", "42")
        self.dedup_window = self._entry_row(row, "时间窗口 (s)", "0.5")
        self.dedup_maxdiff = self._entry_row(row, "位置容差 (m)", "50.0")
        self.dedup_dur = self._entry_row(row, "飞行时长 (s)", "20")

        btn_frame = ttk.Frame(input_frame, style="Dark.TFrame")
        btn_frame.pack(fill=tk.X, pady=(10, 4))
        tk.Button(btn_frame, text="▶  运行去重分析",
                  bg=ACCENT, fg="white", font=("Microsoft YaHei UI", 10, "bold"),
                  relief="flat", padx=20, pady=6,
                  activebackground="#ff6b81", activeforeground="white",
                  cursor="hand2", command=self._run_dedup).pack(side=tk.LEFT, padx=12)

        self._section_label(tab, "▸ 运行结果")
        self.dedup_output = self._make_output(tab, height=18)

    def _run_dedup(self):
        self.dedup_output.delete(1.0, tk.END)
        self.status_var.set("正在运行去重分析...")
        _run_in_thread(self._run_dedup_thread)

    def _run_dedup_thread(self):
        old_stdout = sys.stdout
        sys.stdout = RedirectText(self.dedup_output)
        try:
            from rid_fusion.models import ProtocolType
            from rid_fusion.signals import RIDSignalSimulator, generate_drone_trajectory
            from rid_fusion.tokenizer import deduplicate_tokens

            sim = RIDSignalSimulator(
                [ProtocolType.WIFI_BEACON, ProtocolType.BLE_ADVB, ProtocolType.NR_BROADCAST],
                seed=int(self.dedup_seed.get()),
            )
            traj = generate_drone_trajectory(
                "DJI-Shenzhen-B7F2", start_lat=22.5431, start_lon=113.9544,
                start_alt=80.0, duration_s=float(self.dedup_dur.get()),
            )
            all_tokens = []
            for pt in traj:
                all_tokens.extend(sim.observe(
                    drone_id=pt["drone_id"], timestamp_utc=pt["timestamp_utc"],
                    lat_deg=pt["lat_deg"], lon_deg=pt["lon_deg"], alt_m=pt["alt_m"],
                    vx_ms=pt["vx_ms"], vy_ms=pt["vy_ms"], vz_ms=pt["vz_ms"],
                ))

            print("=" * 55)
            print("  跨协议 Token 去重分析")
            print("=" * 55)

            print(f"\n  去重前:")
            print(f"    总 Token 数: {len(all_tokens)}")
            counts = {}
            for t in all_tokens:
                counts[t.protocol.value] = counts.get(t.protocol.value, 0) + 1
            for proto, cnt in sorted(counts.items()):
                print(f"      {proto:16s}: {cnt:3d}")

            window = float(self.dedup_window.get())
            max_diff = float(self.dedup_maxdiff.get())
            deduped = deduplicate_tokens(all_tokens, time_window_s=window,
                                         max_position_diff_m=max_diff)

            reduction = (1 - len(deduped) / max(len(all_tokens), 1)) * 100
            print(f"\n  去重后 (窗口={window}s, 容差={max_diff}m):")
            print(f"    去重后 Token: {len(deduped)}")
            print(f"    Token 减少:   {reduction:.1f}%")

            with_sources = sum(1 for t in deduped
                               if len(t.protocol_payload.get("dedup_sources", [])) >= 2)
            outliers = sum(1 for t in deduped
                           if len(t.protocol_payload.get("dedup_outliers", [])) > 0)
            print(f"\n  信息保留:")
            print(f"    含 2+ 来源协议: {with_sources}/{len(deduped)} ({with_sources/max(len(deduped),1)*100:.0f}%)")
            print(f"    位置异常排除:   {outliers} 个 token 组")

            if deduped:
                multi = [t for t in deduped if len(t.protocol_payload.get("dedup_sources", [])) >= 2]
                if multi:
                    ex = multi[0]
                    print(f"\n  示例合并 Token:")
                    print(f"    drone={ex.drone_id}  ts={ex.timestamp_utc:.1f}s")
                    print(f"    pos=({ex.lat_deg:.4f}, {ex.lon_deg:.4f})")
                    print(f"    sources={ex.protocol_payload.get('dedup_sources', [])}")
                    out = ex.protocol_payload.get("dedup_outliers", [])
                    if out:
                        print(f"    outliers={out}")

            print(f"\n  ✓ 完成")

        except Exception as e:
            print(f"\n  ✗ 错误: {e}")
            import traceback; traceback.print_exc()
        finally:
            sys.stdout = old_stdout
            self.root.after(0, lambda: self.status_var.set("去重分析完成 — Ready"))

    # ═══════════════════════════════════════════════════════════
    # Tab 4: Tests
    # ═══════════════════════════════════════════════════════════

    def _build_tests_tab(self):
        tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(tab, text="  ✅ 测试套件  ")

        info_frame = ttk.Frame(tab, style="Dark.TFrame")
        info_frame.pack(fill=tk.X, padx=8, pady=(8, 0))

        self._section_label(info_frame, "▸ 单元测试 (17 项)")
        desc = ttk.Label(info_frame,
                         text="运行全部 17 个测试：8 核心 + 9 边界用例。\n"
                              "覆盖空输入、单协议、极端噪声、位置异常检测、复杂轨迹等。",
                         style="Dark.TLabel", wraplength=600)
        desc.pack(anchor=tk.W, padx=12, pady=(0, 4))

        btn_frame = ttk.Frame(info_frame, style="Dark.TFrame")
        btn_frame.pack(fill=tk.X, pady=(6, 4))
        tk.Button(btn_frame, text="▶  运行全部测试",
                  bg=ACCENT, fg="white", font=("Microsoft YaHei UI", 10, "bold"),
                  relief="flat", padx=20, pady=6,
                  activebackground="#ff6b81", activeforeground="white",
                  cursor="hand2", command=self._run_tests).pack(side=tk.LEFT, padx=12)

        self._section_label(tab, "▸ 测试输出")
        self.test_output = self._make_output(tab, height=18)

    def _run_tests(self):
        self.test_output.delete(1.0, tk.END)
        self.status_var.set("正在运行测试套件...")
        _run_in_thread(self._run_tests_thread)

    def _run_tests_thread(self):
        old_stdout = sys.stdout
        sys.stdout = RedirectText(self.test_output)
        try:
            import subprocess
            result = subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "tests" / "test_core.py")],
                capture_output=True, text=True, cwd=str(PROJECT_ROOT),
            )
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr[:500])
            if result.returncode == 0:
                print("\n" + "=" * 55)
                print("  ✓ 全部测试通过")
            else:
                print("\n" + "=" * 55)
                print("  ✗ 部分测试失败 — 请检查上方输出")
        except Exception as e:
            print(f"\n  ✗ 错误: {e}")
            import traceback; traceback.print_exc()
        finally:
            sys.stdout = old_stdout
            self.root.after(0, lambda: self.status_var.set("测试完成 — Ready"))

    # ═══════════════════════════════════════════════════════════
    # Tab 5: Config
    # ═══════════════════════════════════════════════════════════

    def _build_config_tab(self):
        tab = ttk.Frame(self.notebook, style="TFrame")
        self.notebook.add(tab, text="  ⚙ 配置  ")

        self._section_label(tab, "▸ 全局超参数 (FusionConfig)")

        cfg_frame = ttk.Frame(tab, style="Dark.TFrame")
        cfg_frame.pack(fill=tk.X, padx=8, pady=4)

        from rid_fusion.models import FusionConfig
        cfg = FusionConfig()

        params = [
            ("d_model", "嵌入维度", cfg.d_model),
            ("n_heads", "注意力头数", cfg.n_heads),
            ("spatial_dim", "空间编码维度", cfg.spatial_dim),
            ("temporal_dim", "时间编码维度", cfg.temporal_dim),
            ("signal_dim", "信号编码维度", cfg.signal_dim),
            ("protocol_dim", "协议编码维度", cfg.protocol_dim),
            ("context_dim", "气象编码维度", cfg.context_dim),
            ("seed", "全局随机种子", cfg.seed),
            ("inner_lr", "MAML 内循环学习率", cfg.inner_lr),
            ("outer_lr", "MAML 外循环学习率", cfg.outer_lr),
            ("n_inner_steps", "MAML 内循环步数", cfg.n_inner_steps),
            ("time_window_s", "去重时间窗口 (s)", cfg.time_window_s),
            ("max_position_diff_m", "去重位置容差 (m)", cfg.max_position_diff_m),
        ]

        for i, (key, label, default) in enumerate(params):
            row = ttk.Frame(cfg_frame, style="Dark.TFrame")
            row.pack(fill=tk.X, pady=1)
            ttk.Label(row, text=f"  {label}", width=24, anchor=tk.W,
                      style="Dark.TLabel").pack(side=tk.LEFT, padx=(8, 4))
            val_label = ttk.Label(row, text=str(default), width=10, anchor=tk.W,
                                  style="Dark.TLabel",
                                  font=("Consolas", 10))
            val_label.pack(side=tk.LEFT, padx=4)

        note = ttk.Label(tab,
                         text="\n修改超参数请编辑 rid_fusion/models.py 中的 FusionConfig 数据类，\n"
                              "或在各模块构造函数中传入自定义值。",
                         style="TLabel", wraplength=500, justify=tk.LEFT)
        note.pack(anchor=tk.W, padx=16, pady=(12, 4))

        self._section_label(tab, "▸ 项目信息")
        info = ttk.Label(tab,
                         text="rid-fusion v0.2.0\n"
                              "Multi-Protocol Remote ID Signal Fusion Engine\n"
                              "空御科技总体部  |  MIT License\n"
                              "依赖: numpy ≥ 1.24, scipy ≥ 1.10\n"
                              "测试: 17/17 通过",
                         style="TLabel", justify=tk.LEFT)
        info.pack(anchor=tk.W, padx=16, pady=4)

    # ═══════════════════════════════════════════════════════════
    # Launch
    # ═══════════════════════════════════════════════════════════

    def run(self):
        self.root.mainloop()


def main():
    app = RIDFusionGUI()
    app.run()


if __name__ == "__main__":
    main()
