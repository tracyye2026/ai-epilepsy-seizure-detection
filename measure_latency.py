# -*- coding: utf-8 -*-
"""measure_latency.py —— 实测 SeizureNet 在 CPU 上的推理时延/吞吐/参数量（A 档②补实验）"""
from __future__ import annotations
import sys, time, statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import torch
from config import Config
from models.seizure_net import build_model, count_parameters

torch.set_num_threads(max(1, torch.get_num_threads()))
DEVICE = torch.device("cpu")

cfg = Config()
cfg.n_channels = 18
model = build_model(cfg).to(DEVICE).eval()
n_params = count_parameters(model)

# 载入一个 LOPO 折的最终模型权重（仅取 state_dict）
ck = ROOT / "work" / "ckpt" / "chbmit_full18_chb08.pt"
if ck.exists():
    sd = torch.load(ck, map_location="cpu", weights_only=False)["state_dict"]
    model.load_state_dict(sd)
    print(f"已载入权重: {ck.name}")

T = int(round(cfg.win_sec * cfg.sfreq))  # 512
print(f"参数量 = {n_params:,}  (≈{n_params/1e4:.2f} 万)")
print(f"输入窗口: (C={cfg.n_channels}, T={T}) = 4 s × 128 Hz")

# ---- 单窗口时延（模拟流式：每窗口独立前向）----
x = torch.randn(1, cfg.n_channels, T)
# 预热
for _ in range(20):
    with torch.no_grad():
        model(x)
times = []
for _ in range(200):
    t0 = time.perf_counter()
    with torch.no_grad():
        model(x)
    times.append((time.perf_counter() - t0) * 1000)  # ms
ms_mean = statistics.mean(times)
ms_std = statistics.pstdev(times)
print(f"单窗口推理时延: {ms_mean:.3f} ± {ms_std:.3f} ms  (min {min(times):.3f} ms)")

# ---- 批量吞吐：模拟 1 小时连续记录（1800 个 4s 窗，步长 2s）----
batch = torch.randn(1800, cfg.n_channels, T)
t0 = time.perf_counter()
with torch.no_grad():
    for i in range(0, 1800, 64):
        model(batch[i:i+64])
elapsed = time.perf_counter() - t0
windows_per_hour = 1800
print(f"1 小时(1800 窗)批量推理耗时: {elapsed:.2f} s  →  {windows_per_hour/elapsed:.1f} 窗/s")
print(f"等效实时占用: 每 2 s 产生 1 窗，推理仅需 {ms_mean:.3f} ms ≪ 2000 ms，实时裕度 >{(2000/ms_mean):.0f}×")

# ---- 模型体积（FP32 磁盘占用）----
import io
buf = io.BytesIO()
torch.save(model.state_dict(), buf)
mb = len(buf.getvalue()) / 1e6
print(f"模型权重体积(FP32): {mb:.2f} MB")

# ---- 128 Hz / 2 s 步长的端到端报警时延上界 ----
# 平滑窗 8 s + 最短报警 10 s → 报警判定起点最早在发作后 8 s，且事件需 >=10 s 才报出
smooth_sec = cfg.smooth_sec
min_event_sec = cfg.min_event_sec
print(f"端到端报警时延上界(设计值): 平滑 {smooth_sec:.0f} s + 最短报警 {min_event_sec:.0f} s ≈ {smooth_sec+min_event_sec:.0f} s")
# CPU 型号：parallel_info() 首行是 PyTorch 版本号，真正含 CPU 的行以 "CPU" 开头
_info = torch.__config__.parallel_info() if hasattr(torch, "__config__") else ""
_cpu_line = next((ln.strip() for ln in _info.splitlines()
                  if ln.strip().lower().startswith("cpu")), "n/a")
print("CPU 型号:", _cpu_line)
