# -*- coding: utf-8 -*-
"""
config.py —— 全局配置（所有脚本共用）

用途：集中管理"数据路径 / 信号预处理 / 模型 / 训练 / 事件级评估"的超参数。
修改本文件或通过命令行 --xxx 覆盖即可，不需要改动任何算法代码。

作者：面向《癫痫杂志》"人工智能（AI）与癫痫"专题投稿
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    return Path(v) if v else default


@dataclass
class Config:
    # ===================== 1. 数据与路径 =====================
    data_root: str = str(_env_path("EEG_DATA_ROOT", PROJECT_ROOT / "data_raw"))
    work_root: str = str(_env_path("EEG_WORK_ROOT", PROJECT_ROOT / "work"))
    dataset: str = "chbmit"          # chbmit | siena

    # ===================== 2. 信号预处理 =====================
    sfreq: int = 128                 # 统一重采样频率(Hz)。128 Hz 足以覆盖发作期节律，且显著降低算力
    bandpass_lo: float = 0.5         # 高通(Hz)
    bandpass_hi: float = 45.0        # 低通(Hz)
    notch_freq: float = 0.0          # 陷波频率；0 = 关闭(45Hz低通已滤除工频)
    win_sec: float = 4.0             # 分析窗长(s)
    hop_sec: float = 2.0             # 窗滑动步长(s)，50% 重叠
    ictal_overlap: float = 0.5       # 窗口与发作区间的重叠比例 ≥ 该值才标为"发作期"
    guard_after_sec: float = 30.0    # 发作结束后该时长内的窗口不纳入负样本(避免发作后期/衰减期标签污染)
    neg_pos_ratio: int = 8           # 训练集中 负样本:正样本 的抽样比例
    # 无发作记录的负样本抽样数/条。
    # 【关键修正 2026-09-21】原实现仅从"含发作记录"中抽取负样本（8×正样本数），
    # 导致 CHB-MIT 685 条记录里 545 条无发作记录（79%）完全未进入训练集；
    # 模型从未见过"长时程正常脑电"，在新患者上发作间期概率中位数可达 0.89，
    # 固定阈值下 FA/24h 高达 200+。此处显式为无发作记录抽样负样本。
    neg_windows_per_record: int = 120
    # 训练集负:正上限。负样本过多会挤占正样本、降低灵敏度，故设上限后随机抽样。
    max_neg_pos_ratio: int = 6
    clip_z: float = 10.0             # 稳健 Z 分数截断阈值，抑制大幅伪迹
    min_present_channels: int = 16   # 可用导联数少于该值的记录将被跳过

    # ===================== 3. 模型 =====================
    model: str = "seizurenet"        # seizurenet | resnet1d
    n_channels: int = 18             # 输入导联数(由通道子集决定)
    base_filters: int = 32
    hidden: int = 64                 # BiGRU 隐层维度
    gru_layers: int = 1
    dropout: float = 0.3
    use_attention: bool = True       # 时间注意力池化

    # ===================== 4. 训练 =====================
    epochs: int = 30
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    loss: str = "weighted_bce"       # weighted_bce | focal
    focal_alpha: float = 0.75
    focal_gamma: float = 2.0
    pos_weight: float = -1.0         # <0 时自动按 负/正 样本数计算
    early_stop: int = 8
    seed: int = 42
    num_workers: int = 0
    max_train_windows: int = 120000   # 每折训练集窗口上限（超出按比例抽样，控制内存）
    device: str = "auto"             # auto | cpu | cuda
    amp: bool = False                # 混合精度(仅 CUDA 生效)
    # 数据增强
    aug_noise: float = 0.05          # 高斯噪声标准差(相对稳健 Z 单位)
    aug_scale: float = 0.1           # 幅度缩放抖动
    aug_chan_drop: float = 0.1       # 随机置零导联比例(提升缺失导联鲁棒性)
    aug_shift: int = 64              # 随机时间平移的最大采样点数

    # ===================== 5. 事件级评估后处理 =====================
    threshold: float = 0.5           # 概率阈值；也可用 --tune_threshold 在验证集上搜索
    tune_threshold: bool = True      # 是否在验证患者上搜索最优阈值
    # 阈值标定方式：在验证患者上按**事件级**指标搜索
    #  （窗级 F1 最优的阈值在连续推理时会引发大量假报警，不适合临床场景）
    fa_budget: float = 10.0          # 可接受的假报警预算(次/24h)；在此约束下取灵敏度最高阈值
    thr_grid_hi: float = 0.995       # 阈值搜索上限（高阈值区对 FA 极敏感，必须覆盖到）
    # 后处理参数（经阈值-后处理敏感性扫描标定：见 _diag_thr.py / _diag_adapt.py）
    # 1) 4/6/3 相比最初的 2/4/2 已显著降低 FA；
    # 2) 进一步按"发作时长先验"收紧：CHB-MIT 发作时长 10 分位=14 s、中位=39 s，
    #    而伪迹多为秒级孤立尖峰 → 最短报警时长取 10 s 在临床上完全合理。
    #    实测 8 例高假报警患者：min_event 3→10 s 使汇总 FA/24h 由 236 降至 84，
    #    灵敏度由 0.510 降至 0.429（在补全负样本后该代价会明显变小）。
    smooth_sec: float = 8.0          # 概率序列滑动平均窗长
    merge_gap_sec: float = 10.0      # 间隔小于该值的报警合并为一次事件
    min_event_sec: float = 10.0      # 短于该值的报警丢弃(发作时长先验)
    min_overlap_sec: float = 1.0     # 报警与真实发作重叠 ≥ 该值判定为检出

    # 阈值模式：absolute=全局固定阈值（策略 A/C）；quantile=逐记录秩自适应阈值（策略 B）
    #   策略 B 的阈值 = max(该记录平滑概率的 thr_quantile 分位, thr_floor)
    #   动机：跨患者（乃至跨数据库）存在显著的输出分布漂移，
    #         实测部分患者发作间期概率中位数 0.96、另一些仅 0.02，固定阈值无法兼顾。
    #   三个参数 (thr_quantile, thr_floor, min_event_sec) 均在验证患者上按 FA 预算标定。
    threshold_mode: str = "absolute"   # "absolute" | "quantile"
    thr_quantile: float = 0.995        # 策略 B 的分位点
    thr_floor: float = 0.5             # 策略 B 的绝对概率下限（给"整条记录很干净"兜底）

    # ===================== 6. 其他 =====================
    cache_continuous: bool = False
    skip_existing: bool = True    # 分批次下载后重跑 preprocess 时复用已处理记录   # 是否缓存整段连续预处理信号(float16)。磁盘充足可开，能加速反复评估
    n_boot: int = 1000               # Bootstrap 重抽样次数(以患者为单位)
    log_level: str = "INFO"

    # ---------------- 工具方法 ----------------
    @property
    def data_dir(self) -> Path:
        return Path(self.data_root)

    @property
    def work_dir(self) -> Path:
        return Path(self.work_root)

    @property
    def index_dir(self) -> Path:
        return self.work_dir / "index" / self.dataset

    @property
    def windows_dir(self) -> Path:
        return self.work_dir / "windows" / self.dataset

    @property
    def ckpt_dir(self) -> Path:
        return self.work_dir / "ckpt"

    @property
    def result_dir(self) -> Path:
        return self.work_dir / "results"

    @property
    def figure_dir(self) -> Path:
        return self.work_dir / "figures"

    def ensure_dirs(self) -> "Config":
        for p in [self.index_dir, self.windows_dir, self.ckpt_dir,
                  self.result_dir, self.figure_dir]:
            p.mkdir(parents=True, exist_ok=True)
        return self

    def update(self, **kwargs) -> "Config":
        for k, v in kwargs.items():
            if v is None:
                continue
            if not hasattr(self, k):
                raise KeyError(f"未知配置项: {k}")
            setattr(self, k, v)
        return self

    def save(self, path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, path) -> "Config":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))

    def summary(self) -> str:
        keep = ["dataset", "sfreq", "win_sec", "hop_sec", "neg_pos_ratio", "model",
                "n_channels", "epochs", "batch_size", "lr", "loss", "threshold"]
        return " | ".join(f"{k}={getattr(self, k)}" for k in keep)


def add_config_args(parser):
    """把 Config 的字段注册到 argparse（命令行同名覆盖）。"""
    import argparse
    default = Config()
    for k, v in asdict(default).items():
        flag = "--" + k
        if isinstance(v, bool):
            parser.add_argument(flag, type=lambda s: str(s).lower() in ("1", "true", "yes", "y"),
                                default=None, help=f"bool, 默认 {v}")
        elif isinstance(v, int):
            parser.add_argument(flag, type=int, default=None, help=f"int, 默认 {v}")
        elif isinstance(v, float):
            parser.add_argument(flag, type=float, default=None, help=f"float, 默认 {v}")
        else:
            parser.add_argument(flag, type=str, default=None, help=f"str, 默认 {v}")
    return parser


def config_from_args(args) -> Config:
    """从 argparse.Namespace 构造配置；自动忽略不属于 Config 的参数（如 --folds）。"""
    cfg = Config()
    known = {f.name for f in fields(cfg)}
    d = {k: v for k, v in vars(args).items() if (k in known and v is not None)}
    cfg.update(**d)
    return cfg.ensure_dirs()
