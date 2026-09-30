"""代理模型的输入参数空间与采样。

为什么要"设计"采样
------------------
训练代理模型的第一步不是搭网络，而是决定**在哪里取数据点**。

朴素做法是在参数范围内均匀随机撒点，但随机撒点会出现"聚簇"和"空洞"——
有些区域点挤在一起（浪费算力），有些区域完全没点（网络在那儿纯靠猜）。

拉丁超立方采样（Latin Hypercube Sampling, LHS）的做法是：
把每一维都切成 N 个等概率小区间，保证**每一维、每个小区间里有且仅有一个样本**，
然后各维独立打乱顺序。这样在样本数不变的前提下，覆盖性远好于纯随机。

这是飞行器多学科设计优化（MDO）里的标准做法。
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# 参数空间：(字段名, 下界, 上界, 中文名, 单位)
# 范围根据 eVTOL 工程可行域设定，保证绝大多数采样点都能让求解器收敛。
# ---------------------------------------------------------------------------
PARAM_SPACE: list[tuple[str, float, float, str, str]] = [
    ("rotor_radius",              0.50, 1.50, "桨半径",        "m"),
    ("n_rotor",                   4.0,  10.0, "旋翼数量",      "个"),
    ("fm",                        0.58, 0.82, "品质因数",      "-"),
    ("eta_motor",                 0.78, 0.95, "电机效率",      "-"),
    ("wing_area",                 2.0,  12.0, "机翼面积",      "m2"),
    ("aspect_ratio",              5.0,  12.0, "展弦比",        "-"),
    ("structure_fraction",        0.22, 0.38, "结构重量系数",  "-"),
    ("battery_specific_energy", 200.0, 380.0, "电池比能量",    "Wh/kg"),
    ("cruise_speed",             35.0,  75.0, "巡航速度",      "m/s"),
    ("range_km",                 30.0, 200.0, "任务航程",      "km"),
]

# ---------------------------------------------------------------------------
# 输出量：(字段名, 中文名, 单位, 换算系数)
# 换算系数把原始量纲变成更适合展示的量纲（W -> kW，Wh -> kWh）。
# ---------------------------------------------------------------------------
OUTPUT_SPACE: list[tuple[str, str, str, float]] = [
    ("mtow",             "起飞重量",    "kg",   1.0),
    ("battery_mass",     "电池重量",    "kg",   1.0),
    ("hover_power",      "悬停功率",    "kW",   1e-3),
    ("installed_power",  "装机功率",    "kW",   1e-3),
    ("required_energy",  "需装机能量",  "kWh",  1e-3),
    ("lift_to_drag",     "巡航升阻比",  "-",    1.0),
]

PARAM_NAMES = [p[0] for p in PARAM_SPACE]
PARAM_LABELS = {p[0]: p[3] for p in PARAM_SPACE}
PARAM_UNITS = {p[0]: p[4] for p in PARAM_SPACE}
PARAM_LOWER = np.array([p[1] for p in PARAM_SPACE], dtype=float)
PARAM_UPPER = np.array([p[2] for p in PARAM_SPACE], dtype=float)

OUTPUT_NAMES = [o[0] for o in OUTPUT_SPACE]
OUTPUT_LABELS = {o[0]: o[1] for o in OUTPUT_SPACE}
OUTPUT_UNITS = {o[0]: o[2] for o in OUTPUT_SPACE}
OUTPUT_SCALE = np.array([o[3] for o in OUTPUT_SPACE], dtype=float)


def latin_hypercube(n_samples: int, n_dims: int, rng: np.random.Generator) -> np.ndarray:
    """生成 n_samples × n_dims 的拉丁超立方样本，取值落在 [0, 1)。

    步骤：
      1. 把每维切成 n_samples 个等长区间；
      2. 每个区间内随机取一点 —— 这保证每维覆盖均匀；
      3. 每一维独立打乱这些点的顺序 —— 这保证各维之间的组合是随机的。

    第 3 步是关键：如果不打乱，第 i 个样本在所有维度上都落在第 i 个区间，
    样本会分布在一条对角线上，失去空间探索能力。
    """
    samples = np.empty((n_samples, n_dims), dtype=float)
    for j in range(n_dims):
        # 每个区间的取点位置：[k + U(0,1)] / N, k = 0..N-1
        points = (np.arange(n_samples) + rng.random(n_samples)) / n_samples
        samples[:, j] = rng.permutation(points)
    return samples


def sample_parameters(n_samples: int, seed: int = 0) -> np.ndarray:
    """在参数空间内做拉丁超立方采样，返回形状 (n_samples, n_params) 的物理量矩阵。"""
    rng = np.random.default_rng(seed)
    unit = latin_hypercube(n_samples, len(PARAM_SPACE), rng)
    return PARAM_LOWER + unit * (PARAM_UPPER - PARAM_LOWER)
