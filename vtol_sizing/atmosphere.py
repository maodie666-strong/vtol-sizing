"""ISA 标准大气模型（对流层，0 ~ 11 km）。

说明
----
高度 h 按位势高度处理。在 11 km 以下，位势高度与几何高度的差异
小于 1%，对概念设计阶段的估算精度没有影响。

本模块只依赖标准库。
"""

from __future__ import annotations

T0 = 288.15        # 海平面标准温度 [K]
P0 = 101325.0      # 海平面标准压力 [Pa]
LAPSE = 0.0065     # 温度递减率 [K/m]
R_AIR = 287.05287  # 空气气体常数 [J/(kg·K)]
G0 = 9.80665       # 标准重力加速度 [m/s^2]


def temperature(h: float) -> float:
    """位势高度 h [m] -> 温度 [K]。"""
    return T0 - LAPSE * h


def pressure(h: float) -> float:
    """位势高度 h [m] -> 压力 [Pa]。"""
    return P0 * (1.0 - LAPSE * h / T0) ** (G0 / (R_AIR * LAPSE))


def density(h: float) -> float:
    """位势高度 h [m] -> 空气密度 [kg/m^3]。"""
    return pressure(h) / (R_AIR * temperature(h))
