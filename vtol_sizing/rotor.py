"""旋翼 / 螺旋桨气动模型。

模型定位
--------
概念设计阶段使用「动量理论 + 品质因数(FM)」方法，这是 NDARC、NASA
设计手册以及大多数 eVTOL 概念设计工具采用的标准做法：

    P_ideal = T * v_i ,  v_i = sqrt(T / (2 rho A))
    P_shaft = P_ideal / FM

FM 把剖面功率、桨尖损失、入流非均匀性等二阶效应一次性归并进来，
小型螺旋桨典型值 0.60 ~ 0.70，优化过的大直径旋翼 0.75 ~ 0.82。

本模块只负责轴流工况（悬停 + 垂直爬升）。前飞由机翼承载，
巡航功率在 sizing 中按机翼阻力计算。
"""

from __future__ import annotations

import math


def disk_area(n_rotor: int, radius: float) -> float:
    """桨盘总面积 [m^2]。"""
    return n_rotor * math.pi * radius ** 2


def hover_induced_velocity(thrust: float, rho: float, area: float) -> float:
    """悬停诱导速度 v_h = sqrt(T / (2 rho A)) [m/s]。"""
    return math.sqrt(thrust / (2.0 * rho * area))


def hover_shaft_power(thrust: float, rho: float, area: float, fm: float) -> float:
    """悬停轴功率 [W]。"""
    return thrust * hover_induced_velocity(thrust, rho, area) / fm


def axial_climb_shaft_power(
    thrust: float, rho: float, area: float, fm: float, climb_rate: float
) -> float:
    """轴流爬升轴功率 [W]。

    爬升状态下的入流速度满足 v_i = v_h^2 / (V_c + v_i)，取物理解（正根）：

        v_i = ( -V_c + sqrt(V_c^2 + 4 v_h^2) ) / 2

    理想功率 P = T (V_c + v_i)，再除以 FM 计入剖面功率与入流非均匀性。
    爬升率为 0 时退化回悬停解。
    """
    vh = hover_induced_velocity(thrust, rho, area)
    vi = (-climb_rate + math.sqrt(climb_rate ** 2 + 4.0 * vh ** 2)) / 2.0
    return thrust * (climb_rate + vi) / fm


def to_electric_power(
    shaft_power: float, eta_motor: float, eta_transmission: float = 1.0
) -> float:
    """轴功率 -> 电功率 [W]（考虑电机与传动效率）。"""
    return shaft_power / (eta_motor * eta_transmission)
