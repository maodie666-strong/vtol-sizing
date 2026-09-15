"""机翼气动模型（升力线理论 + 经验阻力极曲线）。

适用工况：巡航段由机翼承载全部重量（lift + cruise 与倾转构型在
巡航状态均可近似这样处理）。垂直起降段由旋翼承载，见 rotor.py。
"""

from __future__ import annotations

import math

TWO_PI = 2.0 * math.pi


def lift_curve_slope(aspect_ratio: float) -> float:
    """有限展弦比机翼升力线斜率 a [1/rad]。

    采用 Helmbold 形式（二维斜率取 a0 = 2*pi，不可压、无后掠）：

        a = a0 * AR / (AR + 2)

    AR = 8 时 a ≈ 5.03 /rad，与风洞数据吻合良好。
    """
    return TWO_PI * aspect_ratio / (aspect_ratio + 2.0)


def wing_polar(
    weight: float,
    rho: float,
    speed: float,
    area: float,
    aspect_ratio: float,
    cd0: float,
    oswald_e: float,
) -> dict:
    """给定重量/密度/速度，解平飞配平并返回气动量。

    返回 dict：
        cl      升力系数
        cd      阻力系数
        lift_to_drag  升阻比
        drag    阻力 [N]
        dynamic_pressure  动压 [Pa]
    """
    q = 0.5 * rho * speed ** 2
    cl = weight / (q * area)
    cd = cd0 + cl ** 2 / (math.pi * aspect_ratio * oswald_e)
    return {
        "cl": cl,
        "cd": cd,
        "lift_to_drag": cl / cd if cd > 0 else float("inf"),
        "drag": q * area * cd,
        "dynamic_pressure": q,
    }


def cruise_power(
    weight: float,
    rho: float,
    speed: float,
    area: float,
    aspect_ratio: float,
    cd0: float,
    oswald_e: float,
    eta_prop: float,
    eta_motor: float,
) -> tuple[float, dict]:
    """巡航电功率 [W] 及气动量。

    巡航螺旋桨效率 eta_prop 已包含螺旋桨自身效率与安装损失。
    """
    polar = wing_polar(weight, rho, speed, area, aspect_ratio, cd0, oswald_e)
    p_shaft = polar["drag"] * speed / eta_prop
    return p_shaft / eta_motor, polar
