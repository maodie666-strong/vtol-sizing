"""分系统重量估算。

估算口径
--------
- 结构重量：按 MTOW 的经验比例估算（复合材料机体 0.25 ~ 0.35）。
- 动力系统（电机 + 电调 + 线束）：按所需峰值功率与功率密度估算。
- 旋翼系统（桨叶 + 桨毂 + 变距机构）：按峰值推力与推力密度估算。
- 航电、载荷：固定值。
- 电池：由任务能量需求反推 —— 这是重量闭合的驱动量。

这是一个「重量闭合」问题：MTOW 影响能量需求，能量需求决定电池
重量，电池重量又回到 MTOW。sizing.py 用不动点迭代求解。
"""

from __future__ import annotations


def structure_mass(mtow: float, structure_fraction: float) -> float:
    """结构重量 [kg]。"""
    return structure_fraction * mtow


def propulsion_mass(max_shaft_power: float, specific_power: float) -> float:
    """动力系统重量 [kg]。

    max_shaft_power  峰值轴功率 [W]
    specific_power   电机+电调系统功率密度 [W/kg]，航空用高功率密度
                     电机典型 3000 ~ 8000 W/kg
    """
    return max_shaft_power / specific_power


def rotor_system_mass(max_thrust: float, specific_thrust: float) -> float:
    """旋翼系统重量 [kg]。

    max_thrust      全机最大推力 [N]
    specific_thrust 旋翼系统推力密度 [N/kg]，小型螺旋桨约 200 ~ 350
    """
    return max_thrust / specific_thrust
