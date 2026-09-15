"""电池与能量模型。

只做一阶处理：标称比能量、可用放电深度（DoD）、以及一个可选的
放电末段电压跌落修正。不做电化学建模。

工程上真正要小心的是「可用比能量 ≠ 电芯比能量」：
系统级（含 BMS、结构封装、热管理）通常比电芯低 15% ~ 25%。
本模块用 usable_fraction 统一表达这部分损失，避免重复计入。
"""

from __future__ import annotations


def usable_energy(required_energy: float, usable_fraction: float) -> float:
    """从「任务必须消耗的能量」反推「电池必须携带的能量」。

    required_energy  任务消耗能量 [Wh]
    usable_fraction  可用放电深度（0 ~ 1），锂电池工程取值 0.80 ~ 0.90
    """
    if not 0.0 < usable_fraction <= 1.0:
        raise ValueError("usable_fraction 必须落在 (0, 1] 区间")
    return required_energy / usable_fraction


def pack_mass(required_energy: float, specific_energy: float, usable_fraction: float) -> float:
    """电池包重量 [kg]。

    required_energy  任务消耗能量 [Wh]
    specific_energy  系统级比能量 [Wh/kg]，当前锂电工程值 200 ~ 300
    usable_fraction  可用放电深度
    """
    return usable_energy(required_energy, usable_fraction) / specific_energy


def endurance_seconds(available_energy_wh: float, power_w: float) -> float:
    """给定可用能量与功耗，返回可维持时间 [s]。"""
    if power_w <= 0:
        return float("inf")
    return available_energy_wh * 3600.0 / power_w
