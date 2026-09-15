"""vtol_sizing —— 垂直起降飞行器总体参数快速权衡内核。

用法
----
    from vtol_sizing import VTOLConfig, size

    cfg = VTOLConfig(range_km=100.0, payload_mass=200.0)
    result = size(cfg)
    print(result.mtow)

模块划分
--------
atmosphere  ISA 标准大气
rotor       旋翼动量理论 + 品质因数（悬停 / 轴流爬升）
wing        机翼升力线与阻力极曲线（巡航）
weight      分系统重量估算
energy      电池与能量换算
sizing      总体参数求解与重量闭合
sweep       参数扫描与敏感性分析
"""

from .sizing import (
    G0,
    SizingResult,
    VTOLConfig,
    evaluate,
    hover_endurance_minutes,
    size,
    with_params,
)

__all__ = [
    "G0",
    "VTOLConfig",
    "SizingResult",
    "evaluate",
    "size",
    "with_params",
    "hover_endurance_minutes",
]

__version__ = "0.1.0"
