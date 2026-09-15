"""参数扫描与敏感性分析。

概念设计阶段真正有价值的信息不是「某个方案多重」，而是
「哪个参数动一下，代价变化最大」。本模块提供：

- sweep_1d      单参数扫描
- sweep_2d      双参数网格扫描（为后续画等值线图准备）
- sensitivity   逐参数 ±扰动，给出对目标量的影响排序

所有函数都返回纯 Python 结构，方便直接喂给表格或绘图库。
"""

from __future__ import annotations

import math
from dataclasses import fields as dataclass_fields
from dataclasses import replace
from typing import Any, Callable, Iterable, Sequence

from .sizing import SizingResult, VTOLConfig, size

MetricFn = Callable[[SizingResult], float]

METRICS: dict[str, MetricFn] = {
    "mtow": lambda r: r.mtow,
    "battery_mass": lambda r: r.mass_breakdown["battery"],
    "empty_mass": lambda r: r.mtow - r.mass_breakdown["battery"] - r.config.payload_mass,
    "installed_power": lambda r: r.installed_power / 1000.0,
    "required_energy": lambda r: r.required_energy / 1000.0,
    "cruise_power": lambda r: r.cruise_power / 1000.0,
    "hover_power": lambda r: r.hover_power / 1000.0,
    "lift_to_drag": lambda r: r.lift_to_drag,
    "disk_loading": lambda r: r.disk_loading,
    "wing_loading": lambda r: r.wing_loading,
}

METRIC_UNITS: dict[str, str] = {
    "mtow": "kg",
    "battery_mass": "kg",
    "empty_mass": "kg",
    "installed_power": "kW",
    "required_energy": "kWh",
    "cruise_power": "kW",
    "hover_power": "kW",
    "lift_to_drag": "-",
    "disk_loading": "N/m2",
    "wing_loading": "N/m2",
}


def get_metric(result: SizingResult, metric: str) -> float:
    """按名称取目标量数值。"""
    if metric not in METRICS:
        raise KeyError(f"未知目标量 '{metric}'，可选：{', '.join(sorted(METRICS))}")
    return METRICS[metric](result)


def sweep_1d(
    cfg: VTOLConfig,
    field: str,
    values: Sequence[float],
    metric: str = "mtow",
) -> list[dict[str, float]]:
    """单参数扫描。

    返回 [{'value': v, 'metric': m, 'converged': bool}, ...]
    """
    _check_field(cfg, field)
    out: list[dict[str, float]] = []
    for v in values:
        res = size(replace(cfg, **{field: v}))
        out.append(
            {
                "value": float(v),
                "metric": get_metric(res, metric),
                "mtow": res.mtow,
                "battery_mass": res.mass_breakdown["battery"],
                "disk_loading": res.disk_loading,
                "wing_loading": res.wing_loading,
                "installed_power": res.installed_power / 1000.0,
                "lift_to_drag": res.lift_to_drag,
                "converged": res.converged,
            }
        )
    return out


def sweep_2d(
    cfg: VTOLConfig,
    field_x: str,
    values_x: Sequence[float],
    field_y: str,
    values_y: Sequence[float],
    metric: str = "mtow",
) -> dict[str, Any]:
    """双参数网格扫描，返回可直接画等值线的网格。"""
    _check_field(cfg, field_x)
    _check_field(cfg, field_y)
    grid: list[list[float]] = []
    for vy in values_y:
        row: list[float] = []
        for vx in values_x:
            res = size(replace(cfg, **{field_x: vx, field_y: vy}))
            row.append(get_metric(res, metric))
        grid.append(row)
    return {
        "x_field": field_x,
        "y_field": field_y,
        "x_values": list(values_x),
        "y_values": list(values_y),
        "metric": metric,
        "grid": grid,
    }


def sensitivity(
    cfg: VTOLConfig,
    fields: Iterable[str] | None = None,
    delta: float = 0.10,
    metric: str = "mtow",
) -> list[dict[str, Any]]:
    """逐参数敏感性分析。

    对每个参数施加 ±delta 的相对扰动，记录目标量的最大相对变化，
    按影响程度降序返回。

    返回 [{field, base, plus, minus, swing_pct, direction}, ...]
    ``swing_pct`` 为最大相对变化百分数，``direction`` 取 '+1' 表示参数
    增大使目标量增大，'-1' 表示减小，'?' 表示非线性或未收敛。
    """
    if fields is None:
        fields = _tunable_fields()

    base = get_metric(size(cfg), metric)
    rows: list[dict[str, Any]] = []

    for name in fields:
        current = getattr(cfg, name)
        if not isinstance(current, (int, float)) or isinstance(current, bool):
            continue
        if current == 0:
            continue

        hi = size(replace(cfg, **{name: current * (1.0 + delta)}))
        lo = size(replace(cfg, **{name: current * (1.0 - delta)}))
        v_hi = get_metric(hi, metric)
        v_lo = get_metric(lo, metric)

        if not (math.isfinite(v_hi) and math.isfinite(v_lo)) or base == 0:
            swing = float("inf")
            direction = "?"
        else:
            swing = 100.0 * max(abs(v_hi - base), abs(v_lo - base)) / abs(base)
            if v_hi > base and v_lo < base:
                direction = "+1"
            elif v_hi < base and v_lo > base:
                direction = "-1"
            else:
                direction = "?"

        rows.append(
            {
                "field": name,
                "base": float(base),
                "plus": v_hi,
                "minus": v_lo,
                "swing_pct": swing,
                "direction": direction,
            }
        )

    rows.sort(key=lambda r: (-r["swing_pct"] if math.isfinite(r["swing_pct"]) else 1e18))
    return rows


def _tunable_fields() -> list[str]:
    """可扫描的数值型参数名（排除求解控制与名称）。"""
    skip = {
        "name",
        "mtow_guess",
        "relax",
        "tol",
        "max_iter",
        "payload_mass",
        "cl_max",
        "reserve_fraction",
        "range_km",
        "hover_time_s",
    }
    out: list[str] = []
    for f in dataclass_fields(VTOLConfig):
        if f.name in skip:
            continue
        default = f.default
        if isinstance(default, (int, float)) and not isinstance(default, bool):
            out.append(f.name)
    return out


def _check_field(cfg: VTOLConfig, field: str) -> None:
    if field not in {f.name for f in dataclass_fields(VTOLConfig)}:
        raise KeyError(f"配置中不存在参数 '{field}'")


def linspace(start: float, stop: float, n: int) -> list[float]:
    """等间隔序列（避免为这点功能引入 numpy）。"""
    if n < 2:
        return [float(start)]
    step = (stop - start) / (n - 1)
    return [start + i * step for i in range(n)]


def to_csv(rows: Sequence[dict[str, Any]], path: str) -> None:
    """把扫描结果写成 CSV，供 Excel / 绘图使用。"""
    if not rows:
        return
    headers = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as fh:
        fh.write(",".join(headers) + "\n")
        for row in rows:
            fh.write(",".join(_fmt(row[h]) for h in headers) + "\n")


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)
