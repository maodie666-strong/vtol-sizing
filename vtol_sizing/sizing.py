"""垂直起降飞行器总体参数估算与重量闭合求解。

问题定义
--------
给定任务需求（航程、悬停时间、爬升高度）与设计参数（桨盘载荷、
翼载、电池比能量等），求解满足任务的最小起飞重量 MTOW。

这是一个隐式方程
    MTOW = W_struct(MTOW) + W_prop + W_rotor + W_avionics + W_payload + W_battery(MTOW)
因为 MTOW 影响悬停/巡航功率，功率影响能量，能量决定电池重量，
电池重量又回到 MTOW。用带松弛因子的不动点迭代求解。

坐标与符号约定
--------------
- 力单位 N，功率单位 W（对外报告时换算 kW）
- 悬停与爬升取海平面密度（最恶劣工况），巡航取巡航高度密度
- 正向输出 MTOW；反向可通过 hover_endurance_minutes 评估纯悬停航时
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace
from typing import Any

from . import atmosphere, energy, rotor, weight, wing

G0 = 9.80665


@dataclass
class VTOLConfig:
    """eVTOL 总体方案配置（默认值取中型载人 eVTOL 量级）。"""

    name: str = "reference-vtol"

    # ---- 旋翼系统 ----
    n_rotor: int = 8
    rotor_radius: float = 0.90          # 单桨半径 [m]
    fm: float = 0.72                    # 品质因数
    eta_motor: float = 0.90             # 电机效率
    eta_transmission: float = 0.97      # 传动 / 线束效率

    # ---- 机翼 ----
    wing_area: float = 6.0              # 机翼参考面积 [m^2]
    aspect_ratio: float = 8.0           # 展弦比
    cd0: float = 0.028                  # 零升阻力系数
    oswald_e: float = 0.80              # 奥斯瓦尔德效率因子
    cl_max: float = 1.50                # 可用最大升力系数

    # ---- 重量估算参数 ----
    structure_fraction: float = 0.30    # 结构重量 / MTOW
    avionics_mass: float = 25.0         # 航电 + 系统 [kg]
    payload_mass: float = 200.0         # 有效载荷 [kg]
    motor_specific_power: float = 5000.0    # 动力系统功率密度 [W/kg]
    rotor_specific_thrust: float = 250.0    # 旋翼系统推力密度 [N/kg]
    power_margin: float = 1.15          # 装机功率裕度
    thrust_to_weight: float = 1.30      # 设计推重比

    # ---- 电池 ----
    battery_specific_energy: float = 250.0   # 系统级比能量 [Wh/kg]
    battery_usable_fraction: float = 0.85    # 可用放电深度

    # ---- 任务剖面 ----
    hover_time_s: float = 60.0          # 起降悬停总时间 [s]
    climb_rate: float = 3.0             # 爬升率 [m/s]
    climb_altitude: float = 300.0       # 爬升高度 [m]
    cruise_speed: float = 55.0          # 巡航速度 [m/s]
    cruise_altitude: float = 500.0      # 巡航高度 [m]
    cruise_prop_efficiency: float = 0.78    # 巡航螺旋桨效率
    reserve_fraction: float = 0.20      # 能量储备系数
    range_km: float = 100.0             # 任务航程 [km]

    # ---- 求解控制 ----
    mtow_guess: float = 1000.0
    relax: float = 1.50
    tol: float = 1e-7
    max_iter: int = 400

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SizingResult:
    """求解结果。"""

    converged: bool
    iterations: int
    residual: float
    warning: str
    config: VTOLConfig

    # 重量
    mtow: float
    mass_breakdown: dict[str, float]

    # 功率
    hover_power: float
    climb_power: float
    cruise_power: float
    max_shaft_power: float
    installed_power: float

    # 载荷参数
    wing_loading: float
    disk_loading: float
    thrust_to_weight: float

    # 气动
    cl_cruise: float
    lift_to_drag: float
    induced_ratio: float

    # 能量
    energy_breakdown: dict[str, float]
    mission_energy: float
    required_energy: float

    # 时间
    time_breakdown: dict[str, float]

    def to_flat_dict(self) -> dict[str, Any]:
        """展开为单层字典，便于写入 CSV。"""
        row: dict[str, Any] = {
            "converged": self.converged,
            "iterations": self.iterations,
            "mtow_kg": self.mtow,
            "wing_loading_Nm2": self.wing_loading,
            "disk_loading_Nm2": self.disk_loading,
            "hover_power_kW": self.hover_power / 1000.0,
            "climb_power_kW": self.climb_power / 1000.0,
            "cruise_power_kW": self.cruise_power / 1000.0,
            "installed_power_kW": self.installed_power / 1000.0,
            "cl_cruise": self.cl_cruise,
            "lift_to_drag": self.lift_to_drag,
            "mission_energy_kWh": self.mission_energy / 1000.0,
            "required_energy_kWh": self.required_energy / 1000.0,
        }
        for k, v in self.mass_breakdown.items():
            row[f"mass_{k}_kg"] = v
        return row


def evaluate(cfg: VTOLConfig, mtow: float) -> dict[str, Any]:
    """给定 MTOW，正向计算一次任务闭环所需的各项结果。

    这是求解器的「残差函数」：返回的 total_mass 与传入的 mtow 之差
    即为重量闭合残差。
    """
    w = mtow * G0
    rho_sl = atmosphere.density(0.0)
    rho_cruise = atmosphere.density(cfg.cruise_altitude)
    area = rotor.disk_area(cfg.n_rotor, cfg.rotor_radius)

    # --- 功率 ---
    p_hover = rotor.to_electric_power(
        rotor.hover_shaft_power(w, rho_sl, area, cfg.fm),
        cfg.eta_motor,
        cfg.eta_transmission,
    )
    p_climb = rotor.to_electric_power(
        rotor.axial_climb_shaft_power(w, rho_sl, area, cfg.fm, cfg.climb_rate),
        cfg.eta_motor,
        cfg.eta_transmission,
    )
    p_cruise, polar = wing.cruise_power(
        w,
        rho_cruise,
        cfg.cruise_speed,
        cfg.wing_area,
        cfg.aspect_ratio,
        cfg.cd0,
        cfg.oswald_e,
        cfg.cruise_prop_efficiency,
        cfg.eta_motor,
    )

    # --- 任务时间 ---
    t_climb = cfg.climb_altitude / cfg.climb_rate
    t_cruise = cfg.range_km * 1000.0 / cfg.cruise_speed

    # --- 能量 ---
    e_hover = p_hover * cfg.hover_time_s / 3600.0
    e_climb = p_climb * t_climb / 3600.0
    e_cruise = p_cruise * t_cruise / 3600.0
    e_mission = (e_hover + e_climb + e_cruise) * (1.0 + cfg.reserve_fraction)
    e_required = energy.usable_energy(e_mission, cfg.battery_usable_fraction)
    m_battery = e_required / cfg.battery_specific_energy

    # --- 重量 ---
    p_max_shaft = max(
        p_hover / (cfg.eta_motor * cfg.eta_transmission),
        p_climb / (cfg.eta_motor * cfg.eta_transmission),
        p_cruise / cfg.eta_motor,
    )
    max_thrust = w * cfg.thrust_to_weight

    m_structure = weight.structure_mass(mtow, cfg.structure_fraction)
    m_propulsion = weight.propulsion_mass(p_max_shaft, cfg.motor_specific_power)
    m_rotor = weight.rotor_system_mass(max_thrust, cfg.rotor_specific_thrust)

    total = (
        m_structure
        + m_propulsion
        + m_rotor
        + cfg.avionics_mass
        + cfg.payload_mass
        + m_battery
    )

    cd_induced = polar["cl"] ** 2 / (math.pi * cfg.aspect_ratio * cfg.oswald_e)

    return {
        "total_mass": total,
        "force": w,
        "hover_power": p_hover,
        "climb_power": p_climb,
        "cruise_power": p_cruise,
        "max_shaft_power": p_max_shaft,
        "installed_power": p_max_shaft * cfg.power_margin,
        "mass_breakdown": {
            "structure": m_structure,
            "propulsion": m_propulsion,
            "rotor": m_rotor,
            "avionics": cfg.avionics_mass,
            "payload": cfg.payload_mass,
            "battery": m_battery,
        },
        "wing_loading": w / cfg.wing_area,
        "disk_loading": w / area,
        "thrust_to_weight": cfg.thrust_to_weight,
        "cl_cruise": polar["cl"],
        "lift_to_drag": polar["lift_to_drag"],
        "induced_ratio": cd_induced / polar["cd"] if polar["cd"] > 0 else 0.0,
        "energy_breakdown": {
            "hover": e_hover,
            "climb": e_climb,
            "cruise": e_cruise,
        },
        "mission_energy": e_mission,
        "required_energy": e_required,
        "time_breakdown": {"hover": cfg.hover_time_s, "climb": t_climb, "cruise": t_cruise},
    }


def size(cfg: VTOLConfig) -> SizingResult:
    """求解满足任务剖面的最小起飞重量（重量闭合不动点迭代）。"""
    mtow = cfg.mtow_guess
    residual = float("nan")
    last: dict[str, Any] = {}

    for iteration in range(1, cfg.max_iter + 1):
        last = evaluate(cfg, mtow)
        residual = last["total_mass"] - mtow
        mtow += cfg.relax * residual

        if not math.isfinite(mtow) or mtow <= 0.0:
            return _failure_result(cfg, "迭代发散：任务需求超出能量可行性边界", iteration, residual)

        if abs(residual) / max(mtow, 1e-9) < cfg.tol:
            last = evaluate(cfg, mtow)
            warning = ""
            if last["cl_cruise"] > cfg.cl_max:
                warning = (
                    f"巡航升力系数 {last['cl_cruise']:.2f} 超过可用上限 {cfg.cl_max:.2f}，"
                    "需加大机翼面积或提高巡航速度"
                )
            return _make_result(cfg, True, iteration, residual, warning, mtow, last)

    warning = f"达到最大迭代次数 {cfg.max_iter} 仍未收敛（残余 {residual:.3e} kg）"
    return _make_result(cfg, False, cfg.max_iter, residual, warning, mtow, last)


def _make_result(
    cfg: VTOLConfig,
    converged: bool,
    iterations: int,
    residual: float,
    warning: str,
    mtow: float,
    data: dict[str, Any],
) -> SizingResult:
    return SizingResult(
        converged=converged,
        iterations=iterations,
        residual=residual,
        warning=warning,
        config=cfg,
        mtow=mtow,
        mass_breakdown=data["mass_breakdown"],
        hover_power=data["hover_power"],
        climb_power=data["climb_power"],
        cruise_power=data["cruise_power"],
        max_shaft_power=data["max_shaft_power"],
        installed_power=data["installed_power"],
        wing_loading=data["wing_loading"],
        disk_loading=data["disk_loading"],
        thrust_to_weight=data["thrust_to_weight"],
        cl_cruise=data["cl_cruise"],
        lift_to_drag=data["lift_to_drag"],
        induced_ratio=data["induced_ratio"],
        energy_breakdown=data["energy_breakdown"],
        mission_energy=data["mission_energy"],
        required_energy=data["required_energy"],
        time_breakdown=data["time_breakdown"],
    )


def _failure_result(
    cfg: VTOLConfig, warning: str, iteration: int, residual: float
) -> SizingResult:
    data = evaluate(cfg, cfg.mtow_guess)
    return _make_result(cfg, False, iteration, residual, warning, float("nan"), data)


def with_params(cfg: VTOLConfig, **overrides: Any) -> VTOLConfig:
    """返回替换了部分参数的配置副本（sweep / sensitivity 使用）。"""
    return replace(cfg, **overrides)


def hover_endurance_minutes(cfg: VTOLConfig, all_up_mass: float, battery_mass: float) -> float:
    """纯悬停航时 [min]（给定起飞重量与电池重量）。

    用于小尺寸机型验证：结构/动力重量已知时，反推悬停能撑多久。
    """
    w = all_up_mass * G0
    rho_sl = atmosphere.density(0.0)
    area = rotor.disk_area(cfg.n_rotor, cfg.rotor_radius)
    p_hover = rotor.to_electric_power(
        rotor.hover_shaft_power(w, rho_sl, area, cfg.fm),
        cfg.eta_motor,
        cfg.eta_transmission,
    )
    available = battery_mass * cfg.battery_specific_energy * cfg.battery_usable_fraction
    return energy.endurance_seconds(available, p_hover) / 60.0
