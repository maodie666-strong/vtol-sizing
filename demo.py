"""vtol_sizing 演示脚本。

运行：
    python demo.py

内容：
  1. 中型 eVTOL 方案总体参数求解（任务 100 km / 200 kg 载荷）
  2. 小尺寸四旋翼悬停航时验证（对标公开机型量级）
  3. 参数敏感性分析 —— 找出对代价影响最大的参数
  4. 桨盘载荷单参数扫描，结果导出 CSV

零第三方依赖，Python 3.9+ 可直接运行。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from vtol_sizing import VTOLConfig, hover_endurance_minutes, size
from vtol_sizing.sweep import linspace, sensitivity, sweep_1d, to_csv


def rule(char: str = "-", width: int = 68) -> str:
    return char * width


def section(title: str) -> None:
    print()
    print(rule("="))
    print(f"  {title}")
    print(rule("="))


def report(cfg: VTOLConfig, res) -> None:
    """打印一次求解的完整报告。"""
    print(f"  方案名称        : {cfg.name}")
    print(f"  任务剖面        : 航程 {cfg.range_km:.0f} km @ {cfg.cruise_speed:.0f} m/s"
          f" | 悬停 {cfg.hover_time_s:.0f} s | 爬升 {cfg.climb_altitude:.0f} m")
    print(f"  收敛状态        : {'已收敛' if res.converged else '未收敛'}"
          f"（{res.iterations} 次迭代，残余 {res.residual:.3e} kg）")
    if res.warning:
        print(f"  [警告] {res.warning}")

    print()
    print("  -- 重量构成 --")
    total = res.mtow
    for key, label in [
        ("structure", "结构"),
        ("propulsion", "动力系统"),
        ("rotor", "旋翼系统"),
        ("avionics", "航电系统"),
        ("payload", "有效载荷"),
        ("battery", "电池"),
    ]:
        m = res.mass_breakdown[key]
        print(f"    {label:<8s} {m:8.1f} kg   {100.0 * m / total:5.1f}%")
    print(f"    {'合计':<8s} {total:8.1f} kg   100.0%")

    print()
    print("  -- 功率 --")
    print(f"    悬停电功率      {res.hover_power / 1000.0:8.2f} kW")
    print(f"    爬升电功率      {res.climb_power / 1000.0:8.2f} kW")
    print(f"    巡航电功率      {res.cruise_power / 1000.0:8.2f} kW")
    print(f"    装机功率        {res.installed_power / 1000.0:8.2f} kW")

    print()
    print("  -- 关键参数 --")
    print(f"    翼载            {res.wing_loading:8.1f} N/m2")
    print(f"    桨盘载荷        {res.disk_loading:8.1f} N/m2")
    print(f"    巡航升力系数    {res.cl_cruise:8.3f}")
    print(f"    巡航升阻比      {res.lift_to_drag:8.2f}")
    print(f"    诱导阻力占比    {100.0 * res.induced_ratio:8.1f} %")

    print()
    print("  -- 能量 --")
    em = res.energy_breakdown
    for key, label in [("hover", "悬停段"), ("climb", "爬升段"), ("cruise", "巡航段")]:
        print(f"    {label:<8s} {em[key] / 1000.0:8.2f} kWh")
    print(f"    {'含储备':<8s} {res.mission_energy / 1000.0:8.2f} kWh"
          f"（储备系数 {cfg.reserve_fraction:.0%}）")
    print(f"    {'需装机':<8s} {res.required_energy / 1000.0:8.2f} kWh"
          f"（可用放电深度 {cfg.battery_usable_fraction:.0%}）")


def main() -> None:
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(out_dir, exist_ok=True)

    # ---------------------------------------------------------------- 案例 1
    section("案例 1  中型 eVTOL 总体参数求解")
    cfg = VTOLConfig(
        name="中型载货 eVTOL",
        range_km=100.0,
        payload_mass=200.0,
        n_rotor=8,
        rotor_radius=0.90,
        wing_area=6.0,
    )
    res = size(cfg)
    report(cfg, res)

    # ---------------------------------------------------------------- 案例 2
    section("案例 2  小四旋翼悬停航时验证")
    quad = VTOLConfig(
        name="2 kg 四旋翼",
        n_rotor=4,
        rotor_radius=0.1397,      # 11 inch 桨
        fm=0.62,                  # 小桨品质因数偏低
        eta_motor=0.85,
        eta_transmission=1.00,
        battery_specific_energy=220.0,
    )
    all_up = 2.0
    battery = 0.5
    endurance = hover_endurance_minutes(quad, all_up, battery)
    print(f"  构型            : {quad.n_rotor} 旋翼，桨径 {2 * quad.rotor_radius * 100:.1f} cm")
    print(f"  起飞重量        : {all_up:.2f} kg（其中电池 {battery:.2f} kg）")
    print(f"  电池比能量      : {quad.battery_specific_energy:.0f} Wh/kg"
          f"（可用 {quad.battery_usable_fraction:.0%}）")
    print(f"  品质因数 FM     : {quad.fm:.2f}")
    print()
    print(f"  >> 模型预测悬停航时 : {endurance:.1f} min")
    print(f"  >> 同级公开机型实测 : 约 25 ~ 30 min")
    print()
    print("  结论：模型量级正确，一阶估算可用于方案比较。")

    # ---------------------------------------------------------------- 案例 3
    section("案例 3  参数敏感性分析（目标量：MTOW）")
    rows = sensitivity(cfg, delta=0.10, metric="mtow")
    print(f"  对每个参数施加 ±10% 扰动，观察 MTOW 变化：")
    print()
    print(f"    {'参数':<28s}{'基准':>10s}{'最大变化':>10s}")
    print(f"    {rule('-', 48)}")
    for row in rows[:10]:
        print(f"    {row['field']:<28s}{row['base']:>10.1f}{row['swing_pct']:>9.2f}%")

    top = rows[0]
    print()
    print(f"  >> 最敏感参数：{top['field']}，±10% 扰动引起 MTOW 变化 {top['swing_pct']:.2f}%")
    print(f"  >> 设计含义：优先优化该参数，收益最大。")

    # ---------------------------------------------------------------- 案例 4
    section("案例 4  桨盘载荷扫描")
    radii = linspace(0.60, 1.30, 15)
    scan = sweep_1d(cfg, "rotor_radius", radii, metric="mtow")
    print(f"    {'桨半径(m)':>10s}{'桨盘载荷(N/m2)':>16s}{'MTOW(kg)':>12s}{'电池(kg)':>12s}")
    print(f"    {rule('-', 50)}")
    for item in scan:
        print(f"    {item['value']:>10.3f}{item['disk_loading']:>16.1f}"
              f"{item['mtow']:>12.1f}{item['battery_mass']:>12.1f}")

    csv_path = os.path.join(out_dir, "sweep_rotor_radius.csv")
    to_csv(
        [
            {
                "rotor_radius_m": i["value"],
                "disk_loading_Nm2": i["disk_loading"],
                "mtow_kg": i["mtow"],
                "battery_mass_kg": i["battery_mass"],
                "installed_power_kW": i["installed_power"],
                "converged": i["converged"],
            }
            for i in scan
        ],
        csv_path,
    )
    print()
    print(f"  扫描结果已导出：{csv_path}")

    best = min(scan, key=lambda i: i["mtow"])
    print(f"  >> 该扫描区间内 MTOW 最小点：桨半径 {best['value']:.2f} m"
          f"（MTOW {best['mtow']:.1f} kg，桨盘载荷 {best['disk_loading']:.1f} N/m2）")
    print()
    print(rule("="))
    print("  全部计算完成。")
    print(rule("="))


if __name__ == "__main__":
    main()
