"""导出 Python 内核的全精度基准值，供 JS 移植版做跨语言一致性验证。

运行（在 vtol-sizing 目录下）：
    python vtol-web/dump_reference.py

输出：vtol-web/reference.json
"""

from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from vtol_sizing import VTOLConfig, hover_endurance_minutes, size
from vtol_sizing.sweep import sensitivity, sweep_1d, linspace


def main() -> None:
    cfg = VTOLConfig(
        name="中型载货 eVTOL",
        range_km=100.0,
        payload_mass=200.0,
        n_rotor=8,
        rotor_radius=0.90,
        wing_area=6.0,
    )
    res = size(cfg)

    ref: dict = {
        "case1": {
            "iterations": res.iterations,
            "converged": res.converged,
            "mtow": res.mtow,
            "mass_breakdown": res.mass_breakdown,
            "hover_power_w": res.hover_power,
            "climb_power_w": res.climb_power,
            "cruise_power_w": res.cruise_power,
            "installed_power_w": res.installed_power,
            "wing_loading": res.wing_loading,
            "disk_loading": res.disk_loading,
            "cl_cruise": res.cl_cruise,
            "lift_to_drag": res.lift_to_drag,
            "induced_ratio": res.induced_ratio,
            "energy_breakdown_wh": res.energy_breakdown,
            "mission_energy_wh": res.mission_energy,
            "required_energy_wh": res.required_energy,
        },
        "case2": {
            "endurance_min": hover_endurance_minutes(
                VTOLConfig(
                    n_rotor=4,
                    rotor_radius=0.1397,
                    fm=0.62,
                    eta_motor=0.85,
                    eta_transmission=1.00,
                    battery_specific_energy=220.0,
                ),
                all_up_mass=2.0,
                battery_mass=0.5,
            )
        },
        "sensitivity_order": [r["field"] for r in sensitivity(cfg, delta=0.10)],
        "sensitivity_swing": {
            r["field"]: r["swing_pct"] for r in sensitivity(cfg, delta=0.10)
        },
        "scan_rotor_radius": [
            {"radius": i["value"], "mtow": i["mtow"]}
            for i in sweep_1d(cfg, "rotor_radius", linspace(0.60, 1.30, 15))
        ],
    }

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(ref, fh, ensure_ascii=False, indent=2)
    print(f"基准已导出：{out}")


if __name__ == "__main__":
    main()
