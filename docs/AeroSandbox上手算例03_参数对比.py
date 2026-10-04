"""
上手算例 3：参数对比——AR 和弯度这两个"设计旋钮"各自怎么拨动 CL-α 曲线。

2×2 组合：AR ∈ {4, 8} × 翼型 ∈ {NACA2412（带弯度）, NACA0012（对称）}
控制变量法（复试实验设计加分点）：所有机翼面积固定 S=8 m²，
改 AR 只改展长和弦长（b=√(AR·S)，c=S/b），保证对比公平。

口算预告（算例 02 留的思考题）：
  ② AR=4 理论斜率 = 2π·4/6 = 4.19 /rad = 0.0731 /°
     比 AR=8 的 0.0877 /° 低 16.7% —— 下面和程序对答案。
  ① 矩形翼够不到椭圆环量理论上限的原因 = 展向环量非椭圆分布，
     诱导效率 e<1 —— 本次从程序输出里直接读 e 验证。
预期结论：
  - AR 只改【斜率】：AR 越小，同样迎角下升力越低、诱导阻力越大；
  - 弯度只做【平移】：同一 AR 下 2412 与 0012 斜率应几乎相同，
    差别全在零升迎角（0012 对称 → α_L0≈0）。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import aerosandbox as asb

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

S = 8.0          # 机翼面积 m²，全程固定
ALPHAS = np.linspace(-8, 16, 25)
FIT_MASK = (ALPHAS >= -6) & (ALPHAS <= 10)


def make_wing(ar: float, airfoil_name: str) -> asb.Wing:
    """矩形翼：面积固定 S，展弦比 ar。"""
    b = np.sqrt(ar * S)      # 展长
    c = S / b                # 弦长
    return asb.Wing(
        symmetric=True,
        xsecs=[
            asb.WingXSec(xyz_le=[0, 0, 0], chord=c, airfoil=asb.Airfoil(airfoil_name)),
            asb.WingXSec(xyz_le=[0, b / 2, 0], chord=c, airfoil=asb.Airfoil(airfoil_name)),
        ],
    )


def theory_slope(ar: float) -> float:
    """升力线理论（椭圆环量）CL_α，单位 /°。"""
    return (2 * np.pi * ar / (ar + 2)) * np.pi / 180


rows = []   # (label, alphas, CL, slope, alpha_L0, e)
fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=150)

for airfoil_name, color in [("naca2412", "tab:orange"), ("naca0012", "tab:blue")]:
    for ar, ls in [(8, "-"), (4, "--")]:
        aero = asb.AeroBuildup(
            airplane=asb.Airplane(wings=[make_wing(ar, airfoil_name)]),
            op_point=asb.OperatingPoint(velocity=30, alpha=ALPHAS),
        ).run()
        CL = np.array(aero["CL"])
        slope, intercept = np.polyfit(ALPHAS[FIT_MASK], CL[FIT_MASK], 1)
        alpha_L0 = -intercept / slope

        # 诱导效率 e：wing_aero_components 按机翼分组，e 为标量
        e = float(aero["wing_aero_components"][0].oswalds_efficiency)

        dev = slope / theory_slope(ar) - 1
        rows.append((f"{airfoil_name.upper()} AR={ar}", slope, alpha_L0, e, dev))

        ax.plot(ALPHAS, CL, ls, color=color, lw=1.2,
                label=f"{airfoil_name.upper()}  AR={ar}  ({slope:.4f} /°)")

# ---- 口算答案 ② 对程序 ----
s_2412_ar4 = [r for r in rows if r[0] == "NACA2412 AR=4"][0][1]
mental = theory_slope(4)
print("=== 思考题②对答案 ===")
print(f"口算 AR=4 理论斜率: {mental:.4f} /°（比 AR=8 低 {(1 - mental / theory_slope(8)) * 100:.1f}%）")
print(f"程序 AR=4  拟合斜率: {s_2412_ar4:.4f} /°（偏差 {(s_2412_ar4 / mental - 1) * 100:+.1f}%）")

print("\n=== 全部结果 ===")
print(f"{'方案':<18}{'斜率 /°':>10}{'α_L0 /°':>10}{'诱导效率e':>10}{'斜率偏差':>10}")
for label, slope, alpha_L0, e, dev in rows:
    print(f"{label:<18}{slope:>10.4f}{alpha_L0:>10.2f}{e:>10.3f}{dev:>+9.1f}%")

# ---- 断言：弯度只平移不改斜率 ----
for ar in (8, 4):
    s2412 = [r for r in rows if r[0] == f"NACA2412 AR={ar}"][0][1]
    s0012 = [r for r in rows if r[0] == f"NACA0012 AR={ar}"][0][1]
    assert abs(s2412 - s0012) < 0.002, f"弯度改变了斜率？AR={ar}: {s2412} vs {s0012}"
print("\n断言通过：同一 AR 下带弯度与对称翼型斜率几乎一致（弯度只平移曲线）✓")

# ---- 画图 ----
ax.axvspan(12, 16, color="red", alpha=0.08)
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("迎角 α (°)")
ax.set_ylabel("升力系数 CL")
ax.set_title("算例03：AR 与弯度对 CL-α 曲线的影响（S=8 m² 固定，V=30 m/s）")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "output" / "算例03_参数对比_CL-alpha.png"
fig.savefig(out)
print(f"图已保存: {out}")
