"""
上手算例 2：迎角扫描——画 CL-α 曲线，并与升力线理论对比。

题目：还是算例 1 那个机翼（AR=8 矩形翼、NACA2412、弦长 1 m、30 m/s），
把迎角从 -8° 扫到 +16°，看升力系数怎么变。

理论预期（复试要能口算）：
  1. 有限翼展修正（普朗特升力线，椭圆环量）：CL_α = 2π·AR/(AR+2) /rad
     AR=8 → 2π×8/10 ≈ 5.03 /rad ≈ 0.0877 /°
  2. NACA2412 是带弯度翼型，零升迎角 α_L0 ≈ -2°，所以曲线整体右移；
  3. 纯 VLM 是线性位势理论：诱导部分只给直线、算不出失速。
     但 AeroBuildup 的二维翼型气动用的是 NeuralFoil（XFOIL 数据训练的
     神经网络），含粘性效应——高迎角段会看到曲线开始弯下，这是二维
     数据在起作用；整机失速形态（展向分离顺序、CL_max）仍需试验/CFD 标定。

一致性校核：α=5° 的点必须复现算例 1 的 CL=0.6675。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import aerosandbox as asb

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]  # 中文显示
plt.rcParams["axes.unicode_minus"] = False

# ---- 几何：与算例 1 完全一致 ----
wing = asb.Wing(
    symmetric=True,
    xsecs=[
        asb.WingXSec(xyz_le=[0, 0, 0], chord=1.0, airfoil=asb.Airfoil("naca2412")),
        asb.WingXSec(xyz_le=[0, 4, 0], chord=1.0, airfoil=asb.Airfoil("naca2412")),
    ],
)
airplane = asb.Airplane(wings=[wing])
AR = 8.0  # b^2/S = 8^2/8

# ---- 一次调用扫 25 个迎角（OperatingPoint 天然支持向量化）----
alphas = np.linspace(-8, 16, 25)
aero = asb.AeroBuildup(
    airplane=airplane,
    op_point=asb.OperatingPoint(velocity=30, alpha=alphas),
).run()
CL = np.array(aero["CL"])

# ---- 校核 1：α=5° 要对上算例 1 ----
CL_at_5 = CL[np.argmin(np.abs(alphas - 5))]
assert abs(CL_at_5 - 0.6675) < 0.005, f"与算例1不一致: {CL_at_5=}"

# ---- 线性段拟合（-6° ~ 10°，避开端点）----
mask = (alphas >= -6) & (alphas <= 10)
slope, intercept = np.polyfit(alphas[mask], CL[mask], 1)  # /°
alpha_L0 = -intercept / slope

CL_alpha_theory_rad = 2 * np.pi * AR / (AR + 2)          # /rad
CL_alpha_theory = CL_alpha_theory_rad * np.pi / 180       # /rad → /°

print(f"拟合斜率   CL_α = {slope:.4f} /°")
print(f"理论斜率   CL_α = {CL_alpha_theory:.4f} /°   (2π·AR/(AR+2))")
print(f"偏差            {(slope / CL_alpha_theory - 1) * 100:+.1f} %")
print(f"拟合零升迎角 α_L0 = {alpha_L0:.2f} °   (NACA2412 手册值约 -2°)")
print(f"α=5° 处 CL     = {CL_at_5:.4f}  (算例1: 0.6675 ✓)")

# ---- 画图 ----
fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
ax.plot(alphas, CL, "o", ms=4, label="AeroSandbox VLM")
ax.plot(alphas, slope * alphas + intercept, "-", lw=1,
        label=f"线性段拟合 {slope:.4f} /°")
ax.plot(alphas, CL_alpha_theory * (alphas - alpha_L0), "--", lw=1,
        label=f"升力线理论 {CL_alpha_theory:.4f} /°")
ax.axvspan(12, 16, color="red", alpha=0.08)
ax.annotate("α>12°：NeuralFoil 二维粘性数据\n开始体现失速趋势，VLM 点偏离直线\n（整机 CL_max 仍需试验/CFD 标定）",
            xy=(13.5, CL[np.argmax(alphas >= 13)]), xytext=(5.5, 1.05),
            arrowprops=dict(arrowstyle="->", lw=0.8), fontsize=9)
ax.axhline(0, color="gray", lw=0.5)
ax.set_xlabel("迎角 α (°)")
ax.set_ylabel("升力系数 CL")
ax.set_title("算例02：AR=8 矩形翼 NACA2412 的 CL-α 曲线（V=30 m/s）")
ax.grid(alpha=0.3)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
out = Path(__file__).resolve().parents[1] / "output" / "算例02_CL-alpha曲线.png"
fig.savefig(out)
print(f"图已保存: {out}")
