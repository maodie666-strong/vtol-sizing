"""
上手算例 4【动手题】：自己写一个最小涡格法（VLM），和 AeroSandbox 对数。
════════════════════════════════════════════════════════════════════
你在算例 02/03 里一直是"用"别人的 VLM。这次你把它"写"出来——
只留三个 TODO（约 30 行代码），物理和几何都已给你备好。

【原理一页纸】（对照 vortex_lattice_method.py 一起看，行号见下）
  1. 布涡：每条展向条带放一个"马蹄涡"——附着涡在 1/4 弦线（左端 A→右端 B），
     两条尾涡腿从 A、B 顺流向 +x 拖到无穷远（用 BIG=1e6·b 截断成有限丝）。
     → 对应源码 L227-233（left/right_vortex_vertices、collocation_points）
  2. 配平点：每个条带的控制点在 3/4 弦线中展向（1/4-3/4 规则，
     这是让二维极限精确等于 2π 的关键 placement）。
  3. 物面不穿透：控制点上 n·(V∞ + v_ind) = 0。v_ind 是所有马蹄涡（单位环量）
     诱导速度的线性叠加 → 线性方程组 AIC·Γ = -n·V∞，AIC[i,j] = n·v_j(CP_i)。
     → 对应源码 L281-312（calculate_induced_velocity_horseshoe → np.linalg.solve）
  4. 受力：Kutta-Joukowski，每条带 ΔL = ρ·V∞·Γ·Δy，CL = ΣΔL/(½ρV²S)。
     → 对应源码 L316-340（F = ρ V×l·Γ）
  Biot-Savart 有限直涡丝公式已帮你写好（filament_velocity），符号约定容易翻车，
  不作为本次练习点；但请你读懂它的两行注释。

【验收靶子】（已用 AeroSandbox asb.VLM 同离散验证过）
  - 同离散 = 弦向 1 格、全展 16 条均布条带、平板（NACA0012 弯度≈0）
  - 斜率 CL_α ≈ 0.0824 /°（asb.VLM 同离散 0.08241，允差 ±0.0015）
  - CL(α=5°) ≈ 0.412（允差 ±0.01）
  - 环量分布左右对称（Γ(y)≈Γ(-y)，数值判据 <1e-8）
  - 对照：升力线理论 2π·AR/(AR+2) = 0.0877 /°——你的结果会低 ~6%，
    这不是 bug：单弦向格马蹄涡模型的已知系统偏差（展向加密不救，
    弦向加密才是收敛方向，ZCode 已做收敛试验证实）。打印出来写进你的学习笔记。

【怎么交作业】填完三个 TODO → python 本文件 → 全部 PASS 后
  git add + commit，或直接把输出发给 ZCode 审。
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

# ══════════════════ 以下已备好，勿改 ══════════════════
b, S = 8.0, 8.0            # 展长 [m]，面积 [m²]
c = S / b                  # 弦长 = 1 m
AR = b * b / S
N = 16                     # 全展长条带数
dy = b / N
y_mid = -b / 2 + (np.arange(N) + 0.5) * dy   # 各条带展向中点

XV, XCP = 0.25 * c, 0.75 * c                 # 附着涡 / 配平点的弦向位置
A_pts = np.stack([np.full(N, XV), y_mid - dy / 2, np.zeros(N)], axis=1)
B_pts = np.stack([np.full(N, XV), y_mid + dy / 2, np.zeros(N)], axis=1)
CP = np.stack([np.full(N, XCP), y_mid, np.zeros(N)], axis=1)
n_hat = np.array([0.0, 0.0, 1.0])            # 平板法向
BIG = 1e6 * b                                # 无穷远的有限截断
RHO = 1.225

# 几何自检：附着涡、配平点、条带数
assert A_pts.shape == (N, 3) and abs(XV - 0.25) < 1e-12 and abs(XCP - 0.75) < 1e-12


def filament_velocity(P, p1, p2, gamma=1.0):
    """直涡丝 p1→p2（环量 gamma，右手定则）在点 P 的诱导速度 [m/s]。

    Biot-Savart：v = Γ/(4π) · (cosθ1 - cosθ2) / h · ê
      θ1、θ2 均相对同一丝轴方向 p1→p2 量取；
      ê = (r1×r2)/|r1×r2| 自动给出垂直方向，无需手调符号。
    """
    r1, r2 = P - p1, P - p2
    cross = np.cross(r1, r2)
    h2 = cross @ cross
    if h2 < 1e-20:
        return np.zeros(3)
    l = p2 - p1
    ll = np.linalg.norm(l)
    cos1 = l @ r1 / (ll * np.linalg.norm(r1))
    cos2 = l @ r2 / (ll * np.linalg.norm(r2))
    return gamma / (4 * np.pi) * (cos1 - cos2) * ll / h2 * cross


# 单元自检：无限长直涡应有 v = Γ/(2πh)，方向由右手定则
_v = filament_velocity(np.array([0.0, 0.0, 1.0]),
                       np.array([-1e4, 0, 0]), np.array([1e4, 0, 0]), 1.0)
assert np.allclose(_v, [0, -1 / (2 * np.pi), 0], atol=1e-6), "涡丝公式自检失败"
print("涡丝公式自检通过 ✓")


def freestream(alpha_deg, V=30.0):
    a = np.radians(alpha_deg)
    return V * np.array([np.cos(a), 0.0, np.sin(a)])


# ══════════════════ 你的三个 TODO ══════════════════

def horseshoe_velocity(P, A, B, gamma=1.0):
    """马蹄涡（单位环量 gamma）在点 P 的诱导速度。

    三段丝（注意每段的起点→终点顺序决定环量方向）：
      腿1：从远下游 (BIG, A_y, 0) → A   （尾涡，空间上在 +x 侧、丝向 -x）
      腿2：A → B                        （附着涡）
      腿3：B → 远下游 (BIG, B_y, 0)     （尾涡，丝向 +x）
    提示：每段调 filament_velocity(P, 起点, 终点, gamma) 再相加；
          远端点 z 坐标取 0（尾涡面与翼面共面）。
    """
    raise NotImplementedError("TODO-1：补全马蹄涡三段丝的诱导速度")


def solve_gamma(alpha_deg, V=30.0):
    """给定迎角，返回 (环量数组 Γ[N], CL)。"""
    V_inf = freestream(alpha_deg, V)

    # TODO-2：组装 AIC 与右端项，解线性方程组
    #   AIC[i, j] = n · (马蹄涡 j 单位环量在 CP_i 的诱导速度)
    #   右端 rhs[i] = -n · V_inf   （不穿透条件 n·(V∞+v)=0）
    #   Γ = np.linalg.solve(AIC, rhs)
    AIC = np.zeros((N, N))   # ← 组装后删掉这行
    rhs = None               # ← 组装后删掉这行
    Gamma = np.zeros(N)      # ← 用 np.linalg.solve 替换

    # TODO-3：Kutta-Joukowski 求 CL
    #   ΔL_j = ρ·V∞·Γ_j·Δy （V∞ 取标量 |V∞|；V∞ 已在 freestream 给出，标量大小 = V）
    #   CL = Σ ΔL / (½ ρ V² S) = 2 Σ Γ_j·Δy / (V·S)
    CL = 0.0                 # ← 替换

    return Gamma, CL


# ══════════════════ 自动判分（勿改） ══════════════════
if __name__ == "__main__":
    alphas = np.array([-4, -2, 0, 2, 4])
    CLs = np.array([solve_gamma(a)[1] for a in alphas])
    slope = np.polyfit(alphas, CLs, 1)[0]
    Gamma5, CL5 = solve_gamma(5.0)
    sym = np.max(np.abs(Gamma5 - Gamma5[::-1])) / np.max(np.abs(Gamma5))

    print(f"\n斜率  CL_α = {slope:.4f} /°   靶 0.0824 ±0.0015   (理论 0.0877, 单格偏差 {slope / 0.0877 - 1:+.1%})")
    print(f"CL(5°)    = {CL5:.4f}   靶 0.412 ±0.010")
    print(f"环量对称  = {sym:.2e}   判据 <1e-8")

    ok = (abs(slope - 0.0824) < 0.0015) and (abs(CL5 - 0.412) < 0.010) and (sym < 1e-8)
    print("\n" + ("🎉 全部 PASS——你自己的 VLM 和 AeroSandbox 对上数了！" if ok else "✗ 未通过，回看对应 TODO。"))

    if ok:  # 附加产出：环量展向分布 vs 椭圆参考
        ell = np.sqrt(np.maximum(1 - (2 * y_mid / b) ** 2, 0))
        ell = ell / np.max(ell) * np.max(np.abs(Gamma5))
        fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
        ax.plot(y_mid, Gamma5, "o-", label="你的最小 VLM")
        ax.plot(y_mid, ell, "--", label="椭圆分布（参考）")
        ax.set_xlabel("展向位置 y (m)")
        ax.set_ylabel("环量 Γ (m²/s)")
        ax.set_title("算例04：α=5° 环量分布——矩形翼 vs 椭圆（这就是 e<1 的来源）")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out = Path(__file__).resolve().parents[1] / "output" / "算例04_最小VLM_环量分布.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out)
        print(f"图已保存: {out}")
