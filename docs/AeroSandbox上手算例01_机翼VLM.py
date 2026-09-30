"""
上手算例 1：用 AeroSandbox 的涡格法（VLM）算一个直机翼的升阻特性。
对应平台内核里"气动模型 → 涡格法"那一层，跑通它你就知道 VLM 输入输出长什么样。

物理题面：AR=8、无扭转、NACA2412 翼型的矩形机翼，来流 30 m/s，迎角 5°。
心算校核：NACA2412 零升迎角约 -2°，有效迎角约 7°，AR=8 时 CL_alpha≈5.0/rad
→ CL ≈ 7° * 0.088/° ≈ 0.61。程序结果应落在这个附近。
"""
import aerosandbox as asb

wing = asb.Wing(
    symmetric=True,
    xsecs=[
        asb.WingXSec(xyz_le=[0, 0, 0], chord=1.0, airfoil=asb.Airfoil("naca2412")),
        asb.WingXSec(xyz_le=[0, 4, 0], chord=1.0, airfoil=asb.Airfoil("naca2412")),  # 翼尖(半展长4m)
    ],
)
airplane = asb.Airplane(wings=[wing])

op_point = asb.OperatingPoint(velocity=30, alpha=5)
aero = asb.AeroBuildup(airplane=airplane, op_point=op_point).run()

print(f"CL = {aero['CL'][0]:.4f}   (心算约 0.61)")
print(f"CD = {aero['CD'][0]:.4f}")
print(f"Cm = {aero['Cm'][0]:.4f}")
