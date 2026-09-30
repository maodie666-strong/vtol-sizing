"""代理模型（surrogate model）子包。

把「慢但准」的精确求解器，替换成「快而够准」的神经网络，
用于参数扫描、实时交互和方案权衡。

模块
----
space     参数空间定义与拉丁超立方采样
dataset   用精确求解器批量生成训练数据
mlp       多层感知机（numpy 手写前向 / 反向传播 + Adam）
train     训练入口，导出浏览器可用的 JSON 权重
"""

from __future__ import annotations

__all__ = ["space", "dataset", "mlp", "train"]
