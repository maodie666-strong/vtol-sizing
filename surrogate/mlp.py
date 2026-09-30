"""多层感知机（MLP）代理模型 —— 用 numpy 手写实现。

为什么不用深度学习框架
----------------------
1. 本项目的核心求解器是零依赖的，训练脚本只额外需要 numpy，
   整个项目的依赖面尽可能小；
2. 手写一遍前向与反向传播，你能真正说清网络在算什么 —— 复试答辩全靠这个；
3. 这个问题的规模（10 输入、6 输出、几千样本）根本用不上框架。

网络结构
--------
    输入(10) → 全连接 → ReLU → 全连接 → ReLU → 全连接 → 输出(6)

- 隐层激活用 ReLU：计算简单、正区间梯度恒为 1 不会饱和、收敛快。
- 输出层不加激活（线性）：我们要回归连续的物理量，不是做分类。

反向传播的数学（这部分要能自己推一遍）
--------------------------------------
记第 i 层的输入为 A_{i-1}，权重 W_i、偏置 b_i，预激活 Z_i = A_{i-1} W_i + b_i，
激活 A_i = f(Z_i)，损失 L 取均方误差。

链式法则给出：
    ∂L/∂W_i = A_{i-1}^T · ∂L/∂Z_i
    ∂L/∂b_i = Σ_batch ∂L/∂Z_i
    ∂L/∂A_{i-1} = ∂L/∂Z_i · W_i^T
    ∂L/∂Z_{i-1} = ∂L/∂A_{i-1} ⊙ f'(Z_{i-1})

其中 ⊙ 是逐元素相乘。ReLU 的导数就是 (Z > 0)，所以最后一行退化成一个布尔掩码相乘 ——
这也是 ReLU 计算便宜的原因。

误差从输出层一层层往回传，每一层只需要上一层传回来的 ∂L/∂Z，
这就是"反向传播"这个名字的由来。
"""

from __future__ import annotations

import numpy as np


class MLP:
    """多层感知机回归器（全连接 + ReLU）。"""

    def __init__(self, layer_sizes: list[int], seed: int = 0) -> None:
        """
        参数
        ----
        layer_sizes : 各层神经元数，例如 [10, 64, 64, 6]
                      第一个是输入维度，最后一个是输出维度
        seed        : 随机种子，保证结果可复现
        """
        if len(layer_sizes) < 2:
            raise ValueError("至少要有一个输入层和一个输出层")

        self.layer_sizes = list(layer_sizes)
        self.n_layers = len(layer_sizes) - 1

        rng = np.random.default_rng(seed)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []

        for i in range(self.n_layers):
            fan_in, fan_out = layer_sizes[i], layer_sizes[i + 1]
            # He 初始化：标准差 sqrt(2 / fan_in)
            # 为什么是 2 而不是 1？因为 ReLU 会把一半的输入置零，
            # 方差会被砍掉一半，所以初始化时要乘 2 补回来。
            std = np.sqrt(2.0 / fan_in)
            self.W.append(rng.normal(0.0, std, size=(fan_in, fan_out)))
            self.b.append(np.zeros(fan_out, dtype=float))

        self._reset_optimizer()

    # ---------------------------------------------------------------- 基本量
    def count_parameters(self) -> int:
        """可训练参数总数（权重 + 偏置）。"""
        return int(sum(w.size for w in self.W) + sum(b.size for b in self.b))

    def _reset_optimizer(self) -> None:
        """重置 Adam 的一阶/二阶动量。"""
        self._mW = [np.zeros_like(w) for w in self.W]
        self._vW = [np.zeros_like(w) for w in self.W]
        self._mb = [np.zeros_like(b) for b in self.b]
        self._vb = [np.zeros_like(b) for b in self.b]
        self._step = 0

    # ---------------------------------------------------------------- 前向
    def forward(self, X: np.ndarray, cache: bool = False):
        """前向传播。

        cache=True 时额外返回中间量 (Zs, As)，供反向传播复用 ——
        这就是所谓的"计算图缓存"：反向传播需要前向的中间结果，
        丢掉重算会白白多花一倍时间。
        """
        A = X
        Zs: list[np.ndarray] = []
        As: list[np.ndarray] = [X]

        for i in range(self.n_layers):
            Z = A @ self.W[i] + self.b[i]          # 仿射变换
            Zs.append(Z)
            if i < self.n_layers - 1:
                A = np.maximum(Z, 0.0)             # ReLU
            else:
                A = Z                              # 输出层：线性
            As.append(A)

        if cache:
            return A, (Zs, As)
        return A

    def predict(self, X: np.ndarray) -> np.ndarray:
        """推理（不缓存中间量，省内存）。"""
        return self.forward(X, cache=False)

    # ---------------------------------------------------------------- 反向
    def backward(self, cache, dY: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """反向传播，返回每一层的 (∂L/∂W, ∂L/∂b)。

        dY 是损失对本层输出的梯度 ∂L/∂A_out。
        """
        Zs, As = cache
        grad_W: list[np.ndarray] = [None] * self.n_layers  # type: ignore[list-item]
        grad_b: list[np.ndarray] = [None] * self.n_layers  # type: ignore[list-item]

        dZ = dY
        for i in range(self.n_layers - 1, -1, -1):
            grad_W[i] = As[i].T @ dZ               # ∂L/∂W_i = A_{i-1}^T · ∂L/∂Z_i
            grad_b[i] = dZ.sum(axis=0)             # ∂L/∂b_i = Σ ∂L/∂Z_i

            if i > 0:
                dA = dZ @ self.W[i].T              # 把梯度传回上一层
                dZ = dA * (Zs[i - 1] > 0.0)        # 乘 ReLU 导数（布尔掩码）

        return grad_W, grad_b

    # ---------------------------------------------------------------- 损失
    @staticmethod
    def mse(pred: np.ndarray, target: np.ndarray) -> float:
        """均方误差。"""
        return float(np.mean((pred - target) ** 2))

    # ---------------------------------------------------------------- 优化
    def _adam_step(
        self,
        grad_W: list[np.ndarray],
        grad_b: list[np.ndarray],
        lr: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        """Adam 优化器更新一步。

        Adam 同时维护梯度的一阶矩（动量，抑制震荡）和二阶矩（自适应步长，
        让不同尺度的参数各自用合适的步长）。两个矩都有初始偏差，
        用 1-β^t 做偏差修正。

        相比朴素 SGD，Adam 对学习率不敏感得多，基本不用调参就能收敛 ——
        这对手写实现特别重要，省掉大量调参时间。
        """
        self._step += 1
        bias1 = 1.0 - beta1 ** self._step
        bias2 = 1.0 - beta2 ** self._step

        for i in range(self.n_layers):
            # ---- 权重 ----
            self._mW[i] = beta1 * self._mW[i] + (1.0 - beta1) * grad_W[i]
            self._vW[i] = beta2 * self._vW[i] + (1.0 - beta2) * grad_W[i] ** 2
            m_hat = self._mW[i] / bias1
            v_hat = self._vW[i] / bias2
            self.W[i] -= lr * m_hat / (np.sqrt(v_hat) + eps)

            # ---- 偏置 ----
            self._mb[i] = beta1 * self._mb[i] + (1.0 - beta1) * grad_b[i]
            self._vb[i] = beta2 * self._vb[i] + (1.0 - beta2) * grad_b[i] ** 2
            m_hat_b = self._mb[i] / bias1
            v_hat_b = self._vb[i] / bias2
            self.b[i] -= lr * m_hat_b / (np.sqrt(v_hat_b) + eps)

    def fit(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        epochs: int = 1500,
        batch_size: int = 64,
        lr: float = 1e-3,
        patience: int = 150,
        decay_every: int = 400,
        decay_gamma: float = 0.6,
        lr_min: float = 1e-4,
        verbose: bool = False,
        log_every: int = 100,
    ) -> dict:
        """小批量 + Adam 训练，带学习率衰减与早停。

        学习率衰减：Adam 虽然对学习率不敏感，但训练后期用一个恒定的大步长
        会在最优点附近来回跳、下不去。每隔 decay_every 轮把学习率乘一个
        decay_gamma，让它逐渐"精修"，最后几个百分点的精度就是这么挤出来的。

        早停：如果验证集损失连续 patience 轮没有改善，就停止训练，
        并**回滚到验证损失最低的那一组权重**。这防止网络在训练集上死记硬背
        （过拟合）——过拟合的模型在训练集上误差很小，但换一组没见过的参数就崩。
        """
        n = X.shape[0]
        rng = np.random.default_rng(12345)

        best_val = float("inf")
        best_epoch = 0
        best_W = [w.copy() for w in self.W]
        best_b = [b.copy() for b in self.b]
        history: list[tuple[int, float, float]] = []

        for epoch in range(1, epochs + 1):
            lr_now = max(lr_min, lr * (decay_gamma ** (epoch // decay_every)))

            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start:start + batch_size]
                Xb, Yb = X[idx], Y[idx]

                pred, cache = self.forward(Xb, cache=True)
                # 均方误差对输出的梯度：2(ŷ - y) / (批大小 × 输出维度)
                dY = 2.0 * (pred - Yb) / (Xb.shape[0] * Yb.shape[1])
                grad_W, grad_b = self.backward(cache, dY)
                self._adam_step(grad_W, grad_b, lr_now)

            train_loss = self.mse(self.predict(X), Y)
            val_loss = self.mse(self.predict(X_val), Y_val) if X_val is not None else train_loss
            history.append((epoch, train_loss, val_loss))

            if val_loss < best_val - 1e-12:
                best_val = val_loss
                best_epoch = epoch
                best_W = [w.copy() for w in self.W]
                best_b = [b.copy() for b in self.b]
            elif epoch - best_epoch >= patience:
                if verbose:
                    print(f"  早停于第 {epoch} 轮（最优在第 {best_epoch} 轮）")
                break

            if verbose and epoch % log_every == 0:
                print(f"  epoch {epoch:5d}  lr {lr_now:.2e}  "
                      f"train MSE {train_loss:.3e}  val MSE {val_loss:.3e}")

        # 回滚到验证集最优
        self.W = best_W
        self.b = best_b
        self._reset_optimizer()

        return {
            "best_epoch": best_epoch,
            "best_val_loss": best_val,
            "epochs_run": len(history),
            "history": history,
        }

    # ---------------------------------------------------------------- 序列化
    @staticmethod
    def _round(arr: np.ndarray, sig: int = 7) -> list:
        """把数组转成列表，并保留 sig 位有效数字。

        权重本身只有 float32 级别的精度，存 17 位十进制纯属浪费 ——
        保留 7 位有效数字能让导出的 JSON 体积缩小约一半，而精度损失
        远低于模型自身的误差（MAPE 约 2%）。
        """
        flat = np.asarray(arr, dtype=float).ravel()
        rounded = np.array([float(f"{v:.{sig}g}") for v in flat]).reshape(arr.shape)
        return rounded.tolist()

    def to_dict(self) -> dict:
        """导出为可 JSON 序列化的字典（供浏览器端推理使用）。"""
        return {
            "layer_sizes": self.layer_sizes,
            "weights": [self._round(w) for w in self.W],
            "biases": [self._round(b) for b in self.b],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MLP":
        """从字典恢复网络。"""
        net = cls(payload["layer_sizes"])
        net.W = [np.asarray(w, dtype=float) for w in payload["weights"]]
        net.b = [np.asarray(b, dtype=float) for b in payload["biases"]]
        net._reset_optimizer()
        return net
