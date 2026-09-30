"""数据集生成：用精确求解器批量计算「参数 -> 性能」。

这一步在工程上等价于 CFD 计算：用高保真但慢的方法算出样本，
之后代理模型只需要"回忆"这些结果，不用再算一次。

同时记录总耗时 —— 这是后面算加速比的基准。
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import replace

import numpy as np

# 让脚本能直接从仓库根目录或 surrogate/ 目录下运行
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from vtol_sizing import VTOLConfig, size  # noqa: E402

from .space import (  # noqa: E402
    OUTPUT_NAMES,
    OUTPUT_SCALE,
    PARAM_NAMES,
    sample_parameters,
)


def evaluate_point(row: np.ndarray) -> list[float] | None:
    """对一组参数调用精确求解器。

    返回各输出量的数值；若求解器未收敛则返回 None（该样本会被丢弃）。
    """
    overrides: dict[str, float | int] = {}
    for name, value in zip(PARAM_NAMES, row):
        # n_rotor 必须是整数，否则配置本身不合法
        overrides[name] = int(round(value)) if name == "n_rotor" else float(value)

    res = size(replace(VTOLConfig(), **overrides))
    if not res.converged:
        return None

    raw = {
        "mtow": res.mtow,
        "battery_mass": res.mass_breakdown["battery"],
        "hover_power": res.hover_power,
        "installed_power": res.installed_power,
        "required_energy": res.required_energy,
        "lift_to_drag": res.lift_to_drag,
    }
    return [raw[name] * scale for name, scale in zip(OUTPUT_NAMES, OUTPUT_SCALE)]


def build_dataset(
    n_samples: int,
    seed: int = 0,
    max_rounds: int = 10,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """生成 n_samples 个有效样本，返回 (X, Y, 统计信息)。

    采样点里会有少数让求解器发散的组合（例如航程极大而电池比能量极低），
    这些点直接丢弃并补采，保证最终样本数达标。
    """
    rng_state = seed
    xs: list[np.ndarray] = []
    ys: list[list[float]] = []
    collected = 0
    discarded = 0
    t0 = time.perf_counter()

    for round_index in range(max_rounds):
        need = n_samples - collected
        if need <= 0:
            break

        # 多采 15%，抵消丢弃率
        batch = sample_parameters(int(need * 1.15) + 8, seed=rng_state)
        rng_state += 1

        for row in batch:
            if collected >= n_samples:
                break
            # n_rotor 取整后再存回，保证输入与求解时用的配置一致
            row = row.copy()
            idx = PARAM_NAMES.index("n_rotor")
            row[idx] = float(int(round(row[idx])))
            out = evaluate_point(row)
            if out is None:
                discarded += 1
                continue
            xs.append(row)
            ys.append(out)
            collected += 1

        if verbose:
            print(f"  第 {round_index + 1} 轮：累计有效样本 {collected}/{n_samples}，"
                  f"已丢弃 {discarded}")

    elapsed = time.perf_counter() - t0

    if not xs:
        raise RuntimeError("没有任何样本收敛，请检查参数范围是否合理")

    X = np.vstack(xs)
    Y = np.asarray(ys, dtype=float)

    stats = {
        "n_samples": int(X.shape[0]),
        "n_params": int(X.shape[1]),
        "n_outputs": int(Y.shape[1]),
        "discarded": discarded,
        "generation_seconds": elapsed,
        "seconds_per_sample": elapsed / max(X.shape[0], 1),
    }

    if verbose:
        print(f"  数据集合成完成：{stats['n_samples']} 样本，"
              f"{stats['generation_seconds']:.2f} s "
              f"（{stats['seconds_per_sample'] * 1000:.2f} ms/样本）")

    return X, Y, stats
