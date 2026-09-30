"""代理模型训练入口。

完整流程
--------
  1. 采参数点（拉丁超立方）
  2. 用精确求解器算出对应的性能 —— 等价于"跑 CFD"
  3. 划分训练 / 验证 / 测试集
  4. 标准化（z-score）
  5. 训练 MLP（小批量 + Adam + 早停）
  6. 在测试集上报告精度
  7. 对比推理速度，量化加速比
  8. 导出权重为 JSON，供浏览器端零依赖推理

运行：
    python surrogate/train.py                # 完整流程（会缓存数据集）
    python surrogate/train.py --force-data   # 重新生成数据集
    python surrogate/train.py --n 8000       # 指定样本数
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from surrogate.dataset import build_dataset  # noqa: E402
from surrogate.mlp import MLP  # noqa: E402
from surrogate.space import (  # noqa: E402
    OUTPUT_LABELS,
    OUTPUT_NAMES,
    OUTPUT_SPACE,
    OUTPUT_UNITS,
    PARAM_LABELS,
    PARAM_NAMES,
    PARAM_SPACE,
    PARAM_UNITS,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
WEB_DIR = os.path.join(_ROOT, "vtol-web")


# ---------------------------------------------------------------------- 工具
def rule(title: str = "", width: int = 74) -> None:
    if title:
        print()
        print("=" * width)
        print(f"  {title}")
        print("=" * width)
    else:
        print("-" * width)


def load_or_build(n_samples: int, force: bool = False) -> tuple[np.ndarray, np.ndarray, dict]:
    """加载缓存的数据集，没有则生成。"""
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, f"dataset_{n_samples}.npz")

    if os.path.exists(path) and not force:
        blob = np.load(path, allow_pickle=True)
        X, Y = blob["X"], blob["Y"]
        stats = json.loads(str(blob["stats"]))
        print(f"  已加载缓存数据集：{X.shape[0]} 样本（{path}）")
        return X, Y, stats

    print(f"  开始生成数据集：{n_samples} 样本 ...")
    X, Y, stats = build_dataset(n_samples, seed=20260930)
    np.savez_compressed(path, X=X, Y=Y, stats=json.dumps(stats, ensure_ascii=False))
    return X, Y, stats


def r_squared(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """逐输出量的决定系数 R²。

    R² = 1 - SS_res / SS_tot
    等于 1 表示完美预测；等于 0 表示和"直接猜平均值"一样差；负数表示比猜平均值还差。
    """
    ss_res = ((true - pred) ** 2).sum(axis=0)
    ss_tot = ((true - true.mean(axis=0)) ** 2).sum(axis=0)
    return 1.0 - ss_res / np.maximum(ss_tot, 1e-30)


def mape(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """逐输出量的平均绝对百分比误差（%）。"""
    denom = np.maximum(np.abs(true), 1e-12)
    return np.mean(np.abs(pred - true) / denom, axis=0) * 100.0


def worst_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """逐输出量的最大相对误差（%），反映最坏情况。"""
    denom = np.maximum(np.abs(true), 1e-12)
    return np.max(np.abs(pred - true) / denom, axis=0) * 100.0


# ---------------------------------------------------------------------- 主流程
def main() -> None:
    parser = argparse.ArgumentParser(description="训练 eVTOL 性能代理模型")
    parser.add_argument("--n", type=int, default=4000, help="数据集样本数")
    parser.add_argument("--force-data", action="store_true", help="强制重新生成数据集")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--hidden", type=int, default=64, help="隐层神经元数")
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rule("1. 数据准备")
    X, Y, data_stats = load_or_build(args.n, force=args.force_data)
    print(f"  输入维度 {X.shape[1]}，输出维度 {Y.shape[1]}")
    print(f"  精确求解耗时 {data_stats['generation_seconds']:.2f} s，"
          f"平均 {data_stats['seconds_per_sample'] * 1000:.2f} ms/样本")

    # ---- 划分数据集（固定随机种子，保证可复现） ----
    rng = np.random.default_rng(2026)
    n = X.shape[0]
    order = rng.permutation(n)
    n_test = int(0.15 * n)
    n_val = int(0.15 * n)
    idx_test = order[:n_test]
    idx_val = order[n_test:n_test + n_val]
    idx_train = order[n_test + n_val:]

    X_train, Y_train = X[idx_train], Y[idx_train]
    X_val, Y_val = X[idx_val], Y[idx_val]
    X_test, Y_test = X[idx_test], Y[idx_test]
    print(f"  训练 {X_train.shape[0]} / 验证 {X_val.shape[0]} / 测试 {X_test.shape[0]}")

    # ---- 标准化：用训练集统计量 ----
    # 为什么必须标准化？各输入量纲差异巨大（比能量 400，效率 0.9）。
    # 不标准化的话，梯度会被大数值维度主导，网络根本学不动。
    # 注意统计量只能从训练集算，用全量数据算会"偷看"测试集，导致指标虚高。
    x_mean, x_std = X_train.mean(axis=0), X_train.std(axis=0)
    x_std = np.maximum(x_std, 1e-12)

    # 输出量全部为正（重量、功率、能量、升阻比），先取对数再标准化。
    # 这一步很关键：在对数空间用均方误差，等价于在原空间优化**相对误差**，
    # 正好对应我们真正关心的指标（MAPE）。否则数值大的样本会主导损失，
    # 小功率工况的误差会被完全忽略。
    y_mean = np.log(Y_train).mean(axis=0)
    y_std = np.maximum(np.log(Y_train).std(axis=0), 1e-12)
    log_space = True

    Xtr = (X_train - x_mean) / x_std
    Xva = (X_val - x_mean) / x_std
    Xte = (X_test - x_mean) / x_std
    Ytr = (np.log(Y_train) - y_mean) / y_std
    Yva = (np.log(Y_val) - y_mean) / y_std

    rule("2. 训练神经网络")
    layer_sizes = [X.shape[1], args.hidden, args.hidden, Y.shape[1]]
    net = MLP(layer_sizes, seed=args.seed)
    print(f"  结构 {layer_sizes}，可训练参数 {net.count_parameters()} 个")
    print(f"  优化器 Adam，学习率 {args.lr}，批大小 {args.batch}，早停耐心 150")

    t_train = time.perf_counter()
    info = net.fit(
        Xtr, Ytr, Xva, Yva,
        epochs=args.epochs, batch_size=args.batch, lr=args.lr, verbose=True,
    )
    train_seconds = time.perf_counter() - t_train
    print(f"  训练完成：{info['epochs_run']} 轮，最优第 {info['best_epoch']} 轮，"
          f"耗时 {train_seconds:.2f} s")

    rule("3. 测试集精度")
    pred_std = net.predict(Xte)
    # 反变换：先去标准化，再取指数回到原空间
    pred = np.exp(pred_std * y_std + y_mean)

    r2 = r_squared(pred, Y_test)
    mp = mape(pred, Y_test)
    we = worst_error(pred, Y_test)

    print(f"  {'输出量':<12}{'R²':>10}{'平均误差':>12}{'最大误差':>12}   单位")
    rule()
    for i, name in enumerate(OUTPUT_NAMES):
        print(f"  {OUTPUT_LABELS[name]:<12}{r2[i]:>10.5f}{mp[i]:>11.2f}%{we[i]:>11.2f}%   "
              f"{OUTPUT_UNITS[name]}")
    rule()
    print(f"  平均 R² = {r2.mean():.5f}")

    rule("4. 加速比")
    # 精确求解：随机抽 200 组参数实际跑一遍
    from dataclasses import replace
    from vtol_sizing import VTOLConfig, size

    rng2 = np.random.default_rng(99)
    sample_idx = rng2.choice(n, size=200, replace=False)
    Xs = X[sample_idx]

    t0 = time.perf_counter()
    for row in Xs:
        ov = {nm: (int(round(v)) if nm == "n_rotor" else float(v))
              for nm, v in zip(PARAM_NAMES, row)}
        size(replace(VTOLConfig(), **ov))
    exact_total = time.perf_counter() - t0
    exact_per = exact_total / Xs.shape[0]

    Xs_std = (Xs - x_mean) / x_std
    t0 = time.perf_counter()
    for _ in range(20):
        net.predict(Xs_std)
    surro_total = (time.perf_counter() - t0) / 20.0
    surro_per = surro_total / Xs.shape[0]

    print(f"  精确求解   {exact_per * 1000:>9.3f} ms/次  （200 次共 {exact_total:.3f} s）")
    print(f"  代理模型   {surro_per * 1e6:>9.1f} μs/次  （批量 {Xs.shape[0]} 组）")
    print(f"  加速比     {exact_per / max(surro_per, 1e-12):>9.0f} ×")

    rule("5. 导出权重")
    os.makedirs(WEB_DIR, exist_ok=True)
    payload = {
        "meta": {
            "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_samples": int(data_stats["n_samples"]),
            "n_train": int(X_train.shape[0]),
            "n_val": int(X_val.shape[0]),
            "n_test": int(X_test.shape[0]),
            "layer_sizes": layer_sizes,
            "n_parameters": net.count_parameters(),
            "best_epoch": int(info["best_epoch"]),
            "train_seconds": round(train_seconds, 3),
            "exact_ms_per_call": round(exact_per * 1000.0, 4),
            "surrogate_us_per_call": round(surro_per * 1e6, 2),
            "speedup": round(exact_per / max(surro_per, 1e-12), 1),
        },
        "params": [
            {
                "name": nm,
                "label": PARAM_LABELS[nm],
                "unit": PARAM_UNITS[nm],
                "low": lo,
                "high": hi,
            }
            for nm, lo, hi, _, _ in PARAM_SPACE
        ],
        "outputs": [
            {
                "name": nm,
                "label": OUTPUT_LABELS[nm],
                "unit": OUTPUT_UNITS[nm],
                "scale": sc,
                "r2": round(float(r2[i]), 6),
                "mape_pct": round(float(mp[i]), 4),
                "worst_pct": round(float(we[i]), 4),
            }
            for i, (nm, _, _, sc) in enumerate(OUTPUT_SPACE)
        ],
        "normalization": {
            "log_space": True,
            "x_mean": x_mean.tolist(),
            "x_std": x_std.tolist(),
            "y_mean": y_mean.tolist(),
            "y_std": y_std.tolist(),
        },
        "network": net.to_dict(),
    }

    out_path = os.path.join(WEB_DIR, "surrogate.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    size_kb = os.path.getsize(out_path) / 1024.0
    print(f"  已导出 {out_path}（{size_kb:.1f} KB）")

    # 同时导出一份 .js 版本。
    # 为什么需要两份？浏览器用 file:// 协议直接打开页面时，fetch() 读本地
    # JSON 会被 CORS 拦掉，而 <script src> 不受影响。所以 .js 版本是为了
    # 保证「双击 index.html 就能用」这个离线特性。
    js_path = os.path.join(WEB_DIR, "surrogate-data.js")
    with open(js_path, "w", encoding="utf-8") as fh:
        fh.write("/* 由 surrogate/train.py 自动生成，请勿手动修改。\n")
        fh.write(f" * 训练时间 {payload['meta']['trained_at']}，"
                 f"{payload['meta']['n_samples']} 样本，"
                 f"平均 R² {r2.mean():.5f}\n */\n")
        fh.write("window.VTOL_SURROGATE_DATA = ")
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
        fh.write(";\n")
    js_kb = os.path.getsize(js_path) / 1024.0
    print(f"  已导出 {js_path}（{js_kb:.1f} KB，供浏览器离线加载）")

    stats_path = os.path.join(DATA_DIR, "training_report.json")
    with open(stats_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "data": data_stats,
                "training": {k: v for k, v in info.items() if k != "history"},
                "metrics": {
                    name: {
                        "r2": float(r2[i]),
                        "mape_pct": float(mp[i]),
                        "worst_pct": float(we[i]),
                    }
                    for i, name in enumerate(OUTPUT_NAMES)
                },
                "speedup": {
                    "exact_ms_per_call": exact_per * 1000.0,
                    "surrogate_us_per_call": surro_per * 1e6,
                    "ratio": exact_per / max(surro_per, 1e-12),
                },
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )
    print(f"  训练报告已写入 {stats_path}")

    rule()
    print("  完成。下一步：浏览器端推理与界面切换（见 vtol-web/surrogate.js）")
    rule()


if __name__ == "__main__":
    main()
