"""导出测试点的输入与代理模型预测，用于校验浏览器端推理的一致性。

运行：
    python surrogate/dump_predictions.py

产物：vtol-web/surrogate_test_points.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from surrogate.dataset import build_dataset  # noqa: E402
from surrogate.mlp import MLP  # noqa: E402
from surrogate.space import OUTPUT_NAMES, PARAM_NAMES  # noqa: E402

WEB_DIR = os.path.join(_ROOT, "vtol-web")


def main() -> None:
    # 复用缓存的数据集（若不存在则重新生成一份小规模数据）
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "dataset_5000.npz")
    if os.path.exists(data_path):
        blob = np.load(data_path, allow_pickle=True)
        X = blob["X"]
    else:
        print("未找到缓存数据集，重新生成 500 组 ...")
        X, _, _ = build_dataset(500, seed=4242, verbose=False)

    weights_path = os.path.join(WEB_DIR, "surrogate.json")
    with open(weights_path, encoding="utf-8") as fh:
        payload = json.load(fh)

    net = MLP.from_dict(payload["network"])
    norm = payload["normalization"]
    x_mean = np.asarray(norm["x_mean"])
    x_std = np.asarray(norm["x_std"])
    y_mean = np.asarray(norm["y_mean"])
    y_std = np.asarray(norm["y_std"])
    log_space = bool(norm.get("log_space", False))

    # 固定随机种子，保证可复现
    rng = np.random.default_rng(20260930)
    idx = rng.choice(X.shape[0], size=min(200, X.shape[0]), replace=False)
    Xs = X[idx]

    pred_std = net.predict((Xs - x_mean) / x_std)
    raw = pred_std * y_std + y_mean
    Ys = np.exp(raw) if log_space else raw

    out = {
        "param_names": PARAM_NAMES,
        "output_names": OUTPUT_NAMES,
        "log_space": log_space,
        "cases": [
            {
                "inputs": {nm: float(v) for nm, v in zip(PARAM_NAMES, Xs[i])},
                "expected": {nm: float(v) for nm, v in zip(OUTPUT_NAMES, Ys[i])},
            }
            for i in range(Xs.shape[0])
        ],
    }

    path = os.path.join(WEB_DIR, "surrogate_test_points.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)
    print(f"已导出 {Xs.shape[0]} 个测试点到 {path}")


if __name__ == "__main__":
    main()
