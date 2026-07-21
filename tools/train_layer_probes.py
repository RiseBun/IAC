#!/usr/bin/env python3
"""Train simple probes on extracted IAC hidden states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, auc, roc_curve, r2_score, f1_score
from sklearn.model_selection import StratifiedKFold, KFold


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train probes on IAC features")
    p.add_argument("--input", required=True)
    p.add_argument("--task", choices=["classification", "regression"], required=True)
    p.add_argument("--target", required=True)
    p.add_argument(
        "--layers",
        nargs="+",
        default=[
            "z_hist",
            "z_fut",
            "z_traj_cons",
            "z_traj_val",
            "z_ego",
            "z_shared",
            "z_validity",
            "future_consistency_evidence",
            "future_traj_geometry_pred",
        ],
    )
    p.add_argument("--output", required=True)
    return p.parse_args()


LAYER_ALIASES = {
    "z_traj_consistency": "z_traj_cons",
    "z_traj_validity": "z_traj_val",
}


def load_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def metrics_clf(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    pred = (y_prob >= 0.5).astype(np.int32)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        "acc": float(accuracy_score(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auc": float(auc(fpr, tpr)),
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input))
    out: Dict[str, Dict[str, float]] = {}

    for layer in args.layers:
        key = LAYER_ALIASES.get(layer, layer)
        X = np.asarray([r[key] for r in rows], dtype=np.float32)
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        y = np.asarray([r[args.target] for r in rows], dtype=np.float32)
        if args.task == "classification":
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            probs = np.zeros_like(y, dtype=np.float32)
            for train_idx, test_idx in skf.split(X, y.astype(int)):
                clf = LogisticRegression(max_iter=2000, n_jobs=1)
                clf.fit(X[train_idx], y[train_idx].astype(int))
                probs[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
            out[layer] = metrics_clf(y.astype(int), probs)
        else:
            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            preds = np.zeros_like(y, dtype=np.float32)
            for train_idx, test_idx in kf.split(X):
                reg = Ridge(alpha=1.0)
                reg.fit(X[train_idx], y[train_idx])
                preds[test_idx] = reg.predict(X[test_idx])
            out[layer] = {
                "r2": float(r2_score(y, preds)),
                "mse": float(np.mean((preds - y) ** 2)),
            }

    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
